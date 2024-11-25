import os
import time
import torch
import random
import argparse
from utils import *
from utils import load_model
from utils import set_seed
from tqdm.auto import tqdm
from trie import Trie
from evaluate_per_query import evaluator
from collections import defaultdict
from torch.utils.data import DataLoader
import sys
# Add the root project directory to the Python path
sys.path.append("/gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro")
from pretrain.T5ForPretrain import T5ForPretrain
from pretrain_dataset import PretrainDataForT5
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda:0")
parser = argparse.ArgumentParser()
### training settings
parser.add_argument("--epochs", default=2, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_batch_size", default=25, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--warmup_ratio", default=0, type=float, help="The ratio of warmup steps.")
parser.add_argument("--output_every_n_step", default=25, type=int, help="The steps to output training information.")
parser.add_argument("--save_every_n_epoch", default=25, type=int, help="The epochs to save the trained models.")
parser.add_argument("--operation", default="training", type=str, help="which operation to take, training/testing")
parser.add_argument("--use_docid_rank", default="False", type=str, help="whether to use docid for ranking, or only doc code.")
parser.add_argument("--load_ckpt", default="False", type=str, help="whether to load a trained model checkpoint.")

### path to load data and save models
parser.add_argument("--save_path", default="./model/", type=str, help="The path to save trained models.")
parser.add_argument("--log_path", default="./log/", type=str, help="The path to save log.")
parser.add_argument("--doc_file_path", default="/ivi/ilps/personal/kmekonn/projects/DPO-Enhanced-DSI/data/processed/msmarco-docs-sents.top.300k.json", type=str, help='path of origin sent data.')
parser.add_argument("--docid_path", default="None", type=str, help='path of the encoded docid.')
parser.add_argument("--train_file_path", type=str, help="the path/directory of the training file.")
parser.add_argument("--test_file_path", type=str, help="the path/directory of the testing file.")
parser.add_argument("--pretrain_model_path", type=str, help="path of the pretrained model checkpoint")
parser.add_argument("--load_ckpt_path", default="./model/", type=str, help="The path to load ckpt of a trained model.")
parser.add_argument("--dataset_script_dir", type=str, help="The path of dataset script.")
parser.add_argument("--dataset_cache_dir", type=str, help="The path of dataset cache.")

### hyper-parameters to control the model
parser.add_argument("--add_doc_num", type=int, help="the number of docid to be added.")
parser.add_argument("--max_seq_length", type=int, default=512, help="the max length of input sequences.")
parser.add_argument("--max_docid_length", type=int, default=1, help="the max length of docid sequences.")
parser.add_argument("--use_origin_head", default="False", type=str, help="whether to load the lm_head from the pretrained model.")
parser.add_argument("--num_beams", default=10, type=int, help="the number of beams.")

args = parser.parse_args()
print("args:", args)
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
print("batch_size:", args.batch_size)
print("start a new running with args: ", args)

logger = open(args.log_path, "a")
logger.write("\n")
logger.write(f"start a new running with args: {args}\n")
tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

def load_data(file_path):
    """
        function: load data from the file_path
        args: file_path  -- a directory or a specific file
    """
    if os.path.isfile(file_path):
        fns = [file_path]
    else:
        data_dir = file_path
        fns = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]
    print("file path: ", fns)
    return fns

def load_encoded_docid(docid_path):
    """
        function: load encoded docid data from the docid_path
        return:
            encoded_docids: list of all encoded document identifiers.
            encode_2_docid: dict from encoded document identifiers to original unique id.
    """
    encode_2_docid = {}
    encoded_docids = []
    with open(docid_path, "r") as fr:
        for line in fr:
            docid, encode = line.strip().split("\t")
            docid = docid.lower()
            # since I added padding when I generate the ids , I need to remove the padding
            encode_list = encode.split(",")
            encode= [int(x) for x in encode_list if x not in ["0", "1"]]
            encoded_docids.append(encode)
            encode = ','.join([str(x) for x in encode])
            if encode not in encode_2_docid:
                encode_2_docid[encode] = [docid]
            else:
                encode_2_docid[encode].append(docid)
    return encoded_docids, encode_2_docid


def evaluate_beamsearch():
    '''
        function: Generate the document identifiers with constrained beam search, and evaluate the ranking results.
    '''
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)
    model = T5ForPretrain(pretrain_model, args)
    save_model = load_model(args.save_path)
    model.load_state_dict(save_model)
    model = model.to(device)
    model.eval()
    myevaluator = evaluator()

    encoded_docid, encode_2_docid = load_encoded_docid(args.docid_path)
    docid_trie = Trie([[0] + item for item in encoded_docid])

    def prefix_allowed_tokens_fn(batch_id, sent): 
        outputs = docid_trie.get(sent.tolist())
        # we add the below line to avoid the case that the outputs is empty (mxk)
        if len(outputs) == 0:
            return [tokenizer.pad_token_id]
        return outputs

    def docid2string(docid):
        x_list = []
        for x in docid:
            if x != 0:
                x_list.append(str(x))
            if x == 1:
                break
        return ",".join(x_list)

    if os.path.exists(args.test_file_path):
        localtime = time.asctime(time.localtime(time.time()))
        print(f"Evaluate on the {args.test_file_path}.")
        logger.write(f"{localtime} Evaluate on the {args.test_file_path}.\n")
        test_data = load_data(args.test_file_path)
        test_dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args) # 构建训练集
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        truth, prediction, inputs = [], [], []
    
        for i, testing_data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Evaluating"):
            with torch.no_grad():
                for key in testing_data.keys():
                    if key in ["query_id", "doc_id"]:
                        continue
                    testing_data[key] = testing_data[key].to(device)
            
            input_ids = testing_data["input_ids"]
            if args.use_docid_rank == "False":
                labels = testing_data["docid_labels"] # encoded docid
                truth.extend([[docid2string(docid)] for docid in labels.cpu().numpy().tolist()])
            else:
                labels = testing_data["query_id"] # docid
                truth.extend([[docid] for docid in labels])
            
            inputs.extend(input_ids)

            outputs = model.generate(input_ids, max_length=args.max_docid_length+1, num_return_sequences=args.num_beams, num_beams=args.num_beams, do_sample=False, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)

            for j in range(input_ids.shape[0]):
                query_term = tokenizer.decode(input_ids[j], skip_special_tokens=True).split()
                doc_rank = []
                batch_output = outputs[j*args.num_beams:(j+1)*args.num_beams].cpu().numpy().tolist()
                for docid in batch_output:
                    if args.use_docid_rank == "False":
                        doc_rank.append(docid2string(docid))
                    else:
                        docid_list = encode_2_docid[docid2string(docid)]
                        if len(docid_list) > 1:
                            random.shuffle(docid_list)
                            doc_rank.extend(docid_list)
                        else:
                            doc_rank.extend(docid_list)                           
                prediction.append(doc_rank)
        result_df = myevaluator.evaluate_ranking(truth, prediction)

        # Extracting metrics from the DataFrame's first row
        _mrr10 = result_df['MRR@10'].values.mean()
        _mrr = result_df['MRR'].values.mean()
        _ndcg10 = result_df['NDCG@10'].values.mean()
        _ndcg20 = result_df['NDCG@20'].values.mean()
        _ndcg100 = result_df['NDCG@100'].values.mean()
        _map20 = result_df['MAP@20'].values.mean()
        _p1 = result_df['P@1'].values.mean()
        _p10 = result_df['P@10'].values.mean()
        _p20 = result_df['P@20'].values.mean()
        _p100 = result_df['P@100'].values.mean()
        _r1 = result_df['R@1'].values.mean()
        _r10 = result_df['R@10'].values.mean()
        _r100 = result_df['R@100'].values.mean()
        _r1000 = result_df['R@1000'].values.mean()
        _hit1 = result_df['Hit@1'].values.mean()
        _hit5 = result_df['Hit@5'].values.mean()
        _hit10 = result_df['Hit@10'].values.mean()
        _hit100 = result_df['Hit@100'].values.mean()


        localtime = time.asctime(time.localtime(time.time()))
        print(f"mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100}, r@1000:{_r1000}, hit@1:{_hit1}, hit@5:{_hit5}, hit@10:{_hit10}, hit@100:{_hit100}")
        logger.write(f"mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100}, r@1000:{_r1000}, hit@1:{_hit1}, hit@5:{_hit5}, hit@10:{_hit10}, hit@100:{_hit100}\n")
        
        result_df.to_csv(args.log_path.replace(".log", ".csv"), index=False)
        
if __name__ == '__main__':
    
    if args.operation == "testing":
        evaluate_beamsearch() # evaluate the model