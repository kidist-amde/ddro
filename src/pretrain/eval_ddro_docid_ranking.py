import os
import sys
import torch
import random
import argparse
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Add src to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.utils import load_model
from utils.trie import Trie
from utils.evaluate import evaluator
from pretrain.T5ForPretrain import T5ForPretrain
from utils.pretrain_dataset import PretrainDataForT5

device = torch.device("cuda:0")
parser = argparse.ArgumentParser()
parser.add_argument("--per_gpu_batch_size", default=25, type=int)
parser.add_argument("--save_path", default="./model/", type=str)
parser.add_argument("--log_path", default="./log/", type=str)
parser.add_argument("--docid_path", default="None", type=str)
parser.add_argument("--test_file_path", type=str)
parser.add_argument("--pretrain_model_path", type=str)
parser.add_argument("--dataset_script_dir", type=str)
parser.add_argument("--dataset_cache_dir", type=str)
parser.add_argument("--add_doc_num", type=int)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--max_docid_length", type=int, default=1)
parser.add_argument("--use_origin_head", default="False", type=str)
parser.add_argument("--num_beams", default=10, type=int)
parser.add_argument("--use_docid_rank", default="False", type=str)


args = parser.parse_args()
print("args:", args)
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
print("batch_size:", args.batch_size)

logger = open(args.log_path, "a")
logger.write("\n")
logger.write(f"start a new running with args: {args}\n")
tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

def load_data(file_path):
    if os.path.isfile(file_path):
        fns = [file_path]
    else:
        fns = [os.path.join(file_path, fn) for fn in os.listdir(file_path)]
    print("file path: ", fns)
    return fns


# FOR THE NQ DATA THE DOC ID MUST BE UPLOADED WITH THIS FUNCTION for all ids 

# def load_encoded_docid(docid_path):
#     """
#         function: load encoded docid data from the docid_path
#         return:
#             encoded_docids: list of all encoded document identifiers.
#             encode_2_docid: dict from encoded document identifiers to original unique id.
#     """
#     encode_2_docid = {}
#     encoded_docids = []
#     with open(docid_path, "r") as fr:
#         for line in fr:
#             docid, encode = line.strip().split("\t")
#             docid = docid.lower()
#             # since I added padding when I generate the ids , I need to remove the padding
#             encode_list = encode.split(",")
#             encode= [int(x) for x in encode_list if x not in ["0", "1"]]
#             encoded_docids.append(encode)
#             encode = ','.join([str(x) for x in encode])
#             if encode not in encode_2_docid:
#                 encode_2_docid[encode] = [docid]
#             else:
#                 encode_2_docid[encode].append(docid)
#     return encoded_docids, encode_2_docid

# FOR MSMARCO DATA THE DOC ID MUST BE UPLOADED WITH THIS FUNCTION
def load_encoded_docid(docid_path):
    encode_2_docid = {}
    encoded_docids = []
    with open(docid_path, "r") as fr:
        for line in fr:
            docid, encode = line.strip().split("\t")
            docid = docid.lower()
            encode = [int(x) for x in encode.split(",")]
            encoded_docids.append(encode)
            encode = ','.join([str(x) for x in encode])
            if encode not in encode_2_docid:
                encode_2_docid[encode] = [docid]
            else:
                encode_2_docid[encode].append(docid)
    return encoded_docids, encode_2_docid



def evaluate_beamsearch():
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
        return outputs if outputs else [tokenizer.pad_token_id]

    def docid2string(docid):
        """Drop 0s, keep a single EOS(1), trim everything after first 1."""
        seq = []
        for x in docid:
            if x == 0:
                continue
            if x == 1:
                seq.append(1)
                break
            seq.append(x)
        return ",".join(map(str, seq))



    if os.path.exists(args.test_file_path):
        print(f"Evaluate on the {args.test_file_path}.")
        logger.write(f"Evaluate on the {args.test_file_path}.\n")
        test_data = load_data(args.test_file_path)
        test_dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        truth, prediction, inputs = [], [], []
        print("Start evaluating... ")
        for i, testing_data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Evaluating"):
            with torch.no_grad():
                for key in testing_data:
                    if key not in ["query_id", "doc_id"]:
                        testing_data[key] = testing_data[key].to(device)

            input_ids = testing_data["input_ids"]
            if args.use_docid_rank == "False":
                labels = testing_data["docid_labels"]
                truth.extend([[docid2string(docid)] for docid in labels.cpu().numpy().tolist()])
            else:
                labels = testing_data["query_id"]
                truth.extend([[docid] for docid in labels])

            inputs.extend(input_ids)

            outputs = model.generate(
                input_ids,
                max_length=args.max_docid_length+1,
                num_return_sequences=args.num_beams,
                num_beams=args.num_beams,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )

            for j in range(input_ids.shape[0]):
                doc_rank = []
                batch_output = outputs[j*args.num_beams:(j+1)*args.num_beams].cpu().numpy().tolist()
                for docid in batch_output:
                    if args.use_docid_rank == "False":
                        doc_rank.append(docid2string(docid))
                    else:
                        docid_list = encode_2_docid[docid2string(docid)]
                        random.shuffle(docid_list)
                        doc_rank.extend(docid_list)
                prediction.append(doc_rank)

        result_df = myevaluator.evaluate_ranking(truth, prediction)
        _mrr10 = result_df['MRR@10'].values.mean()
        _mrr = result_df['MRR'].values.mean()
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

        print(f"mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100}, r@1000:{_r1000}, hit@1:{_hit1}, hit@5:{_hit5}, hit@10:{_hit10}, hit@100:{_hit100}")
        logger.write(f"mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100}, r@1000:{_r1000}, hit@1:{_hit1}, hit@5:{_hit5}, hit@10:{_hit10}, hit@100:{_hit100}\n")

        result_df.to_csv(args.log_path.replace(".log", ".csv"), index=False)

if __name__ == '__main__':
    evaluate_beamsearch()
