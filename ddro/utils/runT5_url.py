import os
import time
import torch
import random
import argparse
from utils_url import *
from tqdm import tqdm
import torch.nn as nn
from trie_url import Trie
from eval_url import evaluator
from collections import defaultdict
from torch.utils.data import DataLoader
from T5pretrain import T5ForPretrain
from pretrain_dataset_url import PretrainDataForT5
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
parser.add_argument("--doc_file_path", default="/home/dou/talent/data/msmarco-docs-sents.100k.json", type=str, help='path of origin sent data.')
parser.add_argument("--docid_path", default=None, type=str, help='path of the encoded docid.')
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
            encode = [int(x) for x in encode.split(",")]
            encoded_docids.append(encode)
            encode = ','.join([str(x) for x in encode])
            if encode not in encode_2_docid:
                encode_2_docid[encode] = [docid]
            else:
                encode_2_docid[encode].append(docid)
    return encoded_docids, encode_2_docid
    
def train_model(train_data):
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)

    model = T5ForPretrain(pretrain_model, args)
    
    if args.load_ckpt == "True":
        save_model = load_model(os.path.join(args.load_ckpt_path))
        model.load_state_dict(save_model)
        print("Successfully load checkpoint from ", args.load_ckpt_path)
    
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data)

def train_step(model, train_data):
    with torch.no_grad():
        for key in train_data.keys():
            if key in ["query_id", "doc_id"]:
                continue
            train_data[key] = train_data[key].to(device)
    input_ids = train_data["input_ids"]
    attention_mask = train_data["attention_mask"]
    labels = train_data["docid_labels"]

    loss = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return loss

def fit(model, X_train):
    print("start training...")
    train_dataset = PretrainDataForT5(X_train, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_ratio * int(t_total), num_training_steps=t_total)
    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
        for i, training_data in enumerate(tqdm(train_dataloader)):
            loss = train_step(model, training_data)
            loss = loss.mean()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            if i % args.output_every_n_step == 0:
                localtime = time.asctime(time.localtime(time.time()))
                print(f"{localtime} step: {i}, lr: {args.learning_rate}, loss: {loss.item()}")
                logger.write(f"{localtime} step: {i}, lr: {args.learning_rate}, loss: {loss.item()}\n")
                logger.flush()
            avg_loss += loss.item()
        cnt = len(train_dataset) // args.batch_size + 1
        print("Average loss:{:.6f} ".format(avg_loss / cnt))
        logger.write("Average loss:{:.6f} \n".format(avg_loss / cnt))

        if (epoch+1) % args.save_every_n_epoch == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f"model_{epoch}.pkl"))
            print(f"Save the model in {args.save_path}")
    logger.close()


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

    # def prefix_allowed_tokens_fn(batch_id, sent):
    #     return docid_trie.get(sent.tolist())

    # def prefix_allowed_tokens_fn(batch_id, sent): 
    #     # print("sent:", sent)
    #     outputs = docid_trie.get(sent.tolist())
    #     if len(outputs) == 0:
    #         return [tokenizer.pad_token_id]
    #     return outputs

    def prefix_allowed_tokens_fn(batch_id, sent):
        allowed_tokens = docid_trie.get(sent.tolist())
        if not allowed_tokens:  # Handle unsatisfiable constraints
            
            # Attempt to fallback to the parent prefix
            partial_prefix = sent.tolist()[:-1]
            fallback_tokens = docid_trie.get(partial_prefix) if partial_prefix else []
            
            if fallback_tokens:
                return fallback_tokens
            
            # If no fallback possible, return safe tokens
            return [tokenizer.pad_token_id, tokenizer.eos_token_id]
        
        return allowed_tokens


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
        test_data = load_data(args.test_file_path)  # 加载测试数据集
        test_dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args) # 构建训练集
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        truth, prediction = [], []

        for i, testing_data in tqdm(enumerate(test_dataloader)):
            with torch.no_grad():
                for key in testing_data.keys(): # [input_ids, token_type_ids, attention_mask, mlm_labels, label]
                    if key in ["query_id", "doc_id"]:
                        continue
                    testing_data[key] = testing_data[key].to(device)

            input_ids = testing_data["input_ids"]
            attention_mask = testing_data["attention_mask"]
            qid = testing_data["query_id"]

            if args.use_docid_rank == "False":
                labels = testing_data["docid_labels"] # encode
                truth.extend([[docid2string(docid)] for docid in labels.cpu().numpy().tolist()]) 
            else:
                labels = testing_data["query_id"] # docid
                truth.extend([[docid] for docid in labels])
            # outputs = model.generate(input_ids=input_ids, 
            #                          attention_mask=attention_mask, 
            #                          max_length=args.max_docid_length+1, 
            #                          num_return_sequences=args.num_beams, 
            #                          num_beams=args.num_beams, 
            #                          do_sample=False, 
            #                          prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
            outputs = model.generate(
                input_ids,
                max_length=args.max_docid_length + 1,
                num_return_sequences=args.num_beams,
                num_beams=args.num_beams,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                return_dict_in_generate=True,  # Fix here
                output_scores=True  # Include scores if needed
            )

            for j in range(input_ids.shape[0]):
                doc_rank = []
                batch_output = outputs.sequences[j * args.num_beams:(j + 1) * args.num_beams].cpu().numpy().tolist()
                batch_scores = outputs.sequences_scores[j * args.num_beams:(j + 1) * args.num_beams].cpu().numpy().tolist()
                for docid, scores in zip(batch_output, batch_scores):
                    if args.use_docid_rank == "False":
                        doc_rank.append(docid2string(docid))
                    else:
                        docid_list = encode_2_docid[docid2string(docid)]
                        if len(docid_list) > 1:
                            docid_list.reverse()
                            doc_rank.extend(docid_list)
                        else:
                            doc_rank.extend(docid_list)
                prediction.append(doc_rank)

                
        result = myevaluator.evaluate_ranking(truth, prediction, operation="sorted_generate_doc")
        _mrr5, _mrr10, _mrr, _ndcg10, _ndcg20, _ndcg100, _map20, _p1, _p10, _p20, _p100, _r1, _r5, _r10, _r100, _bleu_1, _bleu_2 = result
        localtime = time.asctime(time.localtime(time.time()))
        print(f"mrr@5:{_mrr5}, mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@5:{_r5}, r@10:{_r10}, r@100:{_r100}, bleu_1:{_bleu_1}, bleu_2:{_bleu_2}")
        logger.write(f"mrr@5:{_mrr5}, mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@5:{_r5}, r@10:{_r10}, r@100:{_r100}, bleu_1:{_bleu_1}, bleu_2:{_bleu_2}\n")

if __name__ == '__main__':
    if args.operation == "training":
        train_data = load_data(args.train_file_path)
        set_seed()
        train_model(train_data) # start training
    
    if args.operation == "testing":
        evaluate_beamsearch() # evaluate the model