import os
import json
import random
import pickle
import argparse
import collections
import numpy as np
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from transformers import T5Tokenizer

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", default=512, type=int, help="max sequence length of model. default to 512.")
parser.add_argument("--pretrain_model_path", default="../transformer_models/t5-base", type=str, help='bert model path')
parser.add_argument("--data_path", default="../dataset/msmarco-data/msmarco-docs-sents.top.300k.json", type=str, help='data path')
parser.add_argument("--docid_path", default=None, type=str, help='docid path')
parser.add_argument("--query_path", default="../dataset/msmarco-data/msmarco-doctrain-queries.tsv", type=str, help='data path')
parser.add_argument("--qrels_path", default="/../dataset/msmarco-data/msmarco-doctrain-qrels.tsv", type=str, help='data path')
parser.add_argument("--output_path", default="../dataset/msmarco-data/train_data/msmarco.top.300k.json", type=str, help='output path')
parser.add_argument("--current_data", default=None, type=str, help="current generating data.")

args = parser.parse_args()

def my_convert_tokens_to_ids(tokens:list, token_to_id:dict): # token_to_id is dict of word:id
    res = []
    for i, t in enumerate(tokens):
        if t in token_to_id:
            res += [token_to_id[t]]
        else:
            res += [token_to_id['<unk>']]
    return res

def my_convert_ids_to_tokens(input_ids:list, id_to_token:dict): # id_to_token is dict of id:word
    res = []
    for i, iid in enumerate(input_ids):
        if iid in id_to_token:
            res += [id_to_token[iid]]
        else:
            print("error!")
    return res

def add_padding(training_instance, tokenizer, id_to_token, token_to_id):
    input_ids = my_convert_tokens_to_ids(training_instance['tokens'], token_to_id)

    new_instance = {
        "input_ids": input_ids,
        "query_id": training_instance["doc_index"],
        "doc_id": training_instance["encoded_docid"],
    }
    return new_instance

def add_docid_to_vocab(doc_file_path):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    vocab = tokenizer.get_vocab()
    new_tokens = []
    with open(doc_file_path) as fin:
        for i, line in tqdm(enumerate(fin), desc='constructing all_documents list'):
            data = json.loads(line)
            sents = data['sents']
            docid = data['docid'].lower()
            new_tokens.append("[{}]".format(docid))

    id_to_token = {vocab[k]:k for k in vocab}
    token_to_id = {id_to_token[k]:k for k in id_to_token}
    for i, doc_id in enumerate(new_tokens):
        token_to_id[doc_id] = i

    return id_to_token, token_to_id, new_tokens, list(vocab.values())

def get_encoded_docid(docid_path, all_docid=None, token_to_id=None):
    encoded_docid = {}
    if docid_path is None:
        for i, doc_id in enumerate(all_docid):
            encoded_docid[doc_id] = str(token_to_id[doc_id])
    else:
        with open(docid_path, "r") as fr:
            for line in fr:
                docid, encode = line.strip().split("\t")
                docid = "[{}]".format(docid.lower().strip('[').strip(']'))
                encoded_docid[docid] = encode
    return encoded_docid

# 测试任务：query -- docid
def gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

    qid_2_query = {}
    docid_2_qid = defaultdict(list)  # 点击了某个doc的queryid
    with open(args.query_path) as fin:
        for line in tqdm(fin):
            qid, query = line.strip().split("\t")
            qid_2_query[qid] = query
    
    count = 0
    with open(args.qrels_path) as fin:
        for line in tqdm(fin):
            qid, _, docid, _ = line.strip().split()
            docid = "[{}]".format(docid.lower())
            if docid not in token_to_id:
                continue
            
            docid_2_qid[docid].append(qid)
            count += 1
    print("total count of clicks: ", count)

    max_num_tokens = args.max_seq_length - 1
    fw = open(args.output_path, "w")

    for docid, qids in tqdm(docid_2_qid.items()):
        for qid in qids:
            query = qid_2_query[qid].lower()
            query_terms = tokenizer.tokenize(query)
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            evaluation_instance = {
                "doc_index":docid,
                "encoded_docid":encoded_docid[docid],
                "tokens": tokens,
            }
            evaluation_instance = add_padding(evaluation_instance, tokenizer, id_to_token, token_to_id)
            fw.write(json.dumps(evaluation_instance, ensure_ascii=False)+"\n")
       
    fw.close()

if __name__ == "__main__":
    id_to_token, token_to_id, all_docid, all_term = add_docid_to_vocab(args.data_path)
    dir_path = os.path.split(args.output_path)[0]
    if not os.path.exists(dir_path):
        os.system(f"mkdir {dir_path}")
    
    if args.current_data == "query_dev":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid)