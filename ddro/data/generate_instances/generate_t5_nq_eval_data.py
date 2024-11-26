import os
import json
import argparse
import collections
from tqdm import tqdm
from collections import defaultdict
from transformers import T5Tokenizer
import gzip

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", default=512, type=int, help="max sequence length of model. default to 512.")
parser.add_argument("--pretrain_model_path", default="/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro/resources/transformer_models/t5-base", type=str, help='bert model path')
parser.add_argument("--data_path", default=None, type=str, help='data path')
parser.add_argument("--docid_path", default="/ivi/ilps/personal/kmekonn/projects/DPO-Enhanced-DSI/WebUltron/dataset/encoded_docid/t5_pq_msmarco.txt", type=str, help='docid path')
parser.add_argument("--query_path", default="../data/raw/msmarco-docdev-queries.tsv.gz", type=str, help='data path')
parser.add_argument("--qrels_path", default="../data/raw/msmarco-docdev-qrels.tsv.gz", type=str, help='data path')
parser.add_argument("--output_path", default="", type=str, help='output path')
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
            sents = data['contents']
            docid = data['id']
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
    print("Total docids in encoded_docid:", len(encoded_docid))
    return encoded_docid

# Generate evaluation instances
def gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

    qid_2_query = {}
    docid_2_qid = defaultdict(list)
    missing_docids = []

    with gzip.open(args.query_path, "rt") as fin:
        for line in tqdm(fin):
            qid, query = line.strip().split("\t")
            qid_2_query[qid] = query

    count = 0
    with gzip.open(args.qrels_path, "rt") as fin:
        for line in tqdm(fin):
            qid, _, docid, _ = line.strip().split()
            # Ensure consistent formatting
            docid = "[{}]".format(docid.lower().strip())
            if docid not in encoded_docid:
                missing_docids.append(docid)
                continue
            docid_2_qid[docid].append(qid)
            count += 1

    print("total count of clicks: ", count)
  

    # Process remaining docids
    max_num_tokens = args.max_seq_length - 1
    fw = open(args.output_path, "w")

    for docid, qids in tqdm(docid_2_qid.items()):
        if docid not in encoded_docid:
            print(f"Warning: docid {docid} not found in encoded_docid. Skipping.")
            continue

        for qid in qids:
            query = qid_2_query[qid].lower()
            query_terms = tokenizer.tokenize(query)
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            evaluation_instance = {
                "doc_index": docid,
                "encoded_docid": encoded_docid[docid],
                "tokens": tokens,
            }
            evaluation_instance = add_padding(evaluation_instance, tokenizer, id_to_token, token_to_id)
            fw.write(json.dumps(evaluation_instance, ensure_ascii=False) + "\n")

    fw.close()

if __name__ == "__main__":
    id_to_token, token_to_id, all_docid, all_term = add_docid_to_vocab(args.data_path)
    dir_path = os.path.split(args.output_path)[0]
    if not os.path.exists(dir_path):
        os.system(f"mkdir {dir_path}")
    
    if args.current_data == "query_dev":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid)
