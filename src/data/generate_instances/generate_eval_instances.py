import os
import json
import gzip
import argparse
import collections
from tqdm import tqdm
from collections import defaultdict
from transformers import T5Tokenizer

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", default=512, type=int, help="Max sequence length. Default is 512.")
parser.add_argument("--pretrain_model_path", default="transformer_models/t5-base", type=str, help="Pretrained model path")
parser.add_argument("--data_path", type=str, required=True, help="Path to input JSONL data file")
parser.add_argument("--docid_path", default="output/encoded_docid.txt", type=str, help="Path to encoded docid file")
parser.add_argument("--query_path", default="data/queries.tsv.gz", type=str, help="Path to queries file")
parser.add_argument("--qrels_path", default="data/qrels.tsv.gz", type=str, help="Path to qrels file")
parser.add_argument("--output_path", default="output/instances.jsonl", type=str, help="Output file path")
parser.add_argument("--current_data", default=None, type=str, help="Task type, e.g., 'query_dev'")
args = parser.parse_args()

def my_convert_tokens_to_ids(tokens, token_to_id):
    return [token_to_id.get(t, token_to_id['<unk>']) for t in tokens]

def my_convert_ids_to_tokens(input_ids, id_to_token):
    return [id_to_token.get(iid, "<unk>") for iid in input_ids]

def add_padding(training_instance, tokenizer, id_to_token, token_to_id):
    input_ids = my_convert_tokens_to_ids(training_instance['tokens'], token_to_id)
    return {
        "input_ids": input_ids,
        "query_id": training_instance["doc_index"],
        "doc_id": training_instance["encoded_docid"]
    }

def add_docid_to_vocab(doc_file_path):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    vocab = tokenizer.get_vocab()
    new_tokens = []
    with open(doc_file_path) as fin:
        for line in tqdm(fin, desc='Loading document IDs'):
            data = json.loads(line)
            docid = data['docid'].lower()
            new_tokens.append(f"[{docid}]")

    id_to_token = {v: k for k, v in vocab.items()}
    token_to_id = dict(vocab)
    for i, doc_id in enumerate(new_tokens):
        token_to_id[doc_id] = i

    return id_to_token, token_to_id, new_tokens, list(vocab.values())

def get_encoded_docid(docid_path, all_docid=None, token_to_id=None):
    encoded_docid = {}
    if docid_path is None:
        for doc_id in all_docid:
            encoded_docid[doc_id] = str(token_to_id[doc_id])
    else:
        with open(docid_path, "r") as fr:
            for line in fr:
                docid, encode = line.strip().split("\t")
                docid = f"[{docid.lower().strip('[]')}]"
                encoded_docid[docid] = encode
    return encoded_docid

def gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    qid_2_query = {}
    docid_2_qid = defaultdict(list)

    with gzip.open(args.query_path, "rt") as fin:
        for line in tqdm(fin, desc="Reading queries"):
            qid, query = line.strip().split("\t")
            qid_2_query[qid] = query

    with gzip.open(args.qrels_path, "rt") as fin:
        for line in tqdm(fin, desc="Reading qrels"):
            qid, _, docid, _ = line.strip().split()
            docid = f"[{docid.lower()}]"
            if docid in token_to_id:
                docid_2_qid[docid].append(qid)

    print("Total clicked pairs:", sum(len(qids) for qids in docid_2_qid.values()))

    max_num_tokens = args.max_seq_length - 1
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, "w") as fw:
        for docid, qids in tqdm(docid_2_qid.items(), desc="Writing instances"):
            if docid not in encoded_docid:
                print(f"Warning: Missing encoded docid for {docid}")
                continue

            for qid in qids:
                query = qid_2_query[qid].lower()
                tokens = tokenizer.tokenize(query)[:max_num_tokens] + ["</s>"]
                instance = {
                    "doc_index": docid,
                    "encoded_docid": encoded_docid[docid],
                    "tokens": tokens
                }
                padded = add_padding(instance, tokenizer, id_to_token, token_to_id)
                fw.write(json.dumps(padded, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    id_to_token, token_to_id, all_docid, all_term = add_docid_to_vocab(args.data_path)
    if args.current_data == "query_dev":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid)
