# filename suggestion: msmarco_data_preparation.py

import os
import json
import argparse
import collections
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import T5Tokenizer
import gzip

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--pretrain_model_path", default="transformer_models/t5-base", type=str)
parser.add_argument("--data_path", default="dataset/msmarco-data/msmarco-docs-sents.top.300k.json", type=str)
parser.add_argument("--docid_path", default=None, type=str)
parser.add_argument("--source_docid_path", default=None, type=str)
parser.add_argument("--query_path", default="../data/raw/msmarco-doctrain-queries.tsv.gz", type=str)
parser.add_argument("--qrels_path", default="../data/raw/msmarco-doctrain-qrels.tsv.gz", type=str)
parser.add_argument("--output_path", default="dataset/msmarco-data/train_data/msmarco.top.300k.json", type=str)
parser.add_argument("--fake_query_path", default="", type=str)
parser.add_argument("--sample_for_one_doc", default=10, type=int)
parser.add_argument("--current_data", default=None, type=str)

args = parser.parse_args()

def map_tokens_to_ids(tokens, token_to_id):
    return [token_to_id.get(t, token_to_id['<unk>']) for t in tokens]

def map_ids_to_tokens(input_ids, id_to_token):
    return [id_to_token[iid] for iid in input_ids if iid in id_to_token]

def prepare_training_instance(training_instance, token_to_id):
    input_ids = map_tokens_to_ids(training_instance['tokens'], token_to_id)
    return {
        "input_ids": input_ids,
        "query_id": training_instance["doc_index"],
        "doc_id": training_instance["encoded_docid"],
    }

def extend_tokenizer_with_docids(doc_file_path):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    vocab = tokenizer.get_vocab()
    new_tokens = []
    with open(doc_file_path) as fin:
        for line in tqdm(fin, desc='Reading documents'):
            docid = json.loads(line)['docid'].lower()
            new_tokens.append(f"[{docid}]")
    id_to_token = {v: k for k, v in vocab.items()}
    token_to_id = {v: k for k, v in id_to_token.items()}
    max_vid = max(id_to_token)
    for i, doc_id in enumerate(new_tokens):
        token_id = max_vid + 1 + i
        id_to_token[token_id] = doc_id
        token_to_id[doc_id] = token_id
    return id_to_token, token_to_id, new_tokens, list(vocab.values())

def load_encoded_docids(docid_path, all_docid=None, token_to_id=None):
    encoded_docid = {}
    if docid_path is None:
        for doc_id in all_docid:
            encoded_docid[doc_id] = str(token_to_id[doc_id])
    else:
        with open(docid_path, "r") as fr:
            for line in fr:
                docid, encode = line.strip().split("\t")
                encoded_docid[f"[{docid.lower().strip('[]')}]"] = encode
    return encoded_docid

def compute_term_idf(doc_file_path):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    vocab = tokenizer.get_vocab()
    doc_count = 0
    idf_dict = defaultdict(int)
    with open(doc_file_path) as fin:
        for line in tqdm(fin, desc='Building IDF dict'):
            doc_count += 1
            doc_item = json.loads(line)
            tokens = tokenizer.tokenize((doc_item['title'] + ' ' + doc_item['body']).lower().strip())
            for term in set(tokens):
                if term in vocab:
                    idf_dict[term] += 1
    return {k: np.log(doc_count / (v + 1)) for k, v in tqdm(idf_dict.items())}

def create_passage_to_docid_samples(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    sample_count = 0
    with open(args.output_path, "w") as fw, open(args.data_path) as fin:
        for line in tqdm(fin, desc='Generating passage instances'):
            doc_item = json.loads(line)
            docid = f"[{doc_item['docid'].lower()}]"
            title_tokens = tokenizer.tokenize(doc_item['title'].lower())
            current_chunk = title_tokens[:]
            sample_for_one_doc = 0
            for sent in doc_item['sents']:
                sent_tokens = tokenizer.tokenize(sent.lower())
                current_chunk += sent_tokens
                if len(current_chunk) >= args.max_seq_length - 1 or sent == doc_item['sents'][-1]:
                    tokens = current_chunk[:args.max_seq_length - 1] + ["</s>"]
                    instance = {
                        "doc_index": docid,
                        "encoded_docid": encoded_docid[docid],
                        "tokens": tokens,
                    }
                    instance = prepare_training_instance(instance, token_to_id)
                    fw.write(json.dumps(instance, ensure_ascii=False) + "\n")
                    sample_count += 1
                    sample_for_one_doc += 1
                    if sample_for_one_doc >= args.sample_for_one_doc:
                        break
                    current_chunk = title_tokens[:]
    print(f"Total passage samples: {sample_count}")

def create_query_to_docid_samples(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    query_dict = {}
    doc_to_query = defaultdict(list)
    with gzip.open(args.query_path, 'rt', encoding='utf-8') as fin:
        for line in tqdm(fin, desc="Reading queries"):
            qid, query = line.strip().split("\t")
            query_dict[qid] = query
    with gzip.open(args.qrels_path, 'rt', encoding='utf-8') as fin:
        for line in tqdm(fin, desc="Reading query relevance"):
            qid, _, docid, _ = line.strip().split()
            docid = f"[{docid.lower()}]"
            if docid in token_to_id:
                doc_to_query[docid].append(qid)
    max_len = args.max_seq_length - 1
    with open(args.output_path, "w") as fw:
        for docid, qids in tqdm(doc_to_query.items(), desc="Generating query instances"):
            for qid in qids:
                tokens = tokenizer.tokenize(query_dict[qid].lower())[:max_len] + ["</s>"]
                instance = {
                    "doc_index": docid,
                    "encoded_docid": encoded_docid[docid],
                    "tokens": tokens,
                }
                instance = prepare_training_instance(instance, token_to_id)
                fw.write(json.dumps(instance, ensure_ascii=False) + "\n")

def create_synthetic_query_to_docid_samples(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    max_tokens = args.max_seq_length - 1
    with open(args.output_path, "w") as fw, open(args.fake_query_path, "r") as fr:
        for line in tqdm(fr, desc="Generating synthetic queries"):
            docid, query = line.strip().split("\t")
            if docid not in token_to_id:
                continue
            tokens = tokenizer.tokenize(query.lower())[:max_tokens] + ["</s>"]
            instance = {
                "doc_index": docid,
                "encoded_docid": encoded_docid[docid],
                "tokens": tokens,
            }
            instance = prepare_training_instance(instance, token_to_id)
            fw.write(json.dumps(instance, ensure_ascii=False) + "\n")

def create_docid_to_docid_samples():
    with open(args.output_path, "w") as fw:
        label_dict = load_encoded_docids(args.docid_path)
        train_dict = load_encoded_docids(args.source_docid_path)
        for docid, encoded in train_dict.items():
            input_ids = [int(i) for i in encoded.split(',')]
            instance = {
                "input_ids": input_ids,
                "query_id": docid,
                "doc_id": label_dict[docid],
            }
            fw.write(json.dumps(instance, ensure_ascii=False) + "\n")

def create_term_sample_to_docid_samples(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    idf_dict = compute_term_idf(args.data_path)
    max_num_tokens = args.max_seq_length - 1
    sample_count = 0
    with open(args.output_path, "w") as fw, open(args.data_path) as fin:
        for line in tqdm(fin, desc='Generating sample term instances'):
            doc_item = json.loads(line)
            docid = f"[{doc_item['docid'].lower()}]"
            body = f"{doc_item['title']} {doc_item['body']}".lower()
            terms = tokenizer.tokenize(body)[:1024]
            tfidf_scores = [(term, terms.count(term) / len(terms) * idf_dict.get(term, 0)) for term in terms if term in idf_dict]
            if len(tfidf_scores) < 10:
                continue
            tfidf_scores.sort(key=lambda x: x[1], reverse=True)
            selected_terms = [term for term, score in tfidf_scores[:max_num_tokens]]
            if len(set(selected_terms)) < 2:
                continue
            tokens = selected_terms + ["</s>"]
            instance = {
                "query_id": docid,
                "doc_id": encoded_docid[docid],
                "input_ids": map_tokens_to_ids(tokens, token_to_id),
            }
            fw.write(json.dumps(instance, ensure_ascii=False) + "\n")
            sample_count += 1
    print(f"Total sampled terms instances: {sample_count}")

if __name__ == "__main__":
    id_to_token, token_to_id, all_docid, _ = extend_tokenizer_with_docids(args.data_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    encoded_docid = load_encoded_docids(args.docid_path, all_docid, token_to_id)

    if args.current_data == "passage":
        create_passage_to_docid_samples(id_to_token, token_to_id, all_docid, encoded_docid)
    elif args.current_data == "query":
        create_query_to_docid_samples(id_to_token, token_to_id, all_docid, encoded_docid)
    elif args.current_data == "fake_query":
        create_synthetic_query_to_docid_samples(id_to_token, token_to_id, all_docid, encoded_docid)
    elif args.current_data == "enhanced_docid":
        create_docid_to_docid_samples()
    elif args.current_data == "sampled_terms":
        create_term_sample_to_docid_samples(id_to_token, token_to_id, all_docid, encoded_docid)