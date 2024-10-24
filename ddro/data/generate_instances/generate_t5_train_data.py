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
parser.add_argument("--pretrain_model_path", default="resources/transformer_models/t5-base", type=str, help='bert model path')
parser.add_argument("--data_path", default="../dataset/msmarco-data/msmarco-docs-sents.top.300k.json", type=str, help='data path')
parser.add_argument("--docid_path", default=None, type=str, help='docid path')
parser.add_argument("--source_docid_path", default=None, type=str, help='train docid path')
parser.add_argument("--query_path", default="../dataset/msmarco-data/msmarco-doctrain-queries.tsv", type=str, help='data path')
parser.add_argument("--qrels_path", default="/../dataset/msmarco-data/msmarco-doctrain-qrels.tsv", type=str, help='data path')
parser.add_argument("--output_path", default="../dataset/msmarco-data/train_data/msmarco.top.300k.json", type=str, help='output path')
parser.add_argument("--fake_query_path", default="", type=str, help='fake query path')
parser.add_argument("--sample_for_one_doc", default=10, type=int, help="max number of passages sampled for one document.")
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
            docid = data['docid'].lower()
            new_tokens.append("[{}]".format(docid))
    id_to_token = {vocab[k]:k for k in vocab}
    token_to_id = {id_to_token[k]:k for k in id_to_token}
    maxvid = max([k for k in id_to_token])
    start_doc_id = maxvid + 1
    for i, doc_id in enumerate(new_tokens):
        id_to_token[start_doc_id+i] = doc_id
        token_to_id[doc_id] = start_doc_id+i

    return id_to_token, token_to_id, new_tokens, list(vocab.values())

def get_encoded_docid(docid_path, all_docid=None, token_to_id=None):
    encoded_docid = {}
    if docid_path is None:
        for i, doc_id in enumerate(all_docid):
            encoded_docid[doc_id] = str(token_to_id[doc_id])  # atomic
    else:
        with open(docid_path, "r") as fr:
            for line in fr:
                docid, encode = line.strip().split("\t")
                docid = "[{}]".format(docid.lower().strip('[').strip(']'))
                encoded_docid[docid] = encode
    return encoded_docid

def build_idf(doc_file_path):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    vocab = tokenizer.get_vocab()

    doc_count = 0
    idf_dict = {key: 0 for key in vocab}
    with open(doc_file_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='building term idf dict'):
            doc_count += 1
            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            title, url, body = doc_item["title"], doc_item["url"], doc_item["body"]
            all_terms = set(tokenizer.tokenize((title + ' ' + body).lstrip().lower()))

            for term in all_terms:
                if term not in idf_dict:
                    continue
                idf_dict[term] += 1
    
    for key in tqdm(idf_dict):
        idf_dict[key] = np.log(doc_count / (idf_dict[key]+1))

    return idf_dict

# 生成各种预训练任务的训练样本
# 任务1.1：passage --> docid
def gen_passage_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    
    sample_count = 0
    fw = open(args.output_path, "w")
    with open(args.data_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='generating samples'):
            max_num_tokens = args.max_seq_length - 1

            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            sents_list = doc_item['sents']
            title = doc_item['title'].lower().strip()
            head_terms = tokenizer.tokenize(title)
            current_chunk = head_terms[:]
            current_length = len(head_terms)
            
            sent_id = 0
            sample_for_one_doc = 0
            while sent_id < len(sents_list):
                sent = sents_list[sent_id].lower()
                sent_terms = tokenizer.tokenize(sent)
                current_chunk += sent_terms
                current_length += len(sent_terms)

                if sent_id == len(sents_list) - 1 or current_length >= max_num_tokens: 
                    tokens = current_chunk[:max_num_tokens] + ["</s>"] # truncate the sequence

                    training_instance = {
                        "doc_index":docid,
                        "encoded_docid":encoded_docid[docid],
                        "tokens": tokens,
                    }
                    training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
                    fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
                    sample_count += 1

                    sample_for_one_doc += 1
                    if sample_for_one_doc >= args.sample_for_one_doc:
                        break
                    
                    current_chunk = head_terms[:]
                    current_length = len(head_terms)
                
                sent_id += 1
    fw.close()
    print("total count of samples: ", sample_count)

# 任务1.2：sampled terms --> docid
def gen_sample_terms_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    print("gen_sample_terms_instance")
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    
    sample_count = 0
    fw = open(args.output_path, "w")
    _, top_or_rand, scale, _ = os.path.split(args.data_path)[1].split(".")

    if os.path.exists(os.path.join(os.path.split(args.output_path)[0], f"sampled_terms.t5_128_1.share.{scale}.json")):
        with open(os.path.join(os.path.split(args.output_path)[0], f"sampled_terms.t5_128_1.share.{scale}.json"), "r") as fr:
            for line in fr:
                line = json.loads(line)
                line["doc_id"] = encoded_docid[line["query_id"]]
                fw.write(json.dumps(line, ensure_ascii=False)+"\n")
                sample_count += 1
    
    else:
        idf_dict = build_idf(args.data_path)
        with open(args.data_path) as fin:
            for doc_index, line in tqdm(enumerate(fin), desc='generating samples'):
                max_num_tokens = args.max_seq_length - 1

                doc_item = json.loads(line)
                docid = doc_item['docid'].lower()
                docid = "[{}]".format(docid)
                title = doc_item['title'].lower().strip()
                body = doc_item['body'].lower().strip()
                all_terms = tokenizer.tokenize(title + ' ' + body)[:1024]
                
                temp_tfidf = []
                all_valid_terms = []
                all_term_tfidf = []
                for term in all_terms:
                    if term not in idf_dict:
                        continue
                    tf_idf = all_terms.count(term) / len(all_terms) * idf_dict[term]
                    temp_tfidf.append((term, tf_idf))
                    all_term_tfidf.append(tf_idf)
                if len(all_term_tfidf) < 10:
                    continue
                tfidf_threshold = sorted(all_term_tfidf, reverse=True)[min(max_num_tokens, len(all_term_tfidf))-1]
                for idx, (term, tf_idf) in enumerate(temp_tfidf):
                    if tf_idf >= tfidf_threshold:
                        all_valid_terms.append(term)

                if len(set(all_valid_terms)) < 2:
                    continue

                tokens = all_valid_terms[:max_num_tokens] + ["</s>"]
                training_instance = {
                    "query_id":docid,
                    "doc_id":encoded_docid[docid],
                    "input_ids": my_convert_tokens_to_ids(tokens, token_to_id),
                }

                fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
                sample_count += 1

    fw.close()
    if not os.path.exists(os.path.join(os.path.split(args.output_path)[0], f"sampled_terms.t5_128_1.share.{scale}.json")):
        target_path = os.path.join(os.path.split(args.output_path)[0], f"sampled_terms.t5_128_1.share.{scale}.json")
        os.system(f"cp {args.output_path}, {target_path}")
    print("total count of samples: ", sample_count)

# 任务1.3：source docid --> target docid
def gen_enhanced_docid_instance(label_filename, train_filename):
    fw = open(args.output_path, "w")
    label_dict = get_encoded_docid(label_filename)
    train_dict = get_encoded_docid(train_filename)
    for docid, encoded in train_dict.items():
        input_ids = [int(item) for item in encoded.split(',')]
        training_instance = {"input_ids": input_ids, "query_id": docid, "doc_id": label_dict[docid]}
        fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
    fw.close()
 
# 任务2：pseudo query --> docid
def gen_fake_query_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
         
    # 从ms-marco数据集中检索出点击了docid的对应query 
    fw = open(args.output_path, "w")   
    max_num_tokens = args.max_seq_length - 1
        
    _, top_or_rand, scale, _ = os.path.split(args.data_path)[1].split(".")
    
    with open(args.fake_query_path, "r") as fr:
        for line in tqdm(fr, desc="load all fake queries"):
            docid, query = line.strip("\n").split("\t")
            if docid not in token_to_id:
                continue

            query_terms = tokenizer.tokenize(query.lower())
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            training_instance = {
                "doc_index":docid,
                "encoded_docid":encoded_docid[docid],
                "tokens": tokens,
            }
            training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")

    fw.close()

# 任务3：query --> docid,  finetune
def gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
         
    # 从ms-marco数据集中检索出点击了docid的对应query 
    fw = open(args.output_path, "w")
    qid_2_query = {}
    docid_2_qid = defaultdict(list)  # 点击了某个doc的queryid
    with open(args.query_path) as fin:
        for line in tqdm(fin, desc="reading all queries"):
            qid, query = line.strip().split("\t")
            qid_2_query[qid] = query
    
    count = 0
    with open(args.qrels_path) as fin:
        for line in tqdm(fin, desc="reading all click samples"):
            qid, _, docid, _ = line.strip().split()
            
            docid = "[{}]".format(docid.lower())
            if docid not in token_to_id:
                continue
            
            docid_2_qid[docid].append(qid)
            count += 1
    print("total count of clicks: ", count)
    
    max_num_tokens = args.max_seq_length - 1
    
    for docid, qids in tqdm(docid_2_qid.items(), desc="constructing click samples"):
        for qid in qids:
            query = qid_2_query[qid].lower()
            query_terms = tokenizer.tokenize(query)
            tokens = query_terms[:max_num_tokens] + ["</s>"]

            training_instance = {
                "doc_index":docid,
                "encoded_docid":encoded_docid[docid],
                "tokens": tokens,
            }
            training_instance = add_padding(training_instance, tokenizer, id_to_token, token_to_id)
            fw.write(json.dumps(training_instance, ensure_ascii=False)+"\n")
      
    fw.close()

if __name__ == "__main__":
    id_to_token, token_to_id, all_docid, all_term = add_docid_to_vocab(args.data_path)
    dir_path = os.path.split(args.output_path)[0]
    if not os.path.exists(dir_path):
        os.system(f"mkdir {dir_path}")
    
    if args.current_data == "passage":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_passage_instance(id_to_token, token_to_id, all_docid, encoded_docid)

    if args.current_data == "sampled_terms":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_sample_terms_instance(id_to_token, token_to_id, all_docid, encoded_docid)
    
    if args.current_data == "enhanced_docid":
        gen_enhanced_docid_instance(args.docid_path, args.source_docid_path)

    if args.current_data == "fake_query":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_fake_query_instance(id_to_token, token_to_id, all_docid, encoded_docid)
    
    if args.current_data == "query":
        encoded_docid = get_encoded_docid(args.docid_path, all_docid, token_to_id)
        gen_query_instance(id_to_token, token_to_id, all_docid, encoded_docid)