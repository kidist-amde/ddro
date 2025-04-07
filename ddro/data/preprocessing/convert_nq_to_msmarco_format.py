import argparse
import gzip
import json
import os
import math
import pandas as pd
from tqdm.auto import tqdm


def clean_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def load_nq_dataset(file_path, total_docs=None, start_qid=0):
    data = []
    with gzip.open(file_path, 'rt') as f:
        print(f"Loading NQ data from {file_path}")
        columns = ['query', 'id', 'long_answer', 'short_answer', 'title', 
                   'abstract', 'content', 'document_url', 'doc_tac', 'language']
        df = pd.read_csv(f, sep='\t', names=columns)
        for idx, row in tqdm(df.iterrows(), total=total_docs, desc="Loading NQ data"):
            data.append({
                "query_id": start_qid + idx,
                "query": row['query'],
                "id": row['id'],
                "long_answer": row['long_answer'],
                "short_answer": row['short_answer'],
                "title": row['title'],
                "abstract": row['abstract'],
                "content": row['content'],
                "url": row['document_url'],
                "doc_tac": row['doc_tac'],
                "language": row['language']
            })
    return data


def get_queries(data):
    return [{"qid": entry["query_id"], "query": entry["query"]} for entry in data]


def get_qrels(data):
    return [{"qid": entry["query_id"], "0": 0, "docid": entry["id"], "rel": 1} for entry in data]


def save_tsv_gz(data, file_path, fields):
    with gzip.open(file_path, "wt") as f:
        for row in data:
            f.write("\t".join(str(row[field]) for field in fields) + "\n")


def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


def create_msmarco_format(train_file, dev_file, output_dir):
    train_data = load_nq_dataset(train_file, total_docs=87791, start_qid=0)
    dev_data = load_nq_dataset(dev_file, total_docs=21948, start_qid=len(train_data))

    queries_train = get_queries(train_data)
    queries_dev = get_queries(dev_data)
    qrels_train = get_qrels(train_data)
    qrels_dev = get_qrels(dev_data)

    os.makedirs(output_dir, exist_ok=True)
    save_tsv_gz(queries_train, os.path.join(output_dir, "nq_queries_train.tsv.gz"), ["qid", "query"])
    save_tsv_gz(queries_dev, os.path.join(output_dir, "nq_queries_dev.tsv.gz"), ["qid", "query"])
    save_tsv_gz(qrels_train, os.path.join(output_dir, "nq_qrels_train.tsv.gz"), ["qid", "0", "docid", "rel"])
    save_tsv_gz(qrels_dev, os.path.join(output_dir, "nq_qrels_dev.tsv.gz"), ["qid", "0", "docid", "rel"])

    json_dir = os.path.join(output_dir, "nq-merged-json")
    os.makedirs(json_dir, exist_ok=True)

    docs_json = [{
        "id": entry["id"],
        "url": clean_value(entry["url"]),
        "title": clean_value(entry["title"]),
        "contents": clean_value(entry["content"]),
        "doc_tac": clean_value(entry["doc_tac"])
    } for entry in train_data + dev_data]
    save_jsonl(docs_json, os.path.join(json_dir, "nq-docs-sents.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NQ dataset to MSMARCO-style format")
    parser.add_argument('--nq_train_file', required=True, help="Path to the gzipped NQ training file")
    parser.add_argument('--nq_dev_file', required=True, help="Path to the gzipped NQ dev file")
    parser.add_argument('--output_dir', required=True, help="Directory to save the outputs")
    args = parser.parse_args()

    create_msmarco_format(args.nq_train_file, args.nq_dev_file, args.output_dir)