import argparse
import gzip
import json
import os  # Added to handle directory creation
from tqdm.auto import tqdm
import pandas as pd
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nq_train_file', default="resources/datasets/processed/nq-data/nq_train.tsv.gz", type=str, required=True, help="Path to the nq_train_modified.tsv.gz file")
    parser.add_argument('--nq_dev_file', type=str, default="resources/datasets/processed/nq-data/nq_dev.tsv.gz", required=True, help="Path to the nq_dev_modified.tsv.gz file")
    parser.add_argument('--output_dir', type=str, required=True, default="resources/datasets/processed/nq-data", help="Path to the output directory")
    args = parser.parse_args()

    create_msmarco_resumbling_dataset(args.nq_train_file, args.nq_dev_file, args.output_dir)


def clean_value(value):
    """Utility function to clean up NaN and None values."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value

def load_nq_dataset(file_path, total_docs=None, start_qid=0):
    """
    Load the NQ dataset from a gzipped TSV file and assign unique query IDs.

    Args:
        file_path (str): Path to the gzipped TSV file.
        total_docs (int): Total number of documents (for progress tracking).
        start_qid (int): The starting query ID (for sequential unique query IDs).

    Returns:
        List[Dict]: A list of dictionaries containing the dataset.
    """
    data = []
    with gzip.open(file_path, 'rt') as f:
        print(f"Loading NQ data from {file_path}")
        columns = ['query', 'id', 'long_answer', 'short_answer', 'title', 
                   'abstract', 'content', 'document_url', 'doc_tac', 'language']
        df = pd.read_csv(f, sep='\t', names=columns)
        
        for idx, row in tqdm(df.iterrows(), total=total_docs, desc="Loading NQ data"):
            doc = {
                "query_id": start_qid + idx,  # Generate query_id based on index + start_qid
                "query": row['query'],
                "id": row['id'],  # Keep document ID as it is
                "long_answer": row['long_answer'],
                "short_answer": row['short_answer'],
                "title": row['title'],
                "abstract": row['abstract'],
                "content": row['content'],
                "url": row['document_url'],
                "doc_tac": row['doc_tac'],
                "language": row['language']
            }
            data.append(doc)
    return data


def get_nq_queries(nq_data):
    """
    Create a list of queries from the NQ dataset.

    Args:
        nq_data (list): List of document dictionaries containing the queries.

    Returns:
        List[Dict]: A list of dictionaries containing 'qid' and 'query' for each query.
    """
    return [
        {
            "qid": data["query_id"],  # Use the generated query_id
            "query": data["query"]
        }
        for data in nq_data
    ]


def get_nq_qrels(nq_data):
    """
    Create a list of qrels (query relevance judgments) from the NQ dataset.

    Args:
        nq_data (list): List of document dictionaries.

    Returns:
        List[Dict]: A list of dictionaries containing query-document relevance pairs.
    """
    return [
        {
            "qid": data["query_id"],  # Use the generated query_id
            "0": 0,
            "docid": data["id"],  # Use the document's original ID
            "rel": 1
        }
        for data in nq_data
    ]


def save_queries_gzip_tsv(data, file_path):
    """
    Save the query data to a gzipped TSV file.

    Args:
        data (list): List of dictionaries containing query data.
        file_path (str): Path to the output file.
    """
    with gzip.open(file_path, "wt") as f:
        for line in data:
            f.write(f"{line['qid']}\t{line['query']}\n")


def save_qrels_gzip_tsv(data, file_path):
    """
    Save the qrels data to a gzipped TSV file.

    Args:
        data (list): List of dictionaries containing qrels data.
        file_path (str): Path to the output file.
    """
    with gzip.open(file_path, "wt") as f:
        for line in data:
            f.write(f"{line['qid']}\t0\t{line['docid']}\t1\n")


def create_msmarco_resumbling_dataset(train_file_path, dev_file_path, output_dir):
    # Load training and dev data, starting query IDs for training set is 0
    traning_data = load_nq_dataset(train_file_path, total_docs=87791, start_qid=0)
    
    # Start query IDs for dev set after the training set
    dev_data = load_nq_dataset(dev_file_path, total_docs=21948, start_qid=len(traning_data))

    # Prepare queries and qrels
    traning_queries = get_nq_queries(traning_data)
    print(f"Length of traning_queries: {len(traning_queries)}")
    dev_queries = get_nq_queries(dev_data)
    print(f"Length of dev_queries: {len(dev_queries)}")
    train_qrels = get_nq_qrels(traning_data)
    print(f"Length of train_qrels: {len(train_qrels)}")
    dev_qrels = get_nq_qrels(dev_data)
    print(f"Length of dev_qrels: {len(dev_qrels)}")

    # Save queries and qrels to gzipped TSV files
    save_queries_gzip_tsv(traning_queries, f"{output_dir}/nq_queries_train.tsv.gz")
    save_queries_gzip_tsv(dev_queries, f"{output_dir}/nq_queries_dev.tsv.gz")
    save_qrels_gzip_tsv(train_qrels, f"{output_dir}/nq_qrels_train.tsv.gz")
    save_qrels_gzip_tsv(dev_qrels, f"{output_dir}/nq_qrels_dev.tsv.gz")

    # Specify the directory where the final JSON will be saved
    json_output_dir = "resources/datasets/processed/nq-data/nq-merged-json"
    os.makedirs(json_output_dir, exist_ok=True)

    # Save NQ sentences in JSONL format in the nq-merged-json folder
    print("Saving NQ sentences in JSONL format...")


    nq_sents = [
        {
            "id": data["id"],  
            "url": clean_value(data["url"]),
            "title": clean_value(data["title"]),
            "contents": clean_value(data["content"]),  # Ensures no NaN or None values
            "doc_tac": clean_value(data["doc_tac"])
        }
        for data in traning_data + dev_data
]


    print(f"Length of nq_sents: {len(nq_sents)}")
    with open(f"{json_output_dir}/nq-docs-sents.json", "w") as f:
        for line in nq_sents:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
