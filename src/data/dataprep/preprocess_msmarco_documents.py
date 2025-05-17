
import os
import json
import gzip
import nltk
import random
import logging
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures

# Set random seed
random.seed(42)

# Setup logging
logfile = "logs/msmarco_preprocessing.log"
os.makedirs(os.path.dirname(logfile), exist_ok=True)
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Ensure nltk punkt tokenizer is available
nltk.data.path.append("your_path_to/tmp/nltk_data")  


def process_document(columns):
    docid, url, title, body = columns
    sents = nltk.sent_tokenize(body)
    return docid, {
        "docid": docid,
        "url": url,
        "title": title,
        "body": body,
        "sents": sents
    }


def generate_top_documents(click_counts, scale="300k"):
    input_path = "resources/datasets/processed/msmarco-docs-sents-all.json.gz"
    output_path = f"resources/datasets/processed/msmarco-docs-sents.top.{scale}.json"

    count = 0
    with gzip.open(input_path, "rt") as fr, open(output_path, "w") as fw:
        for line in tqdm(fr, desc=f"Filtering top {scale} docs"):
            docid = json.loads(line)["docid"]
            if click_counts[docid] > 0:
                fw.write(line)
                count += 1

    logging.info(f"Top {scale} docs written: {count}")


def generate_random_documents(scale="300k"):
    input_path = "resources/datasets/processed/msmarco-docs-sents-all.json.gz"
    output_path = f"resources/datasets/processed/msmarco-docs-sents.rand.{scale}.json"
    docids_path = "resources/datasets/processed/msmarco-docids.rand.300k.txt"

    with open(docids_path, "r") as fr:
        selected_docids = set(line.strip() for line in fr)

    count = 0
    with gzip.open(input_path, "rt") as fr, open(output_path, "w") as fw:
        for line in tqdm(fr, desc="Filtering random docs"):
            docid = json.loads(line)["docid"]
            if docid in selected_docids:
                fw.write(line)
                count += 1

    logging.info(f"Random {scale} docs written: {count}")


if __name__ == "__main__":
    doc_file = "resources/datasets/raw/msmarco-docs.tsv.gz"
    qrels_file = "resources/datasets/raw/msmarco-doctrain-qrels.tsv.gz"
    output_file = "resources/datasets/processed/msmarco-docs-sents-all.json.gz"

    doc_contents = {}
    doc_clicks = defaultdict(int)

    with gzip.open(doc_file, "rt") as fin:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_document, line.split("\t"))
                       for line in fin if len(line.split("\t")) == 4]
            for future in tqdm(concurrent.futures.as_completed(futures), desc="Tokenizing documents"):
                docid, content = future.result()
                doc_contents[docid] = content
                doc_clicks[docid] = 0

    logging.info(f"Documents processed: {len(doc_contents)}")

    with gzip.open(qrels_file, "rt") as fr:
        for line in tqdm(fr, desc="Parsing qrels"):
            _, _, docid, _ = line.strip().split()
            doc_clicks[docid] += 1

    sorted_docs = sorted(doc_clicks.items(), key=lambda x: x[1], reverse=True)

    with gzip.open(output_file, "wt", encoding="utf-8") as fout:
        for docid, _ in tqdm(sorted_docs, desc="Saving sorted docs"):
            if docid in doc_contents:
                fout.write(json.dumps(doc_contents[docid]) + "\n")

    logging.info("Main JSONL file written.")

    generate_top_documents(doc_clicks, scale="300k")
    generate_random_documents(scale="300k")

    logging.info("All outputs complete.")
    print(f"Log saved to: {logfile}")
