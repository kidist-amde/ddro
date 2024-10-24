import os
import nltk
import json
import random
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import gzip
import logging

"""
This script processes a document dataset, generates subsets based on how many times each document 
was marked as relevant across queries (treated as "click counts"), and writes the top clicked documents 
or a random sample of documents to output files. The input files are assumed to follow the MSMARCO 
dataset format, and it supports customizable output scales.

Note: Since MSMARCO doesn't provide actual click data, the "click count" here refers to how many times 
a document was marked as relevant across different queries in the qrels file (relevance judgments).

"""

# Ensure logging directory exists
logfile = "logs/other_logs/msmarco_processing.log"
if not os.path.exists(os.path.dirname(logfile)):
    os.makedirs(os.path.dirname(logfile))

# Set up logging
logging.basicConfig(
    filename=logfile, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set nltk data path to your specified directory
nltk.data.path.append('/ivi/ilps/personal/kmekonn/projects')

def process_document(cols):
    """Process a single document row to extract details and tokenize the body."""
    docid, url, title, body = cols
    sents = nltk.sent_tokenize(body)  # Tokenization
    return docid, {"docid": docid, "url": url, "title": title, "body": body, "sents": sents}

def generate_top_documents(doc_click_count, scale="300k"):
    """Generate a dataset of top documents based on click counts."""
    logging.info(f"Generating top {scale} dataset.")
    input_path = "resources/datasets/processed/msmarco-data/msmarco-docs-sents-all.json.gz"
    output_path = f"resources/datasets/processed/msmarco-data/msmarco-docs-sents.top.{scale}.json"
    count = 0
    
    with gzip.open(input_path, "rt") as fr, open(output_path, "w") as fw:
        for line in tqdm(fr, desc=f"Filtering top {scale} docs"):
            docid = json.loads(line)["docid"]
            if doc_click_count[docid] <= 0:
                continue
            fw.write(line)
            count += 1
    
    logging.info(f"Count of top {scale}: {count}")

def generate_random_documents(scale="300k"):
    """Generate a random subset of documents."""
    logging.info(f"Generating random {scale} dataset.")
    input_path = "resources/datasets/processed/msmarco-data/msmarco-docs-sents-all.json.gz"
    output_path = f"resources/datasets/processed/msmarco-data/msmarco-docs-sents.rand.{scale}.json"

    rand_300k_docids = set()  # Using a set for faster membership checking
    with open("resources/datasets/processed/msmarco-data/msmarco-docids.rand.300k.txt", "r") as fr:
        for line in fr:
            rand_300k_docids.add(line.strip())

    count = 0    
    with gzip.open(input_path, "rt") as fr, open(output_path, "w") as fw:
        for line in tqdm(fr, desc="Filtering random docs"):
            docid = json.loads(line)["docid"]
            if docid in rand_300k_docids:
                fw.write(line)
                count += 1
    
    logging.info(f"Count of random {scale}: {count}")

if __name__ == '__main__':
    # Paths for input data files
    doc_file_path = "resources/datasets/raw/msmarco-data/msmarco-docs.tsv.gz"
    qrels_train_path = "resources/datasets/raw/msmarco-data/msmarco-doctrain-qrels.tsv.gz"
    fout_path = "resources/datasets/processed/msmarco-data/msmarco-docs-sents-all.json.gz"
  
    id_to_content = {}
    doc_click_count = defaultdict(int)

    # Parse documents to extract content and initialize relevance counts
    with gzip.open(doc_file_path,"rt") as fin:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for line in tqdm(fin, desc="Processing document file"):
                cols = line.split("\t")
                if len(cols) != 4:
                    continue
                futures.append(executor.submit(process_document, cols))

            for future in tqdm(concurrent.futures.as_completed(futures), desc="Gathering results"):
                docid, content = future.result()
                id_to_content[docid] = content
                doc_click_count[docid] = 0

    logging.info(f"Total number of unique documents processed: {len(doc_click_count)}")

    # Count the number of times each document is marked as relevant in the training qrels data
    with gzip.open(qrels_train_path, "rt") as fr:
        for line in tqdm(fr, desc="Processing training qrels"):
            queryid, _, docid, _ = line.strip().split()
            doc_click_count[docid] += 1
    
    logging.info("Finished processing training qrels.")

    # Sort documents by their relevance count (popularity)
    sorted_click_count = sorted(doc_click_count.items(), key=lambda x: x[1], reverse=True)
    logging.info(f"Top 100 documents by relevance count: {sorted_click_count[:100]}")

    # Write processed document content to output file incrementally

    with gzip.open(fout_path, "wt", encoding="utf-8") as fout:  
        for docid, count in tqdm(sorted_click_count, desc="Writing sorted documents"):
            if docid not in id_to_content:
                continue
            fout.write(json.dumps(id_to_content[docid]) + "\n")  

    logging.info(f"Finished writing processed documents to {fout_path}.")

    # Generate datasets of top and random documents
    generate_top_documents(doc_click_count, scale="300k")
    generate_random_documents(scale="300k")

logging.info("Processing completed.")
print(f"Log file saved at: {logfile}")