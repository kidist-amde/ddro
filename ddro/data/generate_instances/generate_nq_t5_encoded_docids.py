import re
import numpy as np
import argparse
import torch  # For GPU handling
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nanopq  # For product quantization (pq)
import gzip
import pandas as pd
import json

# Argument parser for command-line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="pq", type=str, help="docid method: atomic/pq/url")
parser.add_argument("--input_embeddings", default="./resources/datasets/processed/nq-data/doc_embedding/t5_512_nq_doc_embedding.txt", type=str, help="Path to the precomputed T5 embeddings")
parser.add_argument("--sub_space", default=24, type=int, help="The number of sub-spaces for PQ on a 768-dim vector")
parser.add_argument("--cluster_num", default=256, type=int, help="The number of clusters in each sub-space for PQ")
parser.add_argument("--output_path", default="./resources/datasets/processed/nq-data/semantic_doc_ids.txt", type=str, help="Path to save the generated document IDs")
parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for processing embeddings")
parser.add_argument("--nq_data_path", default="ddro/resources/datasets/raw/nq-data/nq_merged.tsv.gz", type=str, help="Path to the NQ data file")
parser.add_argument("--pretrain_model_path", default="resources/transformer_models/t5-base", type=str, help="Pre-trained model path for URL tokenization")
parser.add_argument("--summary_path", default="path/to/msmarco-llama3_summaries.json", type=str, help="Path to pre-generated summaries JSON")
args = parser.parse_args()

# Function to load precomputed document embeddings
def load_doc_embeddings(input_path):
    docid_2_idx, idx_2_docid = {}, {}
    doc_embeddings = []

    with open(input_path, "r") as fr:
        for line in tqdm(fr, desc="Loading document embeddings..."):
            docid, demb = line.strip().split('\t')
            d_embedding = [float(x) for x in demb.split(',')]

            docid_2_idx[docid] = len(docid_2_idx)
            idx_2_docid[docid_2_idx[docid]] = docid
            doc_embeddings.append(d_embedding)
    print(f"Loaded {len(docid_2_idx)} document embeddings!")
    return docid_2_idx, idx_2_docid, np.array(doc_embeddings, dtype=np.float32)

# Method 1: Generate atomic document IDs (simple incremental IDs)
def atomic_docid(nq_data_path, output_path, batch_size):
    print("Generating atomic document IDs...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size

    # Load the gzipped TSV file using pandas
    print(f"Loading data from {nq_data_path}...")
    with gzip.open(nq_data_path, 'rt') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'document_url', 'doc_tac', 'language'])

    print(f"Loaded {len(df)} documents!")

    # Open the output file
    with open(output_path, "w") as fw:
        for start_idx in tqdm(range(0, len(df), batch_size), desc="Generating atomic IDs in batches"):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            for doc_index, row in batch_df.iterrows():
                docid = f"[{row['id']}]".lower()  # Convert the ID to a formatted string
                encoded_docid = vocab_size + doc_index  # Generate atomic ID by adding doc_index to vocab_size
                fw.write(f"{docid}\t{encoded_docid}\n")

    print(f"Atomic document IDs have been successfully written to {output_path}.")

        
# Method 2: Generate Product Quantization (PQ) based document IDs with batching
def product_quantization_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path):
    print("Generating Product Quantization (PQ) document IDs...")

    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size

    # Initialize PQ model
    pq = nanopq.PQ(M=args.sub_space, Ks=args.cluster_num)

    # Train PQ on the entire dataset
    print("Training PQ on the entire dataset...")
    pq.fit(doc_embeddings)
    print(np.array(pq.codewords).shape)

    # Encode to PQ-codes
    print("Encoding document embeddings to PQ-codes...")
    X_code = pq.encode(doc_embeddings)  

    with open(output_path, "w") as fw:
        for idx, doc_code in tqdm(enumerate(X_code), desc="writing doc code into the file..."):
            docid = idx_2_docid[idx]
            new_doc_code = [int(x) for x in doc_code]
            for i, x in enumerate(new_doc_code):
                new_doc_code[i] = int(x) + i*256
            code = ','.join(str(x + vocab_size) for x in new_doc_code)
            fw.write(docid + "\t" + code + "\n")    

    print(f"Processed {len(doc_embeddings)} documents!")
    print(f"PQ-based document IDs successfully written to {output_path}")
    

# Method 3: Generate document IDs based on URL tokenization (leveraging GPU for T5)
def url_docid(nq_data_path, output_path, batch_size):
    print("Generating URL-based document IDs...")

    # Initialize T5 model and tokenizer, moving to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path).to(device)
    model.eval()

    max_docid_len = 128  # Maximum length of the document ID
    buffer = []  # Buffer to hold batched documents

    # Load the gzipped TSV file using pandas
    print(f"Loading data from {nq_data_path}...")
    with gzip.open(nq_data_path, 'rt') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'document_url', 'doc_tac', 'language'])

    print(f"Loaded {len(df)} documents!")

    # Open output file for writing
    with open(output_path, "w") as fw:
        for _, row in tqdm(df.iterrows(), desc='Processing documents', total=len(df)):
            docid = "[{}]".format(str(row['id']).lower())  # Cast the 'id' to string first

            url = row['document_url'].lower()
            # Handle missing titles by replacing them with an empty string
            title = str(row['title']) if not pd.isna(row['title']) else ''
            title = title.lower().strip()

            # Clean URL and title
            url = url.replace("http://", "").replace("https://", "").replace("-", " ").replace("_", " ")\
                     .replace("?", " ").replace("=", " ").replace("+", " ").replace(".html", "")\
                     .replace(".php", "").replace(".aspx", "").strip()
            reversed_url = url.split('/')[::-1]
            url_content = " ".join(reversed_url[:-1])
            domain = reversed_url[-1]

            url_content = ''.join([i for i in url_content if not i.isdigit()])
            url_content = re.sub(' +', ' ', url_content).strip()

            # Combine title and domain or use URL content if title is too short
            if len(title.split()) <= 2:
                url = url_content + " " + domain  # Use the URL content and domain if the title is short
            else:
                url = title + " " + domain  # Use the title and domain if the title is longer


            # Append the document to buffer
            buffer.append((docid, url))

            # Process buffer when full (batch size reached)
            if len(buffer) >= batch_size:
                process_buffer(buffer, tokenizer, model, fw, max_docid_len, device)
                buffer = []  # Clear the buffer after processing

        # Process remaining documents in the buffer
        if buffer:
            process_buffer(buffer, tokenizer, model, fw, max_docid_len, device)
    print(f"Processed {len(df)} documents!")
    print(f"URL-based document IDs successfully written to {output_path}")

def process_buffer(buffer, tokenizer, model, fw, max_docid_len, device):
    """
    Process a batch of documents, tokenize the URLs, and generate document IDs using T5 model.
    """
    docids, urls = zip(*buffer)  # Separate doc IDs and URLs
    inputs = tokenizer(list(urls), return_tensors="pt", padding=True, truncation=True, max_length=100).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs)

    # Write the results to file
    for docid, output_id in zip(docids, output_ids):
        doc_code = ','.join([str(x.item()) for x in output_id[:max_docid_len]])  # Convert to string and trim to max length
        fw.write(f"{docid}\t{doc_code}\n")
  
def encode_and_write(buffer, tokenizer, model, output_path, device):
    # Tokenize the URLs and titles
    docids, url_titles = zip(*buffer)
    inputs = tokenizer(list(url_titles), return_tensors="pt", padding=True, truncation=True, max_length=100).to(device)

    with torch.no_grad():
        # Token IDs are the semantic document IDs
        output_ids = model.generate(**inputs)

    # Write the generated document IDs
    with open(output_path, "a") as fw:
        for docid, output_id in zip(docids, output_ids):
            doc_code = ','.join(map(str, output_id.cpu().numpy()))
            fw.write(f"{docid}\t{doc_code}\n")

# Method 4: Generate summary-based document IDs
def summary_based_docid(summary_path, output_path, max_docid_len=128):
    """
    Generate summary-based document IDs from a JSON file with summaries.

    :param summary_path: Path to the input JSON file containing summaries.
    :param output_path: Path to save the output file with encoded docids.
    :param max_docid_len: Maximum token length for encoding summaries.
    """
    print("Generating summary-based document IDs...")
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

    # Load summaries from the input file
    with open(summary_path, "r") as f:
        summaries = json.load(f)

    # Open the output file for writing
    with open(output_path, "w") as fw:
        for summary_item in tqdm(summaries, desc="Encoding summaries"):
            # Validate the structure of each summary item
            if 'id' not in summary_item or 'summary' not in summary_item:
                # print(f"Skipping invalid item: {summary_item}")
                continue

            # Extract document ID and summary
            docid = f"[{summary_item['id'].lower()}]"
            summary = summary_item['summary']

            # Tokenize the summary
            tokenized_summary = tokenizer(summary, truncation=True, max_length=max_docid_len).input_ids

            # Convert tokenized summary to comma-separated string
            doc_code = ','.join(map(str, tokenized_summary))

            # Write to the output file
            fw.write(f"{docid}\t{doc_code}\n")

    print(f"Summary-based document IDs written to {output_path}")


if __name__ == "__main__":
    # Load precomputed embeddings from the file
    docid_2_idx, idx_2_docid, doc_embeddings = load_doc_embeddings(args.input_embeddings)

    # Based on the encoding method, generate IDs
    if args.encoding == "atomic":
        atomic_docid(args.nq_data_path, args.output_path, args.batch_size)
    elif args.encoding == "pq":
        product_quantization_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, args.output_path)
    elif args.encoding == "url_title":
        url_docid(args.nq_data_path, args.output_path, args.batch_size)
    elif args.encoding == "summary":
        summary_based_docid(args.summary_path, args.output_path)

    else:
        raise ValueError("Invalid encoding method! Use one of 'atomic', 'pq', or 'url'.")
