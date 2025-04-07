import os
import re
import json
import argparse
import collections
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nanopq

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

# Argument parsing
parser = argparse.ArgumentParser(description="Generate document IDs using encoding methods")
parser.add_argument("--encoding", default="pq", type=str, help="docid method: atomic/pq/url")
parser.add_argument("--scale", default="top_300k", type=str, help="Scale: top/rand_100k/200k/300k.")
parser.add_argument("--top_or_rand", default="top", type=str, help="Top or random.")
parser.add_argument("--sub_space", default=24, type=int, help="The number of sub-spaces for 768-dim vector.")
parser.add_argument("--cluster_num", default=256, type=int, help="The number of clusters in each sub-space.")
parser.add_argument("--output_path", default="dataset/encoded_docid/t5_pq_top_300k.txt", type=str, help="Output path")
parser.add_argument("--pretrain_model_path", default="transformer_models/t5-base", type=str, help="Path to pre-trained T5 model")
parser.add_argument("--input_doc_path", type=str, required=True, help="Path to input document file")
parser.add_argument("--input_embed_path", type=str, required=True, help="Path to input embedding file for PQ")
parser.add_argument("--summary_path", default="path/to/msmarco-llama3_summaries.json", type=str, help="Path to pre-generated summaries JSON")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for encoding")
args = parser.parse_args()


def load_doc_vec(input_path):
    docid_2_idx, idx_2_docid = {}, {}
    doc_embeddings = []

    with open(input_path, "r") as fr:
        for line in tqdm(fr, desc="Loading document vectors..."):
            did, demb = line.strip().split('\t')
            d_embedding = [float(x) for x in demb.split(',')]

            docid_2_idx[did] = len(docid_2_idx)
            idx_2_docid[docid_2_idx[did]] = did
            doc_embeddings.append(d_embedding)

    print("Document embeddings loaded successfully.")
    return docid_2_idx, idx_2_docid, np.array(doc_embeddings, dtype=np.float32)


def atomic_docid(input_path, output_path):
    print("Generating atomic docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    encoded_docids = {}

    with open(input_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='Loading all docids'):
            doc_item = json.loads(line)
            docid = "[{}]".format(doc_item['docid'].lower())
            encoded_docids[docid] = vocab_size + doc_index

    with open(output_path, "w") as fw:
        for docid, code in encoded_docids.items():
            fw.write(f"{docid}\t{code}\n")


def product_quantization_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path):
    print("Generating product quantization docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size

    pq = nanopq.PQ(M=args.sub_space, Ks=args.cluster_num)

    print("Training codewords...")
    pq.fit(doc_embeddings)
    print(np.array(pq.codewords).shape)

    print("Encoding document embeddings...")
    with open(output_path, "w") as fw:
        # Encode in batches using args.batch_size
        for i in range(0, len(doc_embeddings), args.batch_size):
            batch_embeddings = doc_embeddings[i:i + args.batch_size]
            X_code = pq.encode(batch_embeddings)  # Encode batch to PQ-codes
            
            for idx, doc_code in enumerate(X_code, start=i):
                docid = idx_2_docid[idx]
                new_doc_code = [int(x) + i * 256 for i, x in enumerate(doc_code)]
                code = ','.join(str(x + vocab_size) for x in new_doc_code)
                fw.write(f"{docid}\t{code}\n")


def url_docid(input_path, output_path):
    print("Generating URL-based docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    max_docid_len = 99
    encoded_docids = {}

    with open(input_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='Loading all docids'):
            doc_item = json.loads(line)
            docid = "[{}]".format(doc_item['docid'].lower())
            url = doc_item['url'].lower().replace("http://", "").replace("https://", "").replace("-", " ")
            title = doc_item['title'].lower().strip()

            reversed_url = url.split('/')[::-1]
            url_content = " ".join(reversed_url[:-1])
            domain = reversed_url[-1]

            if len(title.split()) <= 2:
                url = url_content + " " + domain
            else:
                url = title + " " + domain
            
            encoded_docids[docid] = tokenizer(url).input_ids[:-1][:max_docid_len] + [1]

    with open(output_path, "w") as fw:
        for docid, code in encoded_docids.items():
            fw.write(f"{docid}\t{','.join(map(str, code))}\n")

# # Method 4: Generate summary-based document IDs

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
    if args.encoding == "atomic":
        atomic_docid(args.input_doc_path, args.output_path)
    elif args.encoding == "pq":
        docid_2_idx, idx_2_docid, doc_embeddings = load_doc_vec(args.input_embed_path)
        product_quantization_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, args.output_path)
    elif args.encoding == "url":
        url_docid(args.input_doc_path, args.output_path)
    elif args.encoding == "summary":
        summary_based_docid(args.summary_path, args.output_path)
    else:
        raise ValueError("Invalid encoding method. Choose from atomic/pq/url/summary.")
