import os
import re
import json
import argparse
import collections
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

parser = argparse.ArgumentParser(description="Generate document IDs using encoding methods")
parser.add_argument("--encoding", default="pq", type=str, help="docid method: atomic/pq/url/summary")
parser.add_argument("--scale", default="top_300k", type=str, help="Scale of the dataset.")
parser.add_argument("--top_or_rand", default="top", type=str, help="Top or random selection.")
parser.add_argument("--sub_space", default=24, type=int, help="Sub-spaces for 768-dim vector.")
parser.add_argument("--cluster_num", default=256, type=int, help="Clusters per sub-space.")
parser.add_argument("--output_path", default="output/encoded_docid.txt", type=str, help="Output path")
parser.add_argument("--pretrain_model_path", default="transformer_models/t5-base", type=str, help="Path to pre-trained model")
parser.add_argument("--input_doc_path", type=str, required=True, help="Path to input document file")
parser.add_argument("--input_embed_path", type=str, required=True, help="Path to input embedding file")
parser.add_argument("--summary_path", default="data/summaries.json", type=str, help="Path to summaries JSON")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for encoding")
args = parser.parse_args()

def load_doc_vec(input_path):
    docid_2_idx, idx_2_docid = {}, {}
    doc_embeddings = []
    with open(input_path, "r") as fr:
        for line in tqdm(fr, desc="Loading document vectors"):
            did, demb = line.strip().split('\t')
            d_embedding = [float(x) for x in demb.split(',')]
            docid_2_idx[did] = len(docid_2_idx)
            idx_2_docid[docid_2_idx[did]] = did
            doc_embeddings.append(d_embedding)
    return docid_2_idx, idx_2_docid, np.array(doc_embeddings, dtype=np.float32)

def atomic_docid(input_path, output_path):
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    encoded_docids = {}
    with open(input_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='Processing atomic docids'):
            doc_item = json.loads(line)
            docid = f"[{doc_item['docid'].lower()}]"
            encoded_docids[docid] = vocab_size + doc_index
    with open(output_path, "w") as fw:
        for docid, code in encoded_docids.items():
            fw.write(f"{docid}\t{code}\n")

def product_quantization_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path):
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    pq = nanopq.PQ(M=args.sub_space, Ks=args.cluster_num)
    pq.fit(doc_embeddings)
    with open(output_path, "w") as fw:
        for i in range(0, len(doc_embeddings), args.batch_size):
            batch_embeddings = doc_embeddings[i:i + args.batch_size]
            X_code = pq.encode(batch_embeddings)
            for idx, doc_code in enumerate(X_code, start=i):
                docid = idx_2_docid[idx]
                new_doc_code = [int(x) + i * 256 for i, x in enumerate(doc_code)]
                code = ','.join(str(x + vocab_size) for x in new_doc_code)
                fw.write(f"{docid}\t{code}\n")

def is_url_semantically_rich(url_segments):
    """
    Determines if a URL is semantically rich based on its segments.
    A URL is considered semantically rich if the majority of its segments
    are descriptive and not generic or numeric.
    """
    generic_terms = {"index", "page", "item", "view", "default", "home"}
    descriptive_count = 0
    total_segments = len(url_segments)

    for segment in url_segments:
        if not segment.isnumeric() and segment not in generic_terms and len(segment) > 2:
            descriptive_count += 1

    # Consider the URL semantically rich if more than half of the segments are descriptive
    return descriptive_count > total_segments / 2

def url_docid(input_data, max_docid_len=99, pretrain_model_path="transformer_models/t5-base"):
    """
    Generates tokenized document IDs based on reversed URL segments or title + domain.
    
    Args:
        input_data (list): List of JSON objects, each containing `docid`, `url`, and `title`.
        max_docid_len (int): Maximum length of tokenized IDs.
        pretrain_model_path (str): Path to the pre-trained T5 model.
    
    Returns:
        dict: Mapping of each `docid` to its final string, source, and token IDs.
    """
    tokenizer = T5Tokenizer.from_pretrained(pretrain_model_path)
    results = {}
    skipped_docs = []

    for doc_item in tqdm(input_data, desc="Processing URL docids"):
        try:
            # Validate and extract fields
            docid = doc_item.get('docid', '').strip().lower()
            url = doc_item.get('url', '')
            title = doc_item.get('title', '')

            # Ensure docid is present
            if not docid:
                skipped_docs.append({"reason": "Missing or empty docid", "doc_item": doc_item})
                continue

            # Ensure url and title are strings
            url = url.strip().lower() if isinstance(url, str) else ""
            title = title.strip().lower() if isinstance(title, str) else ""

            # Skip if both url and title are empty
            if not url and not title:
                skipped_docs.append({"reason": "Missing both URL and title", "docid": docid})
                continue

            # Process URL
            url = url.replace("http://", "").replace("https://", "").replace("-", " ")
            url_segments = [segment for segment in url.split('/') if segment]  # Remove empty segments

            # Extract domain and reverse URL content
            domain = url_segments[0] if url_segments else ""
            reversed_url = " ".join(reversed(url_segments[1:])) if len(url_segments) > 1 else ""

            # Determine final string and source
            if url_segments and is_url_semantically_rich(url_segments):
                final_string = f"{reversed_url} {domain}".strip()
                source = "URL"
            elif title and domain:
                final_string = f"{title} {domain}".strip()
                source = "TITLE"
            elif title:
                final_string = title
                source = "TITLE"
            else:
                skipped_docs.append({"reason": "Unable to determine final string", "docid": docid})
                continue

            # Tokenize the final string
            tokenized_ids = tokenizer(final_string, truncation=True, max_length=max_docid_len).input_ids
            tokenized_ids = tokenized_ids[:-1][:max_docid_len] + [1]  # Ensure final token is [1]

            # Store results
            results[docid] = {
                "final_string": final_string,
                "source": source,
                "token_ids": tokenized_ids
            }

        except Exception as e:
            skipped_docs.append({"reason": f"Error: {str(e)}", "docid": doc_item.get('docid', 'unknown')})

    # Log skipped documents
    if skipped_docs:
        print("Skipped documents:")
        for skipped in skipped_docs:
            print(skipped)

    return results

def summary_based_docid(summary_path, output_path, max_docid_len=128):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    with open(summary_path, "r") as f:
        summaries = json.load(f)
    with open(output_path, "w") as fw:
        for summary_item in tqdm(summaries, desc="Encoding summaries"):
            if 'id' not in summary_item or 'summary' not in summary_item:
                continue
            docid = f"[{summary_item['id'].lower()}]"
            summary = summary_item['summary']
            tokenized_summary = tokenizer(summary, truncation=True, max_length=max_docid_len).input_ids
            doc_code = ','.join(map(str, tokenized_summary))
            fw.write(f"{docid}\t{doc_code}\n")

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
