import os
import json
import gzip
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def smart_open(path, mode='rt', encoding='utf-8'):
    return gzip.open(path, mode, encoding=encoding) if path.endswith('.gz') else open(path, mode, encoding=encoding)

def load_documents(input_path: str, is_nq: bool = False):
    if is_nq:
        with smart_open(input_path) as f:
            df = pd.read_csv(f, sep='\t', header=None, names=[
                'query', 'id', 'long_answer', 'short_answer', 'title',
                'abstract', 'content', 'document_url', 'doc_tac', 'language'])
        texts = df['doc_tac'].fillna('').astype(str).tolist()
        ids = df['id'].astype(str).tolist()
    else:
        texts, ids = [], []
        with smart_open(input_path) as f:
            for line in tqdm(f, desc="Loading documents"):
                item = json.loads(line.strip())
                ids.append(item['docid'].lower())
                texts.append(item['body'])
    return ids, texts

def save_embeddings(output_path: str, doc_ids, embeddings):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with smart_open(output_path, 'w') as f:
        for docid, emb in zip(doc_ids, embeddings):
            emb_str = ','.join(map(str, emb))
            f.write(f"[{docid}]\t{emb_str}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate T5 document embeddings.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to input data.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to output embeddings.")
    parser.add_argument('--model_name', type=str, default='sentence-transformers/gtr-t5-base')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', choices=['msmarco', 'nq'], required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(args.model_name).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    is_nq = args.dataset == 'nq'
    doc_ids, texts = load_documents(args.input_path, is_nq=is_nq)

    print(f"Encoding {len(texts)} documents using {args.model_name} ...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        device=device
    )
    save_embeddings(args.output_path, doc_ids, embeddings)
    print(f"Embeddings saved to {args.output_path}")

if __name__ == "__main__":
    main()
