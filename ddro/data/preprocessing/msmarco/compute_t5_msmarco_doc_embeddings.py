import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# Set up argument parsing
parser = argparse.ArgumentParser(description='Generate embeddings for documents.')
parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSON file containing documents.')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output file for storing embeddings.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing documents')

args = parser.parse_args()

# Initialize the SentenceTransformer model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('sentence-transformers/gtr-t5-base').to(device)

# If multiple GPUs are available, use DataParallel for the model
multi_gpu = torch.cuda.device_count() > 1
if multi_gpu:
    model = torch.nn.DataParallel(model)

# Function to compute embeddings for a batch of documents
def process_batch(doc_texts, doc_ids):
    if multi_gpu:
        embeddings = model.module.encode(doc_texts, show_progress_bar=False, batch_size=args.batch_size)
    else:
        embeddings = model.encode(doc_texts, show_progress_bar=False, batch_size=args.batch_size)

    # Format embeddings with corresponding docid in the required format
    return [(f"[d{docid[1:]}]", ','.join(map(str, embedding))) for docid, embedding in zip(doc_ids, embeddings)]

# Read input JSON file and write embeddings to output file
with open(args.input_path, 'r') as input_file, open(args.output_path, 'w') as output_file:
    doc_texts, doc_ids = [], []
    
    for line in tqdm(input_file, desc="Processing documents", unit="docs"):
        doc = json.loads(line.strip())
        docid = doc["docid"].lower()
        doc_body = doc["body"]

        doc_texts.append(doc_body)
        doc_ids.append(docid)

        # Process and save embeddings when batch size is reached
        if len(doc_texts) >= args.batch_size:
            results = process_batch(doc_texts, doc_ids)
            for docid, embedding_str in results:
                output_file.write(f'{docid}\t{embedding_str}\n')
            output_file.flush()  # Ensure data is written immediately
            doc_texts, doc_ids = [], []  # Reset batch lists

    # Process any remaining documents in the final batch
    if doc_texts:
        results = process_batch(doc_texts, doc_ids)
        for docid, embedding_str in results:
            output_file.write(f'{docid}\t{embedding_str}\n')
        output_file.flush()

    print(f'Embeddings successfully saved to {args.output_path}')
