import torch
import pandas as pd
import gzip
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer and the fine-tuned model
tokenizer = T5Tokenizer.from_pretrained('resources/transformer_models/finetuned_doc2query_t5_large_msmarco')
model = T5ForConditionalGeneration.from_pretrained('resources/transformer_models/finetuned_doc2query_t5_large_msmarco')
model.to(device)

# Path to your NQ dataset
dataset_path = 'ddro/resources/datasets/raw/nq-data/nq_merged.tsv.gz'

# Output file to save the generated pseudo-queries
output_file = 'pseudo_queries_output.txt'

# Batch size for processing multiple documents at once
batch_size = 8  # Adjust this based on your GPU/CPU capacity

# Function to generate pseudo-queries for a batch of documents
def generate_queries_batch(doc_texts, num_queries=10, max_length=256):
    input_ids = tokenizer.batch_encode_plus(doc_texts, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=10,
        num_return_sequences=num_queries
    )
    
    # Split the generated outputs back into batches
    num_docs = len(doc_texts)
    outputs = outputs.view(num_docs, num_queries, -1)
    
    queries_per_doc = []
    for i in range(num_docs):
        queries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs[i]]
        queries_per_doc.append(queries)
    
    return queries_per_doc

# Open the NQ dataset and process it
with gzip.open(dataset_path, 'rt') as f, open(output_file, 'w', encoding='utf-8') as out_f:
    df = pd.read_csv(f, sep='\t', header=None, 
                     names=['query', 'id', 'long_answer', 'short_answer', 
                            'title', 'abstract', 'content', 'document_url', 'doc_tac', 'language'])

    # Process the documents in batches
    docs = []
    doc_ids = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        doc_id = f'd{idx}'  # Document ID in the format 'd0', 'd1', etc.
        doc_content = row['doc_tac']  # Use 'doc_tac' as the document content
        
        # Skip empty content
        if pd.isna(doc_content) or not doc_content.strip():
            continue

        # Append document content and ID to lists for batching
        docs.append(doc_content)
        doc_ids.append(doc_id)

        # If batch is full, generate queries for the batch
        if len(docs) == batch_size:
            queries_batch = generate_queries_batch(docs, num_queries=10)

            # Write each query in the desired format
            for doc_id, queries in zip(doc_ids, queries_batch):
                for query in queries:
                    out_f.write(f'[{doc_id}]\t{query}\n')
            
            # Clear the batch
            docs = []
            doc_ids = []

    # Process any remaining documents in the last batch
    if docs:
        queries_batch = generate_queries_batch(docs, num_queries=10)
        for doc_id, queries in zip(doc_ids, queries_batch):
            for query in queries:
                out_f.write(f'[{doc_id}]\t{query}\n')

print(f'Pseudo-queries have been saved to {output_file}')
