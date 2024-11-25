import json
import os
import gzip
import argparse
import re
from tqdm import tqdm
from transformers import set_seed, pipeline
import pandas as pd
import torch

set_seed(313)  # Set random seed for reproducibility

PROMPT = """
Generate a concise summary of the following document, capturing its main points, key arguments, and significant insights clearly and accurately.

# Instructions:
# - Include only content explicitly present in the document.
# - Do NOT add personal opinions, external information, or assumptions.
# - Use the format:
#   ## Summary:
#   {{summary}}
# - If the document is too short or empty, output:
#   ## Summary:
#   (No content available)
# - Maintain clear, concise, and neutral language.
"""

# Preprocessing function to clean each document before summarization
def clean_document(doc):
    # Remove metadata, headers, and any irrelevant sections
    doc = re.sub(r"(References|External links|See also|Further reading).*", '', doc, flags=re.IGNORECASE)
    # Remove bullets, special symbols, and clean up whitespace
    doc = re.sub(r"(- |\* )", '', doc)
    doc = re.sub(r'\s+', ' ', doc).strip()
    return doc

# Post-processing to clean the generated summaries
def clean_summary(summary):
    summary = re.sub(r"(## Step \d+:.*|Key Points|Main Points|References|See also|External links).*", '', summary, flags=re.IGNORECASE)
    summary = re.sub(r"(- |\* )", '', summary)  # Remove bullets
    summary = re.sub(r'\s+', ' ', summary).strip()  # Clean up extra whitespace
    return summary.strip()

def summarize_documents_from_batches(batch, generator, max_new_tokens, system_message):
    # Clean documents before summarization
    cleaned_batch = [clean_document(doc) for doc in batch]

    # Prepare the batch messages for the input
    batch_messages = [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": doc}
        ]
        for doc in cleaned_batch
    ]

    # Generate summaries using the text-generation pipeline
    summaries = generator(batch_messages, max_new_tokens=max_new_tokens)
    # Extract the generated summaries
    extracted_summaries = [summary[0]['generated_text'][-1]['content'].replace('## Summary:', '').strip() for summary in summaries]
    return extracted_summaries

def main():
    parser = argparse.ArgumentParser(description="Summarize documents from a gzipped TSV file.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the gzipped TSV input file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the JSONL output file.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens to generate.')
    parser.add_argument('--max_docs', type=int, required=True, help='Maximum number of documents to process.')

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Model configuration
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    generator = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # Load the gzipped TSV file using pandas
    print(f"Loading data from {args.input_file}...")
    with gzip.open(args.input_file, 'rt') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'document_url', 'doc_tac', 'language'])
    print(f"Loaded {len(df)} documents!")

    processed_docs = 0

    with open(args.output_path, 'a', encoding='utf-8') as outfile:
        # Process documents in batches
        pbar = tqdm(total=args.max_docs, desc="Processing Documents", unit="doc")

        for start in range(0, len(df), args.batch_size):
            end = min(start + args.batch_size, len(df))
            batch_df = df.iloc[start:end]

            batch_documents = batch_df['doc_tac'].fillna('').tolist()
            batch_docids = batch_df['id'].tolist()

            # Generate summaries for the batch
            try:
                summaries_batch = summarize_documents_from_batches(batch_documents, generator, args.max_new_tokens, PROMPT)

            except RuntimeError as e:
                print(f"Error processing documents: {e}")
                continue

            # Write summaries to the output JSONL file
            for docid, summary in zip(batch_docids, summaries_batch):
                jitem = json.dumps({'docid': docid, 'summary': summary})
                outfile.write(jitem + '\n')
                processed_docs += 1
                pbar.update(1)

                if processed_docs >= args.max_docs:
                    break
            if processed_docs >= args.max_docs:
                break

    pbar.close()
    print(f"Processed {processed_docs} documents and saved summaries to {args.output_path}")

if __name__ == "__main__":
    main()
