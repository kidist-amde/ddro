import os
from openai import OpenAI
import argparse
import gzip
import pandas as pd
import json
import os
import csv
from tqdm import tqdm
import sys

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

# Initialize OpenAI client using API key from environment variable
API_KEY = "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"
client = OpenAI(api_key=API_KEY)

PROMPT = """
Generate a concise summary of the following document, capturing its main points, key arguments, and significant insights clearly and accurately.

# Instructions:
# - Include only content explicitly present in the document.
# - Do NOT add personal opinions, external information, or assumptions.
# - Use the format:
#   ## Summary:
#   {{summary}}
# - Maintain clear, concise, and neutral language.
"""

# Function to make an API call to OpenAI for summarization
def call_openai_api(doc_content, system_message, model="gpt-4o-mini"):
    """
    Call the OpenAI API for summarization using the specified model.

    Parameters:
    - doc_content: str - The content to summarize.
    - system_message: str - The system message or instructions.
    - model: str - The OpenAI model to use (default: "gpt-4o-mini").

    Returns:
    - str - The summary or an error message.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": doc_content}
            ],
            max_tokens=128  # Adjust as needed
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error processing summary."

# Batchify function to process in batches
def batchify(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# Summarization function to process documents in batches
def summarize_documents_from_batches(batch, system_message, model="gpt-4o-mini"):
    summaries = []
    for doc in batch:
        summary = call_openai_api(doc, system_message, model)
        summaries.append(summary.replace('## Summary:', '').strip())
    return summaries

def main():
    parser = argparse.ArgumentParser(description="Summarize documents from a gzipped TSV file.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the gzipped TSV input file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the JSONL output file.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing.')
    parser.add_argument('--max_docs', type=int, required=True, help='Maximum number of documents to process.')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"Loading data from {args.input_file}...")
    with gzip.open(args.input_file, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'document_url', 'doc_tac', 'language'])
    print(f"Loaded {len(df)} documents!")

    existing_docids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf8') as f:
            existing_docids = set([json.loads(line)['docid'] for line in f])

    processed_docs = 0

    with gzip.open(args.input_file, 'rt', encoding='utf8') as doc_file, \
         open(args.output_path, 'a', encoding='utf-8', buffering=1) as outfile:
        pbar = tqdm(desc="Processing Documents", unit="doc", total=args.max_docs)

        csvreader = csv.reader(doc_file, delimiter='\t')
        for rows in batchify(csvreader, args.batch_size):
            batch_documents = []
            batch_docids = []
            for row in rows:
                docid, document = row[1], row[8]
                if docid in existing_docids:
                    continue
                batch_documents.append(document)
                batch_docids.append(docid)
            if not batch_documents:
                continue
            summaries = summarize_documents_from_batches(batch_documents, PROMPT)
            for docid, summary in zip(batch_docids, summaries):
                jitem = json.dumps({'docid': docid, 'summary': summary})
                outfile.write(jitem + '\n')
                processed_docs += 1
                pbar.update(1)
                if processed_docs >= args.max_docs:
                    break
            if processed_docs >= args.max_docs:
                break

        pbar.close()
        print(f"Processed {processed_docs} documents.")

if __name__ == "__main__":
    main()
