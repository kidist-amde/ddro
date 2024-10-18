# This script processes the Google Natural Questions (NQ) dataset.
# run_scripts/extract_nq_data.py

import argparse
import pandas as pd
import jsonlines
import re
from tqdm.auto import tqdm
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import sys

DEV_SET_SIZE = 7830
TRAIN_SET_SIZE = 307373

# Precompile regex patterns for efficiency
HTML_TAG_RE = re.compile(r'<[^<]+?>')

def process_chunk(input_chunk, dataset_type, total_entries):
    data = []
    for idx, item in enumerate(input_chunk):
        arr = []
        # Process question_text and example_id
        question_text = item['question_text']
        arr.append(question_text)

        example_id = str(item['example_id'])
        arr.append(example_id)

        if dataset_type == 'dev':
            tokens = [t['token'] for t in item['document_tokens']]
            document_text = ' '.join(tokens)
        else:
            document_text = item['document_text']

        # Process long_answer
        annotation = item['annotations'][0]
        has_long_answer = annotation['long_answer']['start_token'] >= 0
        long_answer = ''

        if has_long_answer:
            long_answers = [
                a['long_answer']
                for a in item['annotations']
                if a['long_answer']['start_token'] >= 0
            ]
            start_token = long_answers[0]['start_token']
            end_token = long_answers[0]['end_token']
            words = document_text.split(' ')
            long_answer = ' '.join(words[start_token:end_token])
            long_answer = HTML_TAG_RE.sub('', long_answer).replace('\n', '').strip()
        arr.append(long_answer)

        # Process short_answer
        has_short_answer = annotation['short_answers'] or annotation['yes_no_answer'] != 'NONE'
        short_answer = ''
        if has_short_answer and annotation['short_answers']:
            sa = []
            for i in annotation['short_answers']:
                start_token_s = i['start_token']
                end_token_s = i['end_token']
                shorta = ' '.join(words[start_token_s:end_token_s])
                sa.append(shorta)
            short_answer = '|'.join(sa)
            short_answer = HTML_TAG_RE.sub('', short_answer).replace('\n', '').strip()
        arr.append(short_answer)

        # Process title and abstract differently for dev and train
        title, abs, content = process_document_text(document_text, dataset_type)
        arr.extend([title, abs, content])

        # Add final doc tac (title + abs + content)
        doc_tac = title + abs + content
        arr.append(doc_tac)

        # Add language field (hardcoded 'en')
        arr.append('en')

        # Append processed data to the list
        data.append(arr)
    
    return data

def process_document_text(document_text, dataset_type):
    title = ''
    if dataset_type == 'dev':
        title = document_text.split('<H1>')[0] if '<H1>' in document_text else ''
    else:
        if '<H1>' in document_text:
            title_start = document_text.index('<H1>')
            title_end = document_text.index('</H1>')
            title = document_text[title_start+4:title_end]

    abs = ''
    if '<P>' in document_text:
        abs_start = document_text.index('<P>')
        abs_end = document_text.index('</P>')
        abs = document_text[abs_start+3:abs_end]

    content = ''
    if '</Ul>' in document_text:
        final = document_text.rindex('</Ul>')
        content = document_text[abs_end+4:final]
        content = HTML_TAG_RE.sub('', content).replace('\n', '').strip()

    return title, abs, content

def process_data_parallel(input_file_path, output_file_path, sample_size, total_entries, dataset_type, num_workers=4):
    with gzip.open(input_file_path, "rt") as f:
        reader = jsonlines.Reader(f)
        entries = list(reader)
    
    # If sample size is specified, limit the entries
    if sample_size:
        entries = entries[:sample_size]

    chunk_size = len(entries) // num_workers
    chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]

    # Open the output file for writing
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        # Detect if we're running in an interactive environment
        interactive = sys.stdout.isatty()

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_chunk, chunk, dataset_type, total_entries) for chunk in chunks]
            with tqdm(as_completed(futures), total=len(futures), desc="Processing Chunks", dynamic_ncols=True, 
                      disable=not interactive, file=sys.stdout, leave=True, mininterval=1, flush=True) as pbar:
                for future in pbar:
                    result = future.result()
                    writer.writerows(result)

    print(f"Processed data saved as TSV to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract and process Google NQ data")
    parser.add_argument("--input_file_path", type=str, required=True, help="Path to the NQ dataset .jsonl.gz file")
    parser.add_argument("--output_file_path", type=str, required=True, help="Base path to save the processed data (TSV format)")
    parser.add_argument("--dataset_type", type=str, choices=['dev', 'train'], required=True, help="Specify 'dev' or 'train' dataset")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to process (optional)")
    args = parser.parse_args()

    total_entries = DEV_SET_SIZE if args.dataset_type == 'dev' else TRAIN_SET_SIZE
    process_data_parallel(args.input_file_path, args.output_file_path, args.sample_size, total_entries, args.dataset_type)

if __name__ == "__main__":
    main()
