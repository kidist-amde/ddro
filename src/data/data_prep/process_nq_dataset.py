import argparse
import gzip
import os
import re
import json
import jsonlines
import pandas as pd
from tqdm.auto import tqdm
from transformers import BertTokenizer

# Constants
DEV_SET_SIZE = 7830
TRAIN_SET_SIZE = 307373

def process_nq_dev_dataset(input_file_path, sample_size=None):
    """Process the NQ development dataset."""
    nq_dev = []
    with gzip.open(input_file_path, 'rt') as f:
        for idx, item in enumerate(tqdm(jsonlines.Reader(f), total=sample_size or DEV_SET_SIZE, desc="Processing Dev Dataset")):
            if sample_size and idx >= sample_size:
                break
            nq_dev.append(extract_fields_from_nq_item(item, is_dev=True))

    return pd.DataFrame(nq_dev, columns=DATA_COLUMNS)

def process_nq_train_dataset(input_file_path, sample_size=None):
    """Process the NQ training dataset."""
    nq_train = []
    with gzip.open(input_file_path, 'rt') as f:
        for idx, item in enumerate(tqdm(jsonlines.Reader(f), total=sample_size or TRAIN_SET_SIZE, desc="Processing Train Dataset")):
            if sample_size and idx >= sample_size:
                break
            nq_train.append(extract_fields_from_nq_item(item, is_dev=False))

    return pd.DataFrame(nq_train, columns=DATA_COLUMNS)

def extract_fields_from_nq_item(item, is_dev=True):
    question = item['question_text']
    example_id = str(item['example_id'])
    tokens = [t['token'] for t in item.get('document_tokens', [])] if is_dev else item['document_text'].split()
    document_text = ' '.join(tokens)
    annotation = item['annotations'][0]

    # Long answer
    long_answer = ''
    if annotation['long_answer']['start_token'] >= 0:
        start = annotation['long_answer']['start_token']
        end = annotation['long_answer']['end_token']
        long_answer = clean_text(' '.join(tokens[start:end]))

    # Short answer
    short_answer = ''
    if annotation['short_answers'] or annotation['yes_no_answer'] != 'NONE':
        short_answers = annotation['short_answers']
        short_answer = '|'.join(clean_text(' '.join(tokens[sa['start_token']:sa['end_token']])) for sa in short_answers)

    # Title
    title = item.get('document_title', '') if is_dev else extract_tag_content(document_text, 'H1')

    # Abstract and content
    abstract = extract_tag_content(document_text, 'P')
    content = extract_content(document_text)

    document_url = item.get('document_url', '')
    doc_tac = title + abstract + content

    return [question, example_id, long_answer, short_answer, title, abstract, content, document_url, doc_tac, 'en']

def clean_text(text):
    return re.sub('<[^<]+?>', '', text).replace('\n', '').strip()

def extract_tag_content(text, tag):
    open_tag, close_tag = f'<{tag}>', f'</{tag}>'
    if open_tag in text:
        start = text.index(open_tag) + len(open_tag)
        end = text.index(close_tag)
        return clean_text(text[start:end])
    return ''

def extract_content(text):
    if text.rfind('</Ul>') != -1:
        final = text.rindex('</Ul>')
        return clean_text(text[final+5:])
    return clean_text(text)

def lower_text(x):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))

DATA_COLUMNS = [
    'query', 'id', 'long_answer', 'short_answer', 'title',
    'abstract', 'content', 'document_url', 'doc_tac', 'language'
]

def main():
    parser = argparse.ArgumentParser(description="Process and merge Google NQ dataset")
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_merged_file", type=str, required=True)
    parser.add_argument("--output_train_file", type=str, required=True)
    parser.add_argument("--output_val_file", type=str, required=True)
    parser.add_argument("--output_json_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()

    nq_dev = process_nq_dev_dataset(args.dev_file, args.sample_size)
    nq_train = process_nq_train_dataset(args.train_file, args.sample_size)

    nq_dev['title'] = nq_dev['title'].map(lower_text)
    nq_train['title'] = nq_train['title'].map(lower_text)

    nq_all = pd.concat([nq_train, nq_dev], ignore_index=True).drop_duplicates('title').reset_index(drop=True)

    # Filter training/dev to remove duplicates
    valid_ids = set(nq_all['id'])
    nq_train = nq_train[nq_train['id'].isin(valid_ids)]
    nq_dev = nq_dev[nq_dev['id'].isin(valid_ids)]

    # Save outputs
    os.makedirs(args.output_json_dir, exist_ok=True)

    nq_train.to_csv(args.output_train_file + '.gz', sep='\t', index=False, header=False, compression='gzip')
    nq_dev.to_csv(args.output_val_file + '.gz', sep='\t', index=False, header=False, compression='gzip')
    nq_all.to_csv(args.output_merged_file + '.gz', sep='\t', index=False, header=False, compression='gzip')
    nq_all.to_json(args.output_merged_file + '.json', orient='records', lines=True)

    nq_all[['id', 'doc_tac']].rename(columns={'id': 'id', 'doc_tac': 'contents'})\
        .to_json(os.path.join(args.output_json_dir, 'nq_pyserini_format.json'), orient='records', lines=True)

    nq_all.to_json(os.path.join(args.output_json_dir, 'nq_all_columns.json'), orient='records', lines=True)

if __name__ == "__main__":
    main()
