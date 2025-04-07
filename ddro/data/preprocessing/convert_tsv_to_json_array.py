import csv
import json
import gzip
import os
import sys
import logging
from tqdm import tqdm

def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

def convert_tsv_to_json(tsv_path, output_json_path):
    if not os.path.exists(tsv_path):
        logging.error(f"TSV file not found: {tsv_path}")
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    # Count lines for progress
    with gzip.open(tsv_path, 'rt') as file:
        total_lines = sum(1 for _ in file)

    with gzip.open(tsv_path, 'rt') as file, open(output_json_path, 'w') as json_out:
        reader = csv.DictReader(file, delimiter='\t', fieldnames=['id', 'url', 'title', 'content'])
        json_out.write('[')
        first = True

        with tqdm(total=total_lines, desc="Converting", unit="rows") as pbar:
            for row in reader:
                try:
                    json_entry = {'id': row['id'], 'contents': row['content']}
                    if not first:
                        json_out.write(',\n')
                    json.dump(json_entry, json_out)
                    first = False
                    pbar.update(1)
                except KeyError as e:
                    logging.error(f"Missing expected field: {e}")
                except Exception as e:
                    logging.error(f"Failed to process row: {e}")

        json_out.write(']')

    logging.info(f"TSV to JSON conversion completed: {output_json_path}")
    print(f"Conversion completed: {output_json_path}")

if __name__ == "__main__":
    csv.field_size_limit(sys.maxsize)
    tsv_input = "resources/datasets/raw/msmarco-data/msmarco-docs.tsv.gz"
    json_output = "resources/datasets/processed/msmarco-data/msmarco-json-sents/msmarco-docs.json"
    log_file = "logs/tsv_to_json_conversion.log"

    setup_logging(log_file)
    convert_tsv_to_json(tsv_input, json_output)
