import csv
import json
import gzip
import os
import sys
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    filename='logs-slurm-sft/other-logs/tsv_to_json.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logging.info("Starting TSV to JSON conversion for a large file.")

# Increase the field size limit
csv.field_size_limit(sys.maxsize)

# Input and output file paths
tsv_file = 'resources/datasets/raw/msmarco-data/msmarco-docs.tsv.gz'
json_file = 'resources/datasets/processed/msmarco-data/msmarco-json-sents/msmarco-docs.json'

# Check if the TSV file exists
if not os.path.exists(tsv_file):
    logging.error(f"TSV file not found: {tsv_file}")
    raise FileNotFoundError(f"TSV file not found: {tsv_file}")

# Count total lines for the progress bar
logging.info("Counting total rows in the TSV file.")
with gzip.open(tsv_file, mode='rt') as file:
    total_lines = sum(1 for _ in file)

# Process and write TSV rows incrementally to JSON
logging.info("Starting the conversion process.")
with gzip.open(tsv_file, mode='rt') as file, open(json_file, mode='w') as json_out:
    # Define the TSV field names
    tsv_reader = csv.DictReader(file, delimiter='\t', fieldnames=['id', 'url', 'title', 'content'])
    json_out.write('[')  # Start of JSON array
    first = True

    with tqdm(total=total_lines, desc="Converting", unit="rows") as pbar:
        for row in tsv_reader:
            try:
                # Extract only the required fields
                filtered_row = {'id': row['id'], 'contents': row['content']}
                
                if not first:
                    json_out.write(',\n')  # Add a comma and newline for separation
                json.dump(filtered_row, json_out)
                first = False
                pbar.update(1)
            except KeyError as e:
                logging.error(f"Missing expected field in row: {e}")
            except Exception as e:
                logging.error(f"Error processing row: {e}")
    
    json_out.write(']')  # End of JSON array

logging.info(f"Conversion completed successfully: {json_file}")
print(f"Converted '{tsv_file}' to '{json_file}' successfully!")
