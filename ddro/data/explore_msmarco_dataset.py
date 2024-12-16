# import gzip
# import pandas as pd

# # File path
# file_path = "resources/datasets/raw/msmarco-data/msmarco-docs.tsv.gz"

# # Expected columns
# expected_columns = ["doc_id", "url", "title", "body"]

# def check_missing_columns(file_path, chunk_size=100000):
#     missing_columns = set(expected_columns)  # Track missing columns
#     total_records = 0
#     missing_data_count = {col: 0 for col in expected_columns}  # Initialize missing data count

#     # Read the file in chunks
#     with gzip.open(file_path, 'rt', encoding='utf-8') as f:
#         for chunk in pd.read_csv(f, sep="\t", names=expected_columns, chunksize=chunk_size):
#             total_records += len(chunk)

#             # Check for columns missing in the chunk
#             present_columns = set(chunk.columns)
#             missing_columns -= present_columns

#             # Count missing data per column
#             missing_data_chunk = chunk.isnull().sum()
#             for col in expected_columns:
#                 if col in present_columns:
#                     missing_data_count[col] += missing_data_chunk.get(col, 0)

#     # Print results
#     print(f"Total Records Processed: {total_records}")
#     if missing_columns:
#         print(f"Missing Columns: {missing_columns}")
#     else:
#         print("All expected columns are present.")

#     print("Missing Data Counts per Column:")
#     for col, count in missing_data_count.items():
#         print(f"{col}: {count} missing values")

# # Run the check
# check_missing_columns(file_path)


import gzip
import logging
import argparse

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def validate_docs_file(docs_file, offsets_file, output_file):
    """
    Validates the docs file by:
    1. Checking for malformed lines.
    2. Checking for missing doc_ids referenced in the offsets file.
    3. Logging the results.
    """
    malformed_lines = 0
    total_lines = 0
    missing_doc_ids = []

    # Extract doc_ids from offsets file
    with gzip.open(offsets_file, 'rt', encoding='utf8') as f:
        doc_offsets = {line.split("\t")[0]: int(line.split("\t")[2]) for line in f}

    logging.info(f"Loaded {len(doc_offsets)} doc_ids from offsets file.")

    # Validate docs file
    valid_doc_ids = set()
    with gzip.open(docs_file, 'rt', encoding='utf8', errors='replace') as f:
        for line in f:
            total_lines += 1
            parts = line.strip().split("\t")
            if len(parts) < 4:  # Expecting at least 4 fields
                malformed_lines += 1
                logging.error(f"Malformed line at {total_lines}: {line.strip()}")
            else:
                valid_doc_ids.add(parts[0])  # Add valid doc_id to the set

            if total_lines % 1_000_000 == 0:
                logging.info(f"Processed {total_lines} lines...")

    logging.info(f"Completed validation. Total lines processed: {total_lines}")
    logging.info(f"Malformed lines: {malformed_lines}")

    # Check for missing doc_ids
    for doc_id in doc_offsets:
        if doc_id not in valid_doc_ids:
            missing_doc_ids.append(doc_id)

    logging.info(f"Missing doc_ids: {len(missing_doc_ids)}")
    
    # Save missing doc_ids to a file
    with open(output_file, 'w', encoding='utf8') as f:
        for doc_id in missing_doc_ids:
            f.write(f"{doc_id}\n")

    logging.info(f"Saved missing doc_ids to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate msmarco-docs.tsv.gz file.")
    parser.add_argument("--docs_file", required=True, help="Path to msmarco-docs.tsv.gz")
    parser.add_argument("--offsets_file", required=True, help="Path to msmarco-docs-lookup.tsv.gz")
    parser.add_argument("--output_file", required=True, help="File to save missing doc_ids")
    args = parser.parse_args()

    validate_docs_file(args.docs_file, args.offsets_file, args.output_file)
