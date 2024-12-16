import gzip
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm  # Progress bar

# Paths for input and logging
input_file = 'resources/datasets/processed/nq-data/nq-merged/nq_docs.tsv.gz'
log_dir = 'logs-slurm-sft-nq/other-logs'

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Generate a descriptive log filename based on date and dataset
log_filename = f"NQ_Document_Size_Analysis_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_path = os.path.join(log_dir, log_filename)

def analyze_and_log_document_sizes(file_path, log_path, chunk_size=1000):
    document_lengths = []
    total_rows = 0
    column_dtypes = {}
    missing_values = {}

    # Read the gzipped file in chunks and process with a progress bar
    with gzip.open(file_path, 'rt') as f:
        with tqdm(total=109_739, desc="Processing Documents") as pbar:
            for chunk in pd.read_csv(f, sep='\t', header=None, 
                                     names=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 
                                            'content', 'document_url', 'doc_tac', 'language'], 
                                     chunksize=chunk_size):
                # Track the number of rows
                total_rows += len(chunk)
                
                # Update missing values and data types for the first chunk
                if not column_dtypes:
                    column_dtypes = chunk.dtypes.to_dict()
                
                # Update cumulative missing values count
                for col in chunk.columns:
                    missing_values[col] = missing_values.get(col, 0) + chunk[col].isna().sum()
                
                # Drop rows with empty 'doc_tac'
                chunk.dropna(subset=['doc_tac'], inplace=True)
                
                # Calculate document lengths
                document_lengths.extend(chunk['doc_tac'].map(len))

                # Update progress bar
                pbar.update(len(chunk))

    # Compute statistics
    total_docs = len(document_lengths)
    avg_length = sum(document_lengths) / total_docs if total_docs else 0
    min_length = min(document_lengths) if total_docs else 0
    max_length = max(document_lengths) if total_docs else 0
    median_length = pd.Series(document_lengths).median() if total_docs else 0
    std_dev = pd.Series(document_lengths).std() if total_docs else 0

    # Prepare log content with basic info and missing values
    log_content = (
        f"Document Size Analysis for NQ Dataset\n"
        f"--------------------------------------\n"
        f"Total Rows Processed: {total_rows}\n"
        f"Total Documents Analyzed: {total_docs}\n\n"
        f"Data Types by Column:\n"
        f"{column_dtypes}\n\n"
        f"Missing Values by Column:\n"
        f"{missing_values}\n\n"
        f"Document Length Statistics:\n"
        f"---------------------------\n"
        f"Average Document Length: {avg_length:.2f} characters\n"
        f"Median Document Length: {median_length} characters\n"
        f"Minimum Document Length: {min_length} characters\n"
        f"Maximum Document Length: {max_length} characters\n"
        f"Standard Deviation of Document Lengths: {std_dev:.2f} characters\n\n"
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    # Write the log content to file
    with open(log_path, 'w') as log_file:
        log_file.write(log_content)

    # Display log path
    print(f"Analysis saved to {log_path}")

# Run the analysis and save the log
analyze_and_log_document_sizes(input_file, log_path)
