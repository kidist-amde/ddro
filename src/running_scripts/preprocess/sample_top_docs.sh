#!/bin/bash

# Set paths to input files and output directories
DOC_FILE="resources/datasets/raw/msmarco-docs.tsv.gz"
QRELS_FILE="resources/datasets/raw/msmarco-doctrain-qrels.tsv.gz"
OUTPUT_FILE="resources/datasets/processed/msmarco-docs-sents-all.json.gz"
LOG_FILE="logs/msmarco_preprocessing.log"

# Ensure the logs directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Activate the Python environment if needed
# source /path/to/venv/bin/activate

# Run the Python script
python ./src/data/preprocessing/preprocess_msmarco_documents.py \
  --doc_file "$DOC_FILE" \
  --qrels_file "$QRELS_FILE" \
  --output_file "$OUTPUT_FILE"

# Deactivate the Python environment if activated earlier
# deactivate

echo "Preprocessing completed. Logs saved to $LOG_FILE."