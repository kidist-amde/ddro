#!/bin/bash

# Paths to input files and output locations
DOC_FILE="resources/datasets/raw/msmarco-data/msmarco-docs.tsv.gz"
QRELS_FILE="resources/datasets/raw/msmarco-data/msmarco-doctrain-qrels.tsv.gz"
TOP_OUTPUT_FILE="resources/datasets/processed/msmarco-docs-sents.top.300k.json.gz"
LOG_FILE="logs/msmarco_preprocessing.log"

# Ensure output dirs exist
mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$(dirname "$TOP_OUTPUT_FILE")"
mkdir -p "$(dirname "$LOG_FILE")"

# Run Python preprocessing

echo "Starting preprocessing of MS MARCO documents..."
echo "Input document file: $DOC_FILE"
echo "Input QRELs file: $QRELS_FILE"
echo "Output file for top documents: $TOP_OUTPUT_FILE"

python src/data/data_prep/sample_top300k_msmarco_documents.py \
  --doc_file "$DOC_FILE" \
  --qrels_file "$QRELS_FILE" \
  --top_output_file "$TOP_OUTPUT_FILE" \
  --log_file "$LOG_FILE"

# Only show this if script succeeded
if [ $? -eq 0 ]; then
  echo "Preprocessing completed. Log saved to $LOG_FILE."
else
  echo "Preprocessing failed. Check SLURM log for errors."Add commentMore actions
fi