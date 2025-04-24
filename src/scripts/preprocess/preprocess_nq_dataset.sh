#!/bin/bash

# Set paths to input files and output directories
DEV_FILE="/path/to/nq_dev_file.jsonl.gz"
TRAIN_FILE="/path/to/nq_train_file.jsonl.gz"
OUTPUT_MERGED_FILE="/path/to/output/merged_nq_dataset"
OUTPUT_TRAIN_FILE="/path/to/output/nq_train"
OUTPUT_VAL_FILE="/path/to/output/nq_dev"
OUTPUT_JSON_DIR="/path/to/output/json_dir"

# Optional: Set sample size (set to a number or leave empty for full dataset)
SAMPLE_SIZE=""

# Activate the Python environment if needed
# source /path/to/venv/bin/activate

# Run the Python script
python ./src/data/preprocessing/process_nq_dataset.py \
  --dev_file "$DEV_FILE" \
  --train_file "$TRAIN_FILE" \
  --output_merged_file "$OUTPUT_MERGED_FILE" \
  --output_train_file "$OUTPUT_TRAIN_FILE" \
  --output_val_file "$OUTPUT_VAL_FILE" \
  --output_json_dir "$OUTPUT_JSON_DIR" \
  ${SAMPLE_SIZE:+--sample_size "$SAMPLE_SIZE"}

# Deactivate the Python environment if activated earlier
# deactivate