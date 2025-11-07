#!/bin/bash


# Input NQ train/dev TSV.gz files (created from previous preprocessing)
TRAIN_FILE="resources/datasets/processed/nq-data/nq_train.gz"
DEV_FILE="resources/datasets/processed/nq-data/nq_val.gz"

# Output directory
OUTPUT_DIR="resources/datasets/processed/nq-msmarco"
mkdir -p "$OUTPUT_DIR"

# Run conversion
python src/data/data_prep/nq/convert_nq_to_msmarco_format.py \
  --nq_train_file "$TRAIN_FILE" \
  --nq_dev_file "$DEV_FILE" \
  --output_dir "$OUTPUT_DIR"