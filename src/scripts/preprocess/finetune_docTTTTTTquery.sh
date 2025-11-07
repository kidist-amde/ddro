#!/bin/sh

# Args
DATASET_PATH="your/path/to/nq_dataset.jsonl"  
DATASET_NAME="nq"
OUTPUT_FILE="resources/datasets/processed//${DATASET_NAME}-data.tsv.gz"

python src/pretrain/finetune_docTTTTTquery.py \
  --dataset_path "$DATASET_PATH" \
  --dataset_name "$DATASET_NAME" \
  --output_file "$OUTPUT_FILE"