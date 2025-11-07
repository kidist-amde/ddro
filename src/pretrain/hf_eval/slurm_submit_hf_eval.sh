#!/bin/sh

# ----------------------------
# Configuration
# ----------------------------
ENCODING="url"        # encoding: 'pq', 'url'
DATASET="nq"    # dataset: 'msmarco' or 'nq'
SCALE="top_300k"     # scale: 'top_300k'

echo "=== Evaluation Configuration ==="
echo "Dataset: $DATASET"

echo "Encoding: $ENCODING"
echo "Scale: $SCALE"
echo "================================"

# ----------------------------
# Run Evaluation
# ----------------------------
python src/pretrain/hf_eval/launch_hf_eval_from_config.py \
  --dataset "$DATASET" \
  --encoding "$ENCODING" \
  --scale "$SCALE"

echo "=== Evaluation Completed ==="