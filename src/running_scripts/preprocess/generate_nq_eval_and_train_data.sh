#!/bin/bash

# Set paths to input files and directories
PRETRAIN_MODEL_PATH="/path/to/transformer_models/t5-base"
DATA_PATH="/path/to/data.jsonl"
DOCID_PATH="/path/to/encoded_docid.txt"
QUERY_PATH="/path/to/queries.tsv.gz"
QRELS_PATH="/path/to/qrels.tsv.gz"
FAKE_QUERY_PATH="/path/to/fake_query.txt"
OUTPUT_DIR="/path/to/output_directory"

# Parameters
ENCODING_METHOD="url"  # Choose from atomic, pq, url, or summary
SCALE="top_300k"       # Choose from top_300k, rand_300k
CUR_DATA="general_pretrain"  # Choose from general_pretrain, search_pretrain, finetune
MAX_SEQ_LENGTH=128

# Activate the Python environment if needed
# source /path/to/venv/bin/activate

# Generate evaluation data
echo "Generating evaluation data..."
python ./src/data/generate_instances/generate_eval_data_wrapper.py \
  --encoding "$ENCODING_METHOD" \
  --scale "$SCALE" \
  --qrels_path "$QRELS_PATH" \
  --query_path "$QUERY_PATH" \
  --pretrain_model_path "$PRETRAIN_MODEL_PATH"

# Generate training data
echo "Generating training data..."
python ./src/data/generate_instances/generate_train_data_wrapper.py \
  --encoding "$ENCODING_METHOD" \
  --scale "$SCALE" \
  --cur_data "$CUR_DATA" \
  --query_path "$QUERY_PATH" \
  --qrels_path "$QRELS_PATH" \
  --pretrain_model_path "$PRETRAIN_MODEL_PATH" \
  --fake_query_path "$FAKE_QUERY_PATH"

# Deactivate the Python environment if activated earlier
# deactivate