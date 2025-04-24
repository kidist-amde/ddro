#!/bin/bash

# Set paths to input files and output directory
NQ_TRAIN_FILE="/path/to/nq_train_file.gz"
NQ_DEV_FILE="/path/to/nq_dev_file.gz"
OUTPUT_DIR="/path/to/output_directory"

# Activate the Python environment if needed
# source /path/to/venv/bin/activate

# Run the Python script
python ./src/data/preprocessing/convert_nq_to_msmarco_format.py \
  --nq_train_file "$NQ_TRAIN_FILE" \
  --nq_dev_file "$NQ_DEV_FILE" \
  --output_dir "$OUTPUT_DIR"

# Deactivate the Python environment if activated earlier
# deactivate