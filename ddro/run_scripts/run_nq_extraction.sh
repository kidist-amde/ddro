#!/bin/sh

# Initialize Conda
eval "$(conda shell.bash hook)"  # Ensure Conda is initialized for the shell

# Activate the conda environment
conda activate ddro

# Create output directory if it doesn't exist
mkdir -p resources/datasets/processed/nq-data

# Set input and output file paths
INPUT_FILE="resources/datasets/raw/nq-data/nq-dev-all.jsonl.gz"
OUTPUT_FILE="resources/datasets/raw/nq-data/nq-dev.tsv"
DATASET_TYPE="dev"  # Set to 'dev' or 'train' as needed
SAMPLE_SIZE=""  # Leave empty or set a valid integer if needed

# Run the Python script, always passing the --dataset_type argument
if [ -z "$SAMPLE_SIZE" ]; then
    python run_scripts/extract_nq_dataset.py --input_file_path "$INPUT_FILE" \
        --output_file_path "$OUTPUT_FILE" \
        --dataset_type "$DATASET_TYPE"
else
    python run_scripts/extract_nq_dataset.py --input_file_path "$INPUT_FILE" \
        --output_file_path "$OUTPUT_FILE" \
        --sample_size "$SAMPLE_SIZE" \
        --dataset_type "$DATASET_TYPE"
fi