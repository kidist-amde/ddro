#!/bin/bash
#SBATCH --job-name=nq_preprocess
#SBATCH --time=1-00:00:00
#SBATCH --mem=256gb
#SBATCH -c 48
#SBATCH --output=logs-slurm/nq_preprocess-%j.out

set -e

source /home/kmekonn/.bashrc
conda activate ddro_env

# Input files
DEV_FILE="resources/datasets/raw/nq-data/nq-dev-all.jsonl.gz"
TRAIN_FILE="resources/datasets/raw/nq-data/simplified-nq-train.jsonl.gz"

# Output directory
OUTPUT_DIR="resources/datasets/processed/nq-data"
mkdir -p "$OUTPUT_DIR"

# Output file names (without extensions)
MERGED_FILE="${OUTPUT_DIR}/nq_merged"
TRAIN_FILE_OUT="${OUTPUT_DIR}/nq_train"
VAL_FILE_OUT="${OUTPUT_DIR}/nq_val"

# Run the preprocessing script
python src/data/data_prep/nq/process_nq_dataset.py \
  --dev_file "$DEV_FILE" \
  --train_file "$TRAIN_FILE" \
  --output_merged_file "$MERGED_FILE" \
  --output_train_file "$TRAIN_FILE_OUT" \
  --output_val_file "$VAL_FILE_OUT" \
  --output_json_dir "$OUTPUT_DIR"
