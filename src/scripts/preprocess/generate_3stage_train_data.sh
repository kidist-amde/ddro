#!/bin/sh
#SBATCH --job-name=gen_3stage_train_data_pq
#SBATCH --time=1-00:00:00 
#SBATCH --mem=128gb
#SBATCH -c 48
#SBATCH --output=logs-slurm/gen_3stage_train_data_pq-%j.out

set -e

source /home/kmekonn/.bashrc
conda activate ddro_env

ENCODING_METHOD="pq" # Options: "pq", "url", ...
MAX_SEQ_LENGTH=128
SCALE="300k"

echo "Generating general_pretrain data..."
python src/data/data_prep/generate_train_data_wrapper.py \
  --encoding "$ENCODING_METHOD" \
  --scale "$SCALE" \
  --cur_data "general_pretrain" \
  --max_seq_length "$MAX_SEQ_LENGTH"

echo "Generating search_pretrain training data..."
python src/data/data_prep/generate_train_data_wrapper.py \
  --encoding "$ENCODING_METHOD" \
  --scale "$SCALE" \
  --cur_data "search_pretrain" \
  --max_seq_length "$MAX_SEQ_LENGTH"

echo "Generating finetune training data..."
python src/data/data_prep/generate_train_data_wrapper.py \
  --encoding "$ENCODING_METHOD" \
  --scale "$SCALE" \
  --cur_data "finetune" \
  --max_seq_length "$MAX_SEQ_LENGTH"
