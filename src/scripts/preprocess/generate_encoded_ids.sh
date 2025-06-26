#!/bin/bash
#SBATCH --job-name=generate_encoded_ids
#SBATCH --time=1-00:00:00
#SBATCH --mem=256gb
#SBATCH -c 48
#SBATCH --output=logs-slurm/generate_MSDOC_encoded_ids-%j.out


source /home/kmekonn/.bashrc
conda activate ddro_env



# Encoding method: choose from (atomic, pq, url, or summary) only pq and url reported on our paper 

ENCODING_METHOD="pq"  # Options: atomic, pq, url, summary


# Set paths to input files and output directory
INPUT_DOC_PATH=resources/datasets/processed/msmarco-docs-sents.top.300k.json.gz
INPUT_EMBED_PATH=resources/datasets/processed/msmarco-data/ms_doc_embeddings/doc_embeddings.json.gz
OUTPUT_PATH=resources/datasets/processed/msmarco-data/encoded_docid/${ENCODING_METHOD}_docid.txt
PRETRAIN_MODEL_PATH="t5-base"
SUMMARY_PATH=data/summaries.json



# Additional parameters for product quantization (if applicable)
SUB_SPACE=24
CLUSTER_NUM=256
BATCH_SIZE=1024


# Run the Python script
python src/data/data_prep/generate_encoded_docids.py \
  --encoding "$ENCODING_METHOD" \
  --input_doc_path "$INPUT_DOC_PATH" \
  --input_embed_path "$INPUT_EMBED_PATH" \
  --output_path "$OUTPUT_PATH" \
  --pretrain_model_path "$PRETRAIN_MODEL_PATH" \
  --summary_path "$SUMMARY_PATH" \
  --sub_space "$SUB_SPACE" \
  --cluster_num "$CLUSTER_NUM" \
  --batch_size "$BATCH_SIZE" 

