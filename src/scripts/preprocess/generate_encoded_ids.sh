#!/bin/bash
#SBATCH --job-name=sample_top_docs
#SBATCH --time=1-00:00:00
#SBATCH --mem=64gb
#SBATCH -c 48
#SBATCH --output=logs-slurm/sample_top_ms_docs-%j.out


source /home/kmekonn/.bashrc
conda activate ddro_env


# Set paths to input files and output directory
INPUT_DOC_PATH="/path/to/input_document_file.jsonl"
INPUT_EMBED_PATH="/path/to/input_embedding_file.txt"
OUTPUT_PATH="/path/to/output/encoded_docid.txt"
PRETRAIN_MODEL_PATH="/path/to/transformer_models/t5-base"
SUMMARY_PATH="/path/to/data/summaries.json"

# Encoding method: choose from atomic, pq, url, or summary
ENCODING_METHOD="pq"

# Additional parameters for product quantization (if applicable)
SUB_SPACE=24
CLUSTER_NUM=256
BATCH_SIZE=1024

# Activate the Python environment if needed
# source /path/to/venv/bin/activate

# Run the Python script
python ./src/data/generate_instances/generate_encoded_docids.py \
  --encoding "$ENCODING_METHOD" \
  --input_doc_path "$INPUT_DOC_PATH" \
  --input_embed_path "$INPUT_EMBED_PATH" \
  --output_path "$OUTPUT_PATH" \
  --pretrain_model_path "$PRETRAIN_MODEL_PATH" \
  --summary_path "$SUMMARY_PATH" \
  --sub_space "$SUB_SPACE" \
  --cluster_num "$CLUSTER_NUM" \
  --batch_size "$BATCH_SIZE"

# Deactivate the Python environment if activated earlier
# deactivate