#!/bin/bash
#SBATCH --job-name=generate_docid_embeddings
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=......  # Replace with the actual node name
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=64gb
#SBATCH -c 4
#SBATCH --output=logs-slurm/generate_docid_embeddings-%j.out


# Note: Make sure to replace the placeholder paths with actual paths in your environment.
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

# Print completion message
echo "Document ID encoding completed. Output saved to $OUTPUT_PATH."
# Deactivate the Python environment if needed
# deactivate
