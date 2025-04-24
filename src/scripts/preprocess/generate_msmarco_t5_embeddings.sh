#!/bin/bash

# Script to run generate_doc_embeddings.py

# Set default values for parameters
INPUT_PATH="/path/to/input/data"  # Replace with the actual input data path
OUTPUT_PATH="/path/to/output/embeddings"  # Replace with the desired output path
MODEL_NAME="sentence-transformers/gtr-t5-base"
BATCH_SIZE=128
DATASET="nq"  # Change to 'msmarco' if needed

# Activate Python environment if necessary
# source /path/to/your/virtualenv/bin/activate

# Run the Python script
python ./src/data/preprocessing/generate_doc_embeddings.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --dataset "$DATASET"

# Print completion message
echo "Document embeddings generation completed. Output saved to $OUTPUT_PATH."