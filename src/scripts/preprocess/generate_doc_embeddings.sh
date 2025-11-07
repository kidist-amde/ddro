#!/bin/sh

INPUT_PATH="resources/datasets/processed/msmarco-docs-sents.top.300k.json.gz" 
OUTPUT_PATH="resources/datasets/processed/msmarco-data/ms_doc_embeddings/doc_embeddings.json.gz"
MODEL_NAME="sentence-transformers/gtr-t5-base"
BATCH_SIZE=256
DATASET="msmarco"   # Specify the dataset type (e.g., "msmarco", "nq", etc.)

# Run the Python script
python src/data/data_prep/generate_doc_embeddings.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --dataset "$DATASET" 
  
# Print completion message
echo "Document embeddings generation completed. Output saved to $OUTPUT_PATH."