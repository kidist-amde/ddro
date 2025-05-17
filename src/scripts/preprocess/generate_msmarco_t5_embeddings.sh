#!/bin/sh
#SBATCH --job-name= generate_doc_embeddings
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=......  # Replace with the actual node name
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=64gb
#SBATCH -c 4
#SBATCH --output=logs-slurm/generate_doc_embeddings-%j.out

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