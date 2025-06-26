#!/bin/sh
#SBATCH --job-name=generate_doc_embeddings
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=ilps-cn117,ilps-cn120,ilps-cn116,ilps-cn118,ilps-cn119
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --mem=128gb
#SBATCH -c 48
#SBATCH --output=logs-slurm/generate_MS_doc_embeddings-%j.out

source /home/kmekonn/.bashrc
conda activate ddro_env


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
