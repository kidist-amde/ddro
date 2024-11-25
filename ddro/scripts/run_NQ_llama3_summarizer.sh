#!/bin/sh
#SBATCH --job-name=LLAMA3_Summarizer
##SBATCH --partition gpu_h100
##SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1                                   
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
##SBATCH --mem=360gb
#SBATCH --mem=128gb
#SBATCH -c 32
#SBATCH --output=logs-slurm/other-logs/llama3_summarizer_logs-%j.out

# Load environment
source ${HOME}/.bashrc
conda activate llama-dsi

# Set CUDA memory management strategy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



# Change to the project directory
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

# Define paths and parameters
DATA_FILE="resources/datasets/processed/nq-data/nq-merged/nq_merged.tsv.gz"
OUTPUT_DIR="resources/datasets/processed/nq-data"
OUTPUT_FILENAME="NQ_llama3_summaries.jsonl"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILENAME"
BATCH_SIZE=64
MAX_NEW_TOKENS=128 
MAX_DOCS=109_739

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the Python script for document summarization with all arguments
python data/data_preprocessing/NQ_llama3_summarizer.py \
    --input_file "$DATA_FILE" \
    --output_path "$OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --max_docs $MAX_DOCS
