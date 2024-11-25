#!/bin/sh
#SBATCH --job-name=summarizer       # Job name
#SBATCH --partition gpu_h100         # Partition name
#SBATCH --gres=gpu:1                 # Number of GPUs needed
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of tasks per node
#SBATCH --time=5-00:00:00            # Time limit hrs:min:sec
#SBATCH --mem=32gb                   # Memory limit
#SBATCH -c 4                        # Number of CPUs
#SBATCH --output=logs-slurm-summary/llama3_summarizer_logs-%j.out

# Load environment
source ${HOME}/.bashrc
conda activate open_ai_env

# Set CUDA memory management strategy (check compatibility)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Alternative to expandable_segments

# Ensure the log directory exists
mkdir -p logs-slurm/other-logs
nvidia-smi
# Change to the project directory
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro

# Define paths and parameters
DATA_FILE="resources/datasets/processed/nq-data/nq-merged/nq_merged.tsv.gz"
OUTPUT_DIR="resources/datasets/processed/nq-data"
OUTPUT_FILENAME="NQ_openai_api_summaries.jsonl"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILENAME"
BATCH_SIZE=16
MAX_DOCS=109739  

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the Python script for document summarization with all arguments
python data/data_preprocessing/nq_summrizer_openai.py \
    --input_file "$DATA_FILE" \
    --output_path "$OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    --max_docs $MAX_DOCS
