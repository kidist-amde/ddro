#!/bin/sh
#SBATCH --job-name=nq_Pseudo_query
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:6
##SBATCH --gres=gpu:nvidia_l40:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/nq_Pseudo_query_generation-%j.out # %j is the job ID
# Set up the environment.

source /home/kmekonn/.bashrc
conda activate ddro
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

# Define paths and parameters
CACHE_DIR="cache"
DATA_FILE="resources/datasets/processed/nq-data/nq-merged-json/nq-docs-sents.json"
OUTPUT_PATH="resources/datasets/processed/nq-data/nq_pseudo_query_10_2epoch.txt"
BATCH_SIZE=64
MAX_LENGTH=256
TOP_K=10
NUM_RETURN_SEQUENCES=10
Q_MAX_LENGTH=32
MAX_DOCS=109739 # Set the maximum number of documents to process
CHECKPOINT_PATH="resources/transformer_models/docTTTTTquery_finetuned/finetuned_doc2query_t5_large_msmarco"


# Run the Python script to generate pseudo-queries with all arguments
python data/generate_instances/nq_doc2query_t5_query_generator.py \
    --input_file "$DATA_FILE" \
    --cache_dir "$CACHE_DIR" \
    --output_path "$OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --top_k $TOP_K \
    --num_return_sequences $NUM_RETURN_SEQUENCES \
    --q_max_length $Q_MAX_LENGTH \
    --max_docs $MAX_DOCS \
    --checkpoint_path "$CHECKPOINT_PATH" \
