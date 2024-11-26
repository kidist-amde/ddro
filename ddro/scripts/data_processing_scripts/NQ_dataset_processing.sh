#!/bin/sh
#SBATCH --job-name=nq_dataset_processing
##SBATCH --partition=staging # Accessible partition
#SBATCH --partition=cbuild
#SBATCH --ntasks=1          # Number of tasks
#SBATCH --time=1:00:00      # d-h:m:s
#SBATCH --mem=128gb         # Memory per job
#SBATCH -c 1             # Number of CPUs
#SBATCH --output=logs-slurm/other-logs/NQ_process_and_merge-%j.out # %j is the job ID

# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro


# Ensure the log directory exists
mkdir -p logs-slurm/other-logs

# Set input and output file paths
DEV_FILE="resources/datasets/raw/nq-data/v1.0-simplified_nq-dev-all.jsonl.gz"
TRAIN_FILE="resources/datasets/raw/nq-data/v1.0-simplified_simplified-nq-train.jsonl.gz"
OUTPUT_MERGED_FILE="resources/datasets/processed/nq-data/nq-merged/nq_docs.tsv"
OUTPUT_TRAIN_PATH="resources/datasets/processed/nq-data/nq_train.tsv"
OUTPUT_DEV_PATH="resources/datasets/processed/nq-data/nq_dev.tsv"

# Run the processing script
python data/data_preprocessing/NQ_dataset_processing.py \
    --dev_file $DEV_FILE \
    --train_file $TRAIN_FILE \
    --output_merged_file $OUTPUT_MERGED_FILE \
    --output_train_file $OUTPUT_TRAIN_PATH \
    --output_val_file $OUTPUT_DEV_PATH
