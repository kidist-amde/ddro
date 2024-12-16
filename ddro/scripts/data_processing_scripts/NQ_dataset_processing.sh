#!/bin/sh
#SBATCH --job-name=NQPreprocessing
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm-sft-nq/other-logs/NQ_Preprocessing-%j.out # %j is the job ID

# Set up the environment
source /home/kmekonn/.bashrc
conda activate ddro_env

# Navigate to the project directory
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro


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
    --output_val_file $OUTPUT_DEV_PATH \
    --output_json_dir "resources/datasets/processed/nq-data/nq-merged-json"
