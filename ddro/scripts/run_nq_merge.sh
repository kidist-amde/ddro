#!/bin/sh
#SBATCH --job-name=nq_dataset_merge           
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per job
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/nq_extract_and_merge_nqdataset-%j.out # %j is the job ID

# Set up the environment
source /home/kmekonn/.bashrc
conda activate ddro

# Set input and output file paths
DEV_FILE="resources/datasets/raw/nq-data/v1.0-simplified_nq-dev-all.jsonl.gz"
TRAIN_FILE="resources/datasets/raw/nq-data/v1.0-simplified_simplified-nq-train.jsonl.gz"
OUTPUT_FILE="resources/datasets/raw/nq-data/nq_merged.tsv"
DOC_CONTENT_FILE="resources/datasets/raw/nq-data/nq_doc_content.tsv"

python data/data_preprocessing/extract_and_merge_nq_datasets.py \
    --dev_file $DEV_FILE \
    --train_file $TRAIN_FILE \
    --output_file $OUTPUT_FILE \
    # --doc_content_file $DOC_CONTENT_FILE


