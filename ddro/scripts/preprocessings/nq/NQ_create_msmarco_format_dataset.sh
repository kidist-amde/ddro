#!/bin/sh
#SBATCH --job-name=nq_dataset_processing
##SBATCH --partition=staging # Accessible partition
#SBATCH --partition=cbuild
#SBATCH --ntasks=1          # Number of tasks
#SBATCH --time=1:00:00      # d-h:m:s
#SBATCH --mem=64gb         # Memory per job
#SBATCH -c 1             # Number of CPUs
#SBATCH --output=logs-slurm/other-logs/NQ_create_msmarco_resumbling_dataset-%j.out # %j is the job ID


# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro

# Paths to your train and dev files
NQ_TRAIN_FILE="resources/datasets/processed/nq-data/nq_train.tsv.gz"
NQ_DEV_FILE="resources/datasets/processed/nq-data/nq_dev.tsv.gz"
# Output directory where the results will be saved
OUTPUT_DIR="resources/datasets/processed/nq-data/nq_msmarco_format"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Running the Python script
python data/data_preprocessing/NQ_create_msmarco_format_dataset.py \
  --nq_train_file $NQ_TRAIN_FILE \
  --nq_dev_file $NQ_DEV_FILE \
  --output_dir $OUTPUT_DIR
