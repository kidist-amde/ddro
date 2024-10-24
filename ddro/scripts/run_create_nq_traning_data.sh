#!/bin/sh
#SBATCH --job-name=nq_create_msmarco_resumbling_dataset
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per job
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/nq_create_msmarco_resumbling_dataset-%j.out # %j is the job ID

# Set up the environment
source /home/kmekonn/.bashrc
conda activate ddro

# Paths to your train and dev files
NQ_TRAIN_FILE="resources/datasets/processed/nq-data/nq_train.tsv.gz"
NQ_DEV_FILE="resources/datasets/processed/nq-data/nq_dev.tsv.gz"

# Output directory where the results will be saved
OUTPUT_DIR="resources/datasets/processed/nq-data/nq_msmarco_format"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Running the Python script
python data/data_preprocessing/create_nq_training_data.py \
  --nq_train_file $NQ_TRAIN_FILE \
  --nq_dev_file $NQ_DEV_FILE \
  --output_dir $OUTPUT_DIR
