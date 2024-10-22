#!/bin/sh
#SBATCH --job-name=msmarco_dataset_sampling
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per job
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/msmarco_dataset_sampling-%j.out # %j is the job ID

# Set up the environment
source /home/kmekonn/.bashrc
conda activate ddro

python data/data_preprocessing/sample_msmarco_datase.py \
                --doc_file  /resources/datasets/raw/msmarco-data/msmarco-docs.tsv.gz \
                --qrels_train resources/datasets/raw/msmarco-data/msmarco-doctrain-qrels.tsv.gz \
                --scale "300k"
