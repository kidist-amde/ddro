#!/bin/sh
#SBATCH --job-name=json #create_msmarco_triples_dataset
#SBATCH --time=3-00:00:00 # Reduce time if faster
#SBATCH --mem=256gb       # More memory for batching
#SBATCH -c 16             # Number of CPUs
#SBATCH --output=logs-slurm-msmarco-sft/other-logs/tsv_to_json-%j.out

python data/data_preprocessing/tsv_to_json.py