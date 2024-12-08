#!/bin/sh
#SBATCH --job-name=triples #create_msmarco_triples_dataset
#SBATCH --time=3-00:00:00 # Reduce time if faster
#SBATCH --mem=256gb       # More memory for batching
#SBATCH -c 16             # Number of CPUs
#SBATCH --output=logs-slurm/other-logs/create_msmarco_triples_dataset-%j.out

python data/data_preprocessing/create_msmarco_triples.py
