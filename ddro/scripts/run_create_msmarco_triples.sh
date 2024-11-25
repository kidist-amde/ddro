#!/bin/sh
#SBATCH --job-name=create_msmarco_triples_dataset
#SBATCH --time=2:00:00 # d-h:m:s
#SBATCH --mem=64gb # memory per job
#SBATCH --output=logs-slurm/other-logs/]create_msmarco_triples_dataset-%j.out # %j is the job ID

python data/data_preprocessing/create_msmarco_triples.py