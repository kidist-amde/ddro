#!/bin/sh
#SBATCH --job-name=e
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=256gb # memory per GPU
#SBATCH -c 32 # number of CPUs
#SBATCH --output=logs-slurm-msmarco-sft/other-logs/explore_msmarco_dataset-%j.out # %j is the job ID

# Set up the environment
source /home/kmekonn/.bashrc
conda activate ddro_env
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

# Set variables



# python data/explore_msmarco_dataset.py


python data/explore_msmarco_dataset.py \
    --docs_file resources/datasets/raw/msmarco-data/msmarco-docs.tsv.gz \
    --offsets_file resources/datasets/raw/msmarco-data/msmarco-docs-lookup.tsv.gz \
    --output_file logs-slurm-msmarco-sft/missing_doc_ids.txt
