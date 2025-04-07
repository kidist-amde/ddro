#!/bin/bash
#SBATCH --job-name=msmarco_sample_docs
#SBATCH --output=logs-slurm/sample_top_docs-%j.out
#SBATCH --time=12:00:00
#SBATCH --mem=64gb
#SBATCH -c 16

# Description:
# This script processes the MS MARCO document collection, tokenizes passages,
# and generates top-k and random sampled subsets for training generative retrievers.

# Activate environment
source ~/.bashrc
conda activate ddro

# Navigate to project root
cd "$(dirname "$0")/../.."

# Run preprocessing
python ddro/data/preprocessing/preprocess_msmarco_documents.py
