#!/bin/sh
#SBATCH --job-name=ddro_training_url_msmarco
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --mem=128gb
#SBATCH -c 16
#SBATCH --output=logs-slurm/training_logs/ddro_training_url_msmarco_log-%j.out

# Activate environment and verify GPU
source ~/.bashrc
conda activate ddro
cd /path/to/DDRO-Direct-Document-Relevance-Optimization/ddro

nvidia-smi

# Run DDRO training pipeline
python utils/run_training_pipeline.py \
    --encoding url \
    --scale top_300k
