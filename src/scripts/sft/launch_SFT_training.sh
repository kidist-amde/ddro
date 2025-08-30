#!/bin/sh
#SBATCH --job-name=SFT_wiz_url_id_training
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --mem=128gb
#SBATCH -c 16
#SBATCH --output=logs-slurm/training_logs/SFT_wiz_url_id_training_log-%j.out

# Activate environment and verify GPU
source ~/.bashrc
conda activate ddro
cd /path/to/DDRO-Direct-Document-Relevance-Optimization/ddro

nvidia-smi

#  The --encoding flag supports formats like pq, url, atomic, summary.

# Run DDRO training pipeline
python utils/run_training_pipeline.py \
    --encoding url 
    --scale top_300k
