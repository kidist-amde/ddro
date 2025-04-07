#!/bin/sh
#SBATCH --job-name=SFT-summary-MS  # Job name
#SBATCH --partition=gpu 
##SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --gres=gpu:nvidia_l40:4
##SBATCH --gres=gpu:tesla_p40:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 48 # number of CPUs
##SBATCH --output=logs-slurm-sft/sft_url_nq_40epoch_logs-%j.out # Log output with unique job ID
#SBATCH --output=logs-slurm-summaries/sft_summary_resumed_FromUltron_logs-%j.out # Log output with unique job ID


# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro_env
nvidia-smi

cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

python utils/train_sft_msmarco.py \
                --encoding summary\
                --scale top_300k

