#!/bin/sh
#SBATCH --job-name=SFT-MS-summary  # Job name
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:8 
##SBATCH --gres=gpu:nvidia_l40:8 
##SBATCH --gres=gpu:tesla_p40:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=5-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 4 # number of CPUs
##SBATCH --output=logs-slurm-msmarco-sft/sft_url_msmarco_50epoch_Resumedfrom27_logs-%j.out # Log output with unique job ID
#SBATCH --output=logs-slurm-summaries/sft_summary_msmarco_40epoch_logs-%j.out # Log output with unique job ID



# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro_env
nvidia-smi

cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

python utils/train_sft_msmarco.py \
                --encoding summary \
                --scale top_300k

