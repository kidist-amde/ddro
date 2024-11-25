#!/bin/sh
#SBATCH --job-name=sft_nq_url
#SBATCH --partition gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
##SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --mem=128gb #120gb #180gb
#SBATCH -c 16
#SBATCH --output=logs-slurm-sft/sft_nq_url_logs-%j.out # %j is the job ID

# Set up the environment.
# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

python utils/train_sft.py \
                --encoding url \
                --scale top_300k

