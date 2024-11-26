#!/bin/sh
#SBATCH --job-name=sft_nq_pq
#SBATCH --partition gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
##SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --mem=32gb #120gb #180gb
#SBATCH -c 16
#SBATCH --output=logs-slurm-sft-new/sft_nq_pq_logs-%j.out # %j is the job ID

# Set up the environment.
# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

python utils/train_sft.py \
                --encoding pq \
                --scale top_300k

