#!/bin/sh
#SBATCH --job-name=pretrain_search_finetune_pq_nq_ULTRON
#SBATCH --partition gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
##SBATCH --gres=gpu:a100:2
#SBATCH --job-name=pretrain_search_finetune_pq_nq_30epoch
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
##SBATCH --gres=gpu:nvidia_l40:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --mem=128gb #120gb #180gb
#SBATCH -c 16
#SBATCH --output=logs-slurm/traning_logs/pretrain_search_finetune_pq_nq_ULTRON_log-%j.out # %j is the job ID

# Set up the environment.
# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate dsi-env
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

python utils/nq_t5_training_pipeline.py \
                --encoding pq