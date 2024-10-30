#!/bin/sh
#SBATCH --job-name=pretrain_search_finetune_pq_nq
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
##SBATCH --gres=gpu:nvidia_l40:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=64gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/pretrain_search_finetune_pq_nq_30epoch_log-%j.out # %j is the job ID
# Set up the environment.

source /home/kmekonn/.bashrc
conda activate ddro
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

python utils/nq_t5_training_pipeline.py \
                --encoding pq