#!/bin/sh
#SBATCH --job-name=Eval_pretrain_search_finetune_pq
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
##SBATCH --gres=gpu:nvidia_l40:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=64gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/eval-logs/pq/Eval_pretrain_search_finetune_20_epoch_pq-%j.out # %j is the job ID
# Set up the environment.

source /home/kmekonn/.bashrc
conda activate ddro
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

python utils/test_nq_t5.py \
                --encoding pq