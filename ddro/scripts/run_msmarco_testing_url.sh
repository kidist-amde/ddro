#!/bin/sh
#SBATCH --job-name=Eval_pretrain_search_finetune_url_msmarco_NewrunT5
#SBATCH --partition gpu_h100
#SBATCH --gres=gpu:2
##SBATCH --partition gpu
##SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=16gb #120gb #180gb
#SBATCH -c 2
#SBATCH --output=logs-slurm/eval-logs/Eval_pretrain_search_finetune_url_NewrunT5-%j.out # %j is the job ID

# Set up the environment.
# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 


python utils/test_msmarco_t5_url.py \
                --encoding urlte