#!/bin/sh
#SBATCH --job-name=pretrain_search_finetune_url_ULTRON_
#SBATCH --partition gpu_h100
#SBATCH --gres=gpu:2
##SBATCH --partition gpu
##SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64gb #120gb #180gb
#SBATCH -c 16
#SBATCH --output=logs-slurm/eval-logs/ZEVAL_nq_url__ULTRON__NewRUNT5code-%j.out # %j is the job ID

# Set up the environment.
# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

python utils/test_nq_t5_url.py \
                --encoding url
