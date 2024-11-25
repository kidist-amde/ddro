#!/bin/sh
#SBATCH --job-name=DDRO-atomic-2epo
##SBATCH --partition gpu_h100
##SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=16gb #120gb #180gb
#SBATCH -c 8
#SBATCH --output=logs-slurm-sft/EVAL_DDRO-2epoch_url_lr5e-7_logs-%j.out # %j is the job ID

# Set up the environment.
# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

python utils/test_dpo_run_query_metrics.py \
                --encoding url
              

                
