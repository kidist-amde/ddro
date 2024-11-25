#!/bin/sh
#SBATCH --job-name=sft_url
#SBATCH --partition gpu_h100
#SBATCH --gres=gpu:1
##SBATCH --partition gpu
##SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=32gb #120gb #180gb
#SBATCH -c 4
#SBATCH --output=logs-slurm-sft/NQ_Query_metrics_sft_url-%j.out # %j is the job ID

# Set up the environment.
# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 


python utils/test_t5_run_query_metrics.py \
                --encoding url \
                --scale top_300k