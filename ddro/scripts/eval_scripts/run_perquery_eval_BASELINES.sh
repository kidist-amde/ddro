#!/bin/sh
#SBATCH --job-name=EVal_SFT-nq-pq  # Job name
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:8
##SBATCH --gres=gpu:nvidia_l40:8 
##SBATCH --gres=gpu:tesla_p40:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=32gb # memory per GPU 
#SBATCH -c 2 # number of CPUs
#SBATCH --output=BASELINES/logs-slurm/Eval_LTRG-NQ-URL_logs_10_BEAM-%j.out # Log output with unique job ID


# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro_env
nvidia-smi

cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro



python utils/test_t5_run_query_metrics_BASELINE_MODELS_CKP.py \
                --encoding url \
                --scale top_300k             