#!/bin/sh
#SBATCH --job-name=EVal_ddro-nq-url # Job name
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 8 # number of CPUs
#SBATCH --output=logs-slurm-sft-nq/Eval_DDRO_url_NQ__80_BEAMSIZE_lr1e-5_NEWTRIPLS_logs-%j.out # Log output with unique job ID


# Set up the environment.
source ~/.bashrc
conda activate ddro_env
nvidia-smi

cd ..


python test_dpo_run_query_metrics.py --dataset nq --encoding pq --scale top_300k


                
