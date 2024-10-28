#!/bin/sh

#SBATCH --job-name=DDROTesting5epcReport
#SBATCH --partition=gpu 
##SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=64gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm-dpo/EVAL_DDRO-PQ_7epoch_lr5e-6_beta_049_logs-%j.out # %j is the job ID
# Set up the environment.

source /home/kmekonn/.bashrc
conda activate ddro
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

python utils/test_dpo.py \
                --encoding pq \
                --scale top_300k

                