#!/bin/sh
#SBATCH --job-name=ultron_nq_pq
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
##SBATCH --gres=gpu:nvidia_l40:8 
##SBATCH --gres=gpu:tesla_p40:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=5-00:00:00 # d-h:m:s
#SBATCH --mem=32gb # memory per GPU 
#SBATCH -c 2 # number of CPUs
#SBATCH --output=logs-slurm-ultron/ultron_nq_pq_log-%j.out # %j is the job ID

# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro_env
nvidia-smi.

cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro


python utils/nq_t5_training_pipeline.py \
                --encoding pq