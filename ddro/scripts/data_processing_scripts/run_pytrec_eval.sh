#!/bin/sh
#SBATCH --job-name=bm25eval
##SBATCH --partition=gpu 
##SBATCH --gres=gpu:nvidia_rtx_a6000:1
##SBATCH --gres=gpu:nvidia_l40:1
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=256gb # memory per GPU 
#SBATCH -c 32 # number of CPUs
#SBATCH --output=logs-slurm-msmarco-sft/other-logs/msmarco_dataset_BM25_custom_eval-%j.out # %j is the job ID

# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro_env
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro


# python data/data_preprocessing/pytrec_eval_eval.py
python data/data_preprocessing/evaluate.py