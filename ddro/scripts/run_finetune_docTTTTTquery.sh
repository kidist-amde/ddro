#!/bin/sh
#SBATCH --job-name=finetune_docTTTTTquery
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
##SBATCH --gres=gpu:nvidia_l40:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/finetune_docTTTTTquery_on_nq-%j.out # %j is the job ID
# Set up the environment.

source /home/kmekonn/.bashrc
conda activate ddro
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

DATASET_PATH=resources/datasets/raw/nq-data/nq_merged.tsv.gz
OUTPUT_FILE=resources/datasets/raw/nq-data/qcontent_train_512.tsv.gz

python pretrain/finetune_docTTTTTquery.py \
    --dataset_path $DATASET_PATH \
    --output_file $OUTPUT_FILE