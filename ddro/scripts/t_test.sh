#!/bin/sh

# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro

# Correct variable assignments (no spaces around `=`)
model1="ULTRON"
model2="DDRO"
encoding="url_title"  # pq or atomic or url_title
dataset="nq"  # msmarco or nq
folder_path="logs-slurm-Report/t_test"

# Ensure the folder exists before running the script
mkdir -p ${folder_path}

# Execute the Python script
python utils/statistical_tests.py \
    --metrics_file1 logs-slurm-Report/${model1}-${dataset}-${encoding}.csv \
    --metrics_file2 logs-slurm-Report/${model2}-${dataset}-${encoding}.csv \
    --metrics_to_test MRR@10 Hit@1 Hit@5 Hit@10 \
    --log_path ${folder_path} \
    --output_file ${folder_path}/${model1}-${encoding}_VS_${model2}-${encoding}.csv \
    --comparison_image ${folder_path}/${model1}-${encoding}_VS_${model2}-${encoding}.png \
    --distribution_image ${folder_path}/${model1}-${encoding}_VS_${model2}-${encoding}.png
