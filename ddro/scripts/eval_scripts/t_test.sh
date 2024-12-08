#!/bin/sh
#SBATCH --job-name=t_test  # Job name
##SBATCH --partition=gpu 
##SBATCH --gres=gpu:nvidia_rtx_a6000:1
##SBATCH --gres=gpu:nvidia_l40:1
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm-sft/other-logs/t_test_sft_vs_ddro_url-%j.out # %j is the job ID
# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro_env
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

# Correct variable assignments (no spaces around `=`)
model1="SFT" # SFT OR ULTRON
model2="DDRO"
encoding="url_title"  # pq or atomic or url_title
dataset="nq"  # msmarco or nq
folder_path="logs-sft/t_tests"

# Ensure the folder exists before running the script
mkdir -p ${folder_path}

# Execute the Python script
python utils/statistical_tests.py \
    --metrics_file1 logs-sft/SFT-nq-url_title_20epoch.csv \
    --metrics_file2 logs-sft/dpo/DDRO-nq-url_title_from_20epochSFT.csv \
    --metrics_to_test MRR@10 Hit@1 Hit@5 Hit@10 \
    --log_path ${folder_path} \
    --output_file ${folder_path}/${model1}-${encoding}_VS_${model2}-${encoding}.csv \
    --comparison_image ${folder_path}/${model1}-${encoding}_VS_${model2}-${encoding}.png \
    --distribution_image ${folder_path}/${model1}-${encoding}_VS_${model2}-${encoding}.png
