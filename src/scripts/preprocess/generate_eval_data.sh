#!/bin/sh
#SBATCH --job-name=gen_eval_data
#SBATCH --time=1-00:00:00
#SBATCH --mem=128gb
#SBATCH -c 48
#SBATCH --output=logs-slurm/gen_eval_data_%j.out

source /home/kmekonn/.bashrc
conda activate ddro_env


echo "Generating evaluation data..."

python src/data/data_prep/build_t5_data/gen_eval_data_pipline.py --encoding "url_title"

