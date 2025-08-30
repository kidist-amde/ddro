#!/bin/bash
#SBATCH --job-name=gen_3stage_data
#SBATCH --time=8:00:00 # d-h:m:s
#SBATCH --mem=128gb 
#SBATCH -c 48 
#SBATCH --output=logs-slurm/%x_%j.out  


source /home/kmekonn/.bashrc
conda activate ddro_env


# make sure to change the encoding to either "url_title" or "pq"

echo "generating 3-stage training data"
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py --cur_data general_pretrain --encoding "url_title" 
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py --cur_data search_pretrain --encoding "url_title"
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py --cur_data finetune --encoding "url_title"
