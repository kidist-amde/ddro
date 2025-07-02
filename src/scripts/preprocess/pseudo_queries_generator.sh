#!/bin/sh
#SBATCH --job-name=psuedo_query_gen
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=ilps-cn117,ilps-cn120,ilps-cn116,ilps-cn118,ilps-cn119
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --mem=128gb
#SBATCH -c 48
#SBATCH --output=logs-slurm/psuedo_query_gen_ms-%j.out

source /home/kmekonn/.bashrc
conda activate ddro_env

dataset_name="nq"

python generate_pseudo_queries.py \
  --input_file data/temporal_splits_${dataset_name}/D0_train_filtered_with_sents.jsonl \
  --checkpoint_path resources/checkpoints/finetuned_docTTTTTquery_on_${dataset_name} \
  --output_path resources/datasets/processed/${dataset_name}-data/${dataset_name}_pseudo_queries.jsonl \
  --batch_size 128 \
  --max_docs None \
  --num_return_sequences 10