#!/bin/sh
#SBATCH --job-name=finetune_docTTTTTquery
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=ilps-cn117,ilps-cn120,ilps-cn116,ilps-cn118,ilps-cn119
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --mem=128gb
#SBATCH -c 48
#SBATCH --output=logs-slurm/finetune_docTTTTTquery_on_NQ-%j.out

source /home/kmekonn/.bashrc
conda activate ddro_env

echo "Starting finetuning of docTTTTTquery on MSMARCO dataset..."

# Args
DATASET_PATH="your/path/to/nq_dataset.jsonl"  
DATASET_NAME="nq"
OUTPUT_FILE="resources/datasets/processed//${DATASET_NAME}-data.tsv.gz"

python src/pretrain/finetune_docTTTTTquery.py \
  --dataset_path "$DATASET_PATH" \
  --dataset_name "$DATASET_NAME" \
  --output_file "$OUTPUT_FILE"
