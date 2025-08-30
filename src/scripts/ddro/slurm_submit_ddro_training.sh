#!/bin/sh
#SBATCH --job-name=ddro-pq          # Job name
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1                  # One GPU (A100)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00             # Max time (D-HH:MM:SS)
#SBATCH --mem=64gb                    # Memory per node
#SBATCH -c 4                          # Number of CPU cores
#SBATCH --output=logs/ddro_pq_run-%j.out  # Log output with job ID

# Activate environment
source ~/.bashrc
conda activate ddro_env
nvidia-smi

# Move to project directory
cd /path/to/DDRO-Direct-Document-Relevance-Optimization/ddro

# Set dataset configuration
FILE_PATH=resources/datasets/processed/msmarco-data
ENCODING=pq  # options: pq | url_title 

# Run training script
python pretrain/train_ddro_encoder_decoder.py \
  --train_file $FILE_PATH/hard_negatives_from_bm25_top1000_retrieval/msmarco_train_triplets.txt \
  --dev_file $FILE_PATH/hard_negatives_from_bm25_top1000_retrieval/msmarco_dev_triplets.txt \
  --doc_lookup_path resources/datasets/raw/msmarco-data/msmarco-docs-lookup.tsv.gz \
  --train_queries_file resources/datasets/raw/msmarco-data/msmarco-doctrain-queries.tsv.gz \
  --dev_queries_file resources/datasets/raw/msmarco-data/msmarco-docdev-queries.tsv.gz \
  --docid_path $FILE_PATH/encoded_docid/t5_512_pq_top.300k.txt \
  --output_dir outputs-sft-msmarco/ddro/ddro_ckp_${ENCODING}_5epoch_lr1e-5_BETA_049 \
  --pretrain_model_path resources/transformer_models/t5-base \
  --use_origin_head False \
  --max_prompt_length 128 \
  --checkpoint_path outputs-sft-msmarco/t5_128_10_top_300k_msmarco_pq/model_final.pkl
