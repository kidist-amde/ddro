#!/bin/sh
#SBATCH --job-name=ddro-nq-pq                   # Job name
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00                       # d-h:m:s
#SBATCH --mem=64gb                              # Memory per GPU
#SBATCH -c 4                                     # Number of CPUs
#SBATCH --output=logs/ddro_nq_pq_train-%j.out   # Log output with job ID

# Environment setup
source ~/.bashrc
conda activate ddro_env
nvidia-smi

cd ddro  # Ensure working directory is set correctly

# Configurable paths
FILE_PATH=resources/datasets/processed/nq-data
ENCODING=pq  # Options: pq | url_title 

# Run NQ DDRO training script
python pretrain/train_ddro_model_NQ.py \
    --train_file $FILE_PATH/hard_negatives_from_bm25_top1000_retrieval/nq_train_triples.tsv \
    --dev_file $FILE_PATH/hard_negatives_from_bm25_top1000_retrieval/nq_dev_triples.tsv \
    --docid_path $FILE_PATH/encoded_docid/t5_512_${ENCODING}_docids.txt \
    --output_dir outputs/nq/ddro_ckpt_${ENCODING}_5epoch_lr5e-7 \
    --pretrain_model_path resources/transformer_models/t5-base \
    --use_origin_head False \
    --checkpoint_path outputs/nq/model_final_checkpoint.pkl \
    --max_prompt_length 128
