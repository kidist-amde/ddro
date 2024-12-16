#!/bin/sh
#SBATCH --job-name=DDRO-pq  # Job name
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
##SBATCH --gres=gpu:nvidia_l40:8 
##SBATCH --gres=gpu:tesla_p40:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00 # d-h:m:s
#SBATCH --mem=64gb # memory per GPU 
#SBATCH -c 4 # number of CPUs
#SBATCH --output=logs-slurm-msmarco-sft/DDRO_pq_5epoch_lr1e-5_BETA_049_logs-%j.out # Log output with unique job ID


# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro_env
nvidia-smi

cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro


# --max_prompt_length 64 \

FILE_PATH=resources/datasets/processed/msmarco-data
ENCODING=pq #pq #url_title #atomic

# Run the Python training script
python pretrain/train_dpo_model.py \
    --train_file $FILE_PATH/hard_negatives_from_bm25_top1000_retrieval/msmarco_train_triplets.txt \
    --dev_file $FILE_PATH/hard_negatives_from_bm25_top1000_retrieval/msmarco_dev_triplets.txt \
    --doc_lookup_path resources/datasets/raw/msmarco-data/msmarco-docs-lookup.tsv.gz \
    --train_queries_file resources/datasets/raw/msmarco-data/msmarco-doctrain-queries.tsv.gz \
    --dev_queries_file resources/datasets/raw/msmarco-data/msmarco-docdev-queries.tsv.gz \
    --docid_path $FILE_PATH/encoded_docid/t5_512_pq_top.300k.txt \
    --output_dir outputs-sft-msmarco/dpo/dpo_ckp_${ENCODING}_5epoch_lr1e-5_BETA_049 \
    --pretrain_model_path resources/transformer_models/t5-base \
    --use_origin_head False \
    --max_prompt_length 128 \
    --checkpoint_path outputs-sft-msmarco/t5_128_10_top_300k_msmarco_pq/model_final.pkl \
