#!/bin/sh
#SBATCH --job-name=DDRO-PQ-Ultron  # Job name
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
##SBATCH --gres=gpu:nvidia_l40:8 
##SBATCH --gres=gpu:tesla_p40:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00 # d-h:m:s
#SBATCH --mem=64gb # memory per GPU 
#SBATCH -c 4 # number of CPUs
#SBATCH --output=logs-slurm-ultron-msmarco/DDRO_PQ_2epoch_ULTRON_lr5e-6_logs-%j.out # Log output with unique job ID


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
    --docid_path resources/ENCODED_DOC_IDs/t5_pq_msmarco.txt \
    --output_dir outputs_ULTRON_msmarco_DPO_ENHANCED_DSI/dpo_NewTripls/dpo_ckp_${ENCODING}_2epoch_lr1e-6 \
    --pretrain_model_path resources/transformer_models/t5-base \
    --use_origin_head False \
    --checkpoint_path outputs_ULTRON_msmarco_DPO_ENHANCED_DSI/t5_128_1_top_300k_pq_pretrain_search_finetune/model_9.pkl \
    --max_prompt_length 128 \
    --doc_lookup_path resources/datasets/raw/msmarco-data/msmarco-docs-lookup.tsv.gz \
    --train_queries_file resources/datasets/raw/msmarco-data/msmarco-doctrain-queries.tsv.gz \
    --dev_queries_file resources/datasets/raw/msmarco-data/msmarco-docdev-queries.tsv.gz \
  