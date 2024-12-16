#!/bin/sh
#SBATCH --job-name=SFT-MS-pq  # Job name
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
##SBATCH --gres=gpu:nvidia_l40:8 
##SBATCH --gres=gpu:tesla_p40:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-00:00:00 # d-h:m:s
#SBATCH --mem=256gb # memory per GPU 
#SBATCH -c 3 # number of CPUs
#SBATCH --output=logs-slurm-nq-sft/SFT_PQ_RETRIEVAL_TOP1000_logs-%j.out # Log output with unique job ID


# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro_env
nvidia-smi

cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

python utils/sft_infrence.py \
    --pretrain_model_path resources/transformer_models/t5-base \
    --model_checkpoint outputs-sft-NQ/t5_128_10_top_300k_nq_pq/model_final.pkl \
    --test_file_path resources/datasets/processed/nq-data/test_data/query_dev.t5_128_1.pq_nq.json \
    --docid_path resources/datasets/processed/nq-data/encoded_docid/t5_512_pq_docids.txt \
    --output_path outputs-sft-NQ/NQ-PQ-INFRENCES \
    --max_seq_length 64 \
    --max_docid_length {max_docid_length} \
    --batch_size 8 \
    --dataset_script_dir data/data_scripts \
    --dataset_cache_dir negs_tutorial_cache \
    --use_origin_head "True" \
    --dataset_script_dir data/data_scripts/json_builder.py
         

