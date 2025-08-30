#!/bin/bash
#SBATCH --job-name=Eval_HF
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00 # d-h:m:s
#SBATCH --mem=128gb 
#SBATCH -c 48 
#SBATCH --output=logs-slurm-eval/%x_%j.out  # ensure dir exists *before* sbatch


# ----------------------------
# Environment
# ----------------------------
source ~/.bashrc
conda activate ddro_env

# Add repo + src to PYTHONPATH (why: import utils/* modules)
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"

nvidia-smi 
encoding="url_title" # Choose from: "url_title", "pq"

# Launch the dataset-aware HF evaluator
python src/pretrain/hf_eval/eval_hf_docid_ranking.py \
  --per_gpu_batch_size 4 \
  --log_path logs/msmarco/dpo_HF_url.log \
  --pretrain_model_path kiyam/ddro-msmarco-tu \
  --docid_path resources/datasets/processed/msmarco-data/encoded_docid/${encoding}_docid.txt \
  --test_file_path resources/datasets/processed/msmarco-data/eval_data/query_dev.${encoding}.jsonl \
  --dataset_script_dir src/data/data_scripts \
  --dataset_cache_dir ./cache \
  --num_beams 15 \
  --add_doc_num 6144 \
  --max_seq_length 64 \
  --max_docid_length 100 \
  --use_docid_rank True \
  --docid_format msmarco \
  --lookup_fallback True \
  --device cuda:0
