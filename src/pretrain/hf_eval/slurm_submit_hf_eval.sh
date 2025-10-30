#!/bin/sh
#SBATCH --job-name=EVal_ddro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=128gb
#SBATCH -c 48
#SBATCH --output=logs-slurm-eval/Eval_HF_NQ_URL_%j.out

# ----------------------------
# Environment Setup
# ----------------------------
source ~/.bashrc
conda activate ddro_env

# Set Python path to include the src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"

# Display GPU info
echo "=== GPU Information ==="
nvidia-smi
echo "======================="

# ----------------------------
# Configuration
# ----------------------------
ENCODING="url"        # encoding: 'pq', 'url'
DATASET="nq"    # dataset: 'msmarco' or 'nq'
SCALE="top_300k"     # scale: 'top_300k'

echo "=== Evaluation Configuration ==="
echo "Dataset: $DATASET"

echo "Encoding: $ENCODING"
echo "Scale: $SCALE"
echo "================================"

# ----------------------------
# Run Evaluation
# ----------------------------
python src/pretrain/hf_eval/launch_hf_eval_from_config.py \
  --dataset "$DATASET" \
  --encoding "$ENCODING" \
  --scale "$SCALE"

echo "=== Evaluation Completed ==="