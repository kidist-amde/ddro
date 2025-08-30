#!/bin/sh
#SBATCH --job-name=EVal_ddro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00 # d-h:m:s
#SBATCH --mem=128gb 
#SBATCH -c 48 
#SBATCH --output=logs-slurm-eval/Eval_DDRO_%j.out # Log output with unique job ID

# ----------------------------
# Environment
# ----------------------------
source ~/.bashrc
conda activate ddro_env

# Set Python path to include the src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"

nvidia-smi

# ----------------------------
# Configuration (edit or export as env vars before sbatch)
#   DATASET:   msmarco | nq
#   ENCODING:  pq  | url
#   SCALE:     top_300k 
# ----------------------------

DATASET=${DATASET:-msmarco}
ENCODING=${ENCODING:-url}
SCALE=${SCALE:-top_300k}

echo "Running evaluation with:"
echo "  DATASET=$DATASET"
echo "  ENCODING=$ENCODING" 
echo "  SCALE=$SCALE"
echo "  PYTHONPATH=$PYTHONPATH"


python src/pretrain/launch_ddro_eval_from_config.py \
  --dataset "$DATASET" \
  --encoding "$ENCODING" \
  --scale "$SCALE"