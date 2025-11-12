#!/bin/sh

# ----------------------------
# Environment Setup
# ----------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ddro_env

# Set Python path to include the src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"

# ----------------------------
# Configuration
# ----------------------------
ENCODING="pq"        # 'pq' | 'url' 
DATASET="msmarco"          # 'msmarco' | 'nq'
SCALE="top_300k"      # e.g., 'top_300k' | 'rand_100k'

HF_DOCIDS_REPO="kiyam/ddro-docids"
HF_TESTS_REPO="kiyam/ddro-testsets"

echo "=== Evaluation Configuration ==="
echo "Dataset: $DATASET"
echo "Encoding: $ENCODING"
echo "Scale: $SCALE"
echo "DocIDs repo: $HF_DOCIDS_REPO"
echo "Tests repo: $HF_TESTS_REPO"
echo "================================"

# ----------------------------
# Run Evaluation
# ----------------------------
python src/pretrain/hf_eval/launch_hf_eval_from_config.py \
  --dataset "$DATASET" \
  --encoding "$ENCODING" \
  --scale "$SCALE" \
  --hf_docids_repo "$HF_DOCIDS_REPO" \
  --hf_tests_repo "$HF_TESTS_REPO"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Evaluation Completed Successfully ==="
else
    echo "=== Evaluation Failed with exit code $EXIT_CODE ==="
    exit $EXIT_CODE
fi