#!/bin/bash
#SBATCH --job-name=create_nq_triples
#SBATCH --time=1-00:00:00
#SBATCH --mem=64gb
#SBATCH -c 48
#SBATCH --output=logs-slurm/create_nq_triples-%j.out

set -e  # Exit on error

# Activate environment if needed
source ~/.bashrc
conda activate ddro_env  




# Paths to your relevance, qrel, queries, and document files
RELEVANCE_PATH="resources/datasets/processed/msmarco-data/pyserini_data/nq_train_bm25tuned.txt"
QREL_PATH="resources/datasets/processed/nq-msmarco/nq_qrels_train.tsv.gz"
QUERY_PATH="resources/datasets/processed/nq-msmarco/nq_queries_train.tsv.gz"
DOCS_PATH="resources/datasets/processed/nq-msmarco/nq-merged-json/nq-docs-sents.json"
NUM_NEGS=1  # or whatever number you want

# Output directory where the results will be saved
OUTPUT_DIR="resources/datasets/processed/nq-data/nq_hard_negatives_format"
OUTPUT_FILE="$OUTPUT_DIR/nq_dev_triples_with_hard_negatives.txt"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR


# Run the script
python src/data/data_prep/nq/bm25_negative_Sampling_NQ.py \
  --relevance_path "$RELEVANCE_PATH" \
  --qrel_path "$QREL_PATH" \
  --output_path "$OUTPUT_PATH" \
  --num_negative_per_query "$NUM_NEGS" \
  --query_path "$QUERY_PATH" \
  --docs_path "$DOCS_PATH"
