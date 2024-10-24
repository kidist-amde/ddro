#!/bin/sh
#SBATCH --job-name=nq_create_hard_negatives_dev_dataset
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per job
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/nq_create_hard_negatives_dev_dataset-%j.out # %j is the job ID

# Set up the environment
source /home/kmekonn/.bashrc
conda activate ddro

# Paths to your relevance, qrel, queries, and document files
RELEVANCE_PATH="resources/datasets/processed/nq-data/nq_msmarco_format/nq_dev_bm25tuned.txt"
QREL_PATH="resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_dev.tsv.gz"
QUERY_PATH="resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_dev.tsv.gz"
DOCS_PATH="resources/datasets/processed/nq-data/nq-merged-json/nq-docs-sents.json"

# Output directory where the results will be saved
OUTPUT_DIR="resources/datasets/processed/nq-data/nq_hard_negatives_format"
OUTPUT_FILE="$OUTPUT_DIR/nq_dev_triples_with_hard_negatives.txt"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Running the Python script to create triples with hard negatives
python data/data_preprocessing/create_nq_triples.py \
  --relevance_path $RELEVANCE_PATH \
  --qrel_path $QREL_PATH \
  --query_path $QUERY_PATH \
  --docs_path $DOCS_PATH \
  --output_path $OUTPUT_FILE \
  --num_negative_per_query 1
