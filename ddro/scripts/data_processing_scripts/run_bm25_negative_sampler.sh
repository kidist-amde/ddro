#!/bin/sh
#SBATCH --job-name=negativesampling
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=256gb # memory per GPU
#SBATCH -c 32 # number of CPUs
#SBATCH --output=logs-slurm-msmarco-sft/other-logs/MSMARCO-NEGATIVE-SAMPLER-DEV-%j.out # %j is the job ID

# Set up the environment
source /home/kmekonn/.bashrc
conda activate ddro_env
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

# Set variables
SUBSET="dev" # Options: "train" or "dev"
QUERIES_FILE="resources/datasets/raw/msmarco-data/msmarco-doc${SUBSET}-queries.tsv.gz"
QRELS_FILE="resources/datasets/raw/msmarco-data/msmarco-doc${SUBSET}-qrels.tsv.gz"
PYSERINI_FILE="resources/datasets/processed/msmarco-data/pyserini_data/msmarco_${SUBSET}_bm25tuned.txt"
DOCS_FILE="resources/datasets/raw/msmarco-data/msmarco-docs.tsv"
DOC_OFFSETS_FILE="resources/datasets/raw/msmarco-data/msmarco-docs-lookup.tsv.gz"
OUTFILE="resources/datasets/processed/msmarco-data/hard_negatives_from_bm25_top1000_retrieval/msmarco_${SUBSET}_triplets.txt"
N_NEGATIVES=16
LOG_FILE="logs-slurm-msmarco-sft/other-logs/MSMARCO-NEGATIVE-SAMPLER_${SUBSET}.log"

# Run the negative sampling script
python data/data_preprocessing/bm25_negative_sampling.py \
    --queries_file "$QUERIES_FILE" \
    --qrels_file "$QRELS_FILE" \
    --pyserini_file "$PYSERINI_FILE" \
    --docs_file "$DOCS_FILE" \
    --doc_offsets_file "$DOC_OFFSETS_FILE" \
    --outfile "$OUTFILE" \
    --logfile "$LOG_FILE" \
    --n_negatives "$N_NEGATIVES"

