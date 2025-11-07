#!/bin/bash

# Inputs
QUERIES_FILE="resources/datasets/raw/msmarco-data/msmarco-doctrain-queries.tsv.gz"
QRELS_FILE="resources/datasets/raw/msmarco-data/msmarco-doctrain-qrels.tsv.gz"
PYSERINI_FILE="resources/datasets/processed/msmarco-data/pyserini_data/msmarco_train_bm25tuned.txt"
DOCS_FILE="resources/datasets/processed/msmarco-data/msmarco-docs-sents.top.300k.json.gz"
DOC_OFFSETS_FILE="resources/datasets/raw/msmarco-data/msmarco-docs-lookup.tsv.gz"

# Output
OUTFILE="resources/datasets/processed/msmarco-data/triples/msmarco_hard_negatives.txt"
LOGFILE="logs/msmarco_hard_negatives_generation.log"

# Ensure output dirs
mkdir -p $(dirname "$OUTFILE")
mkdir -p $(dirname "$LOGFILE")

# Run script
python src/data/data_prep/bm25_negative_sampling_msmarco.py \
  --queries_file "$QUERIES_FILE" \
  --qrels_file "$QRELS_FILE" \
  --pyserini_file "$PYSERINI_FILE" \
  --docs_file "$DOCS_FILE" \
  --doc_offsets_file "$DOC_OFFSETS_FILE" \
  --outfile "$OUTFILE" \
  --logfile "$LOGFILE" \
  --n_negatives 32
