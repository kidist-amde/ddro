#!/bin/sh
#SBATCH --job-name=bm25_tuning_msmarco
#SBATCH --time=4-00:00:00
#SBATCH --mem=128gb
#SBATCH -c 32
#SBATCH --output=logs-slurm-BM25/msmarco_eval_train_top1000k-%j.out

# Set up environment
source ~/.bashrc
conda activate pyserini


# Path configurations
input_dir="resources/datasets/processed/msmarco-data/msmarco-json-sents"
index_dir="resources/datasets/processed/msmarco-data/msmarco-indexs/lucene-index-msmarco-docs"

train_queries="resources/datasets/raw/msmarco-data/msmarco-doctrain-queries.tsv.gz"
dev_queries="resources/datasets/raw/msmarco-data/msmarco-docdev-queries.tsv.gz"

train_output="resources/datasets/processed/msmarco-data/pyserini_data/msmarco_train_bm25tuned.txt"
dev_output="resources/datasets/processed/msmarco-data/pyserini_data/msmarco_dev_bm25tuned.txt"

qrels_train="resources/datasets/raw/msmarco-data/msmarco-doctrain-qrels.tsv.gz"
qrels_dev="resources/datasets/raw/msmarco-data/msmarco-docdev-qrels.tsv.gz"

# Indexing (uncomment if needed)
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$input_dir" \
  --index "$index_dir" \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw

# BM25 retrieval - Train set (uncomment to run)
python -m pyserini.search.lucene \
  --index "$index_dir" \
  --topics "$train_queries" \
  --output "$train_output" \
  --output-format msmarco \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 16 --batch-size 16

# BM25 retrieval - Dev set
python -m pyserini.search.lucene \
  --index "$index_dir" \
  --topics "$dev_queries" \
  --output "$dev_output" \
  --output-format msmarco \
  --hits 1000 \
  --bm25 --k1 2.16 --b 0.61 \
  --threads 16 --batch-size 16

# Evaluation - Dev set
python -m pyserini.eval.msmarco_doc_eval \
  --judgments "$qrels_dev" \
  --run "$dev_output"
