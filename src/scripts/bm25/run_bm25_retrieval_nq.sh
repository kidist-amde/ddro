#!/bin/sh
#SBATCH --job-name=bm25_tuning_nq
#SBATCH --time=4-00:00:00
#SBATCH --mem=128gb
#SBATCH -c 16
#SBATCH --output=logs-slurm-BM25/nq_bm25_eval_dev_top1000k-%j.out

# Activate environment
source ~/.bashrc
conda activate pyserini
cd ddro  # Assumes you're in project root

# Define paths
input_dir="resources/datasets/processed/nq-data/nq-merged-json"
index_dir="resources/datasets/processed/nq-data/nq-indexes/lucene-index-nq-docs"

train_queries="resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_train.tsv.gz"
dev_queries="resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_dev.tsv.gz"

train_output="resources/datasets/processed/nq-data/pyserini_data/nq_train_bm25tuned.txt"
dev_output="resources/datasets/processed/nq-data/pyserini_data/nq_dev_bm25tuned.txt"

qrels_train="resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_train.tsv.gz"
qrels_dev="resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_dev.tsv.gz"

# Index creation (uncomment to re-index)
# python -m pyserini.index.lucene \
#   --collection JsonCollection \
#   --input "$input_dir" \
#   --index "$index_dir" \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 16 \
#   --storePositions --storeDocvectors --storeRaw

# Retrieval for train set (uncomment to run)
# python -m pyserini.search.lucene \
#   --index "$index_dir" \
#   --topics "$train_queries" \
#   --output "$train_output" \
#   --output-format msmarco \
#   --hits 1000 \
#   --bm25 --k1 0.82 --b 0.68 \
#   --threads 16 --batch-size 16

# Retrieval for dev set
python -m pyserini.search.lucene \
  --index "$index_dir" \
  --topics "$dev_queries" \
  --output "$dev_output" \
  --output-format msmarco \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 16 --batch-size 16

# Evaluation (uncomment to run)
# python -m pyserini.eval.msmarco_doc_eval \
#   --judgments "$qrels_train" \
#   --run "$train_output"

python -m pyserini.eval.msmarco_doc_eval \
  --judgments "$qrels_dev" \
  --run "$dev_output"
