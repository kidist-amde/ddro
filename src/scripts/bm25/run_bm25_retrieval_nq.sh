#!/bin/sh
#SBATCH --job-name=bm25tuning
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
# SBATCH --output=logs-slurm-BM25/NQ_BM25-%j.out # %j is the job ID

# Set up the environment.
source /home/kmekonn/.bashrc
conda activate pyserini

INPUT=resources/datasets/processed/nq-data/nq-merged-json
INDEX=resources/datasets/processed/nq-data/nq-indexes/lucene-index-nq-docs
NQ_TRAIN_SET=resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_train.tsv.gz
NQ_DEV_SET=resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_dev.tsv.gz
OUTPUT_DEV=resources/datasets/processed/nq-data/pyserini_data/nq_dev_bm25tuned.txt
OUTPUT_TRAIN=resources/datasets/processed/nq-data/pyserini_data/nq_train_bm25tuned.txt
QRELS_DEV=resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_dev.tsv.gz
QRELS_TRAIN=resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_train.tsv.gz
s
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $INPUT \
  --index $INDEX \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw

python -m pyserini.search.lucene \
  --index  $INDEX \
  --topics  $NQ_TRAIN_SET \
  --output $OUTPUT_TRAIN \
  --output-format msmarco \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 16 --batch-size 16

python -m pyserini.search.lucene \
  --index  $INDEX \
  --topics  $NQ_DEV_SET \
  --output $OUTPUT_DEV \
  --output-format msmarco \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 16 --batch-size 16


# Evaluation for train set
python -m pyserini.eval.msmarco_doc_eval \
  --judgments $QRELS_TRAIN \
  --run $OUTPUT_TRAIN \

# Evaluation for dev set
python -m pyserini.eval.msmarco_doc_eval \
  --judgments $QRELS_DEV \
  --run $OUTPUT_DEV \
