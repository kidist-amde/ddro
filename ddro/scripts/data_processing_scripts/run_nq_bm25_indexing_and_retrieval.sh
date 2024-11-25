#!/bin/sh
#SBATCH --job-name=bm25tuning
##SBATCH --partition=gpu 
##SBATCH --gres=gpu:nvidia_rtx_a6000:1
##SBATCH --gres=gpu:nvidia_l40:1
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/nq_retrieval_bm25-%j.out # %j is the job ID
# Set up the environment.
source /home/kmekonn/.bashrc
conda activate pyserini
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

INPUT=resources/datasets/processed/nq-data/nq-merged-json
INDEX=resources/datasets/processed/nq-data/nq-indexes/lucene-index-nq-docs
NQ_TRAIN_SET=resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_train.tsv.gz
NQ_DEV_SET=resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_dev.tsv.gz
OUTPUT_DEV=resources/datasets/processed/nq-data/nq_msmarco_format/nq_dev_bm25tuned.txt
OUTPUT_TRAIN=resources/datasets/processed/nq-data/nq_msmarco_format/nq_train_bm25tuned.txt


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
  --hits 10 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 16 --batch-size 16

python -m pyserini.search.lucene \
  --index  $INDEX \
  --topics  $NQ_DEV_SET \
  --output $OUTPUT_DEV \
  --output-format msmarco \
  --hits 10 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 16 --batch-size 16

