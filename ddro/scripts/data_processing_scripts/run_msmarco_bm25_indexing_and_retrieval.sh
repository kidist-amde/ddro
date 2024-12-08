#!/bin/sh
#SBATCH --job-name=bm25tuning
##SBATCH --partition=gpu 
##SBATCH --gres=gpu:nvidia_rtx_a6000:1
##SBATCH --gres=gpu:nvidia_l40:1
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=256gb # memory per GPU 
#SBATCH -c 32 # number of CPUs
##SBATCH --output=logs-slurm-msmarco-sft/other-logs/msmarco_dataset_BM25_Index_Creation-%j.out # %j is the job ID
##SBATCH --output=logs-slurm-msmarco-sft/other-logs/msmarco_dataset_BM25_retrieval_bm25_train_top_1000k-%j.out # %j is the job ID
##SBATCH --output=logs-slurm-msmarco-sft/other-logs/msmarco_dataset_BM25_retrieval_bm25_dev_top_1000k-%j.out # %j is the job ID
#SBATCH --output=logs-slurm-msmarco-sft/other-logs/msmarco_dataset_BM25_retrieval_EVAL_Train_top_1000k-%j.out # %j is the job ID

# Set up the environment.
source /home/kmekonn/.bashrc
conda activate pyserini
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

INPUT=resources/datasets/processed/msmarco-data/msmarco-json-sents
INDEX=resources/datasets/processed/msmarco-data/msmarco-indexs/lucene-index-msmarco-docs
MS_TRAIN_SET=resources/datasets/raw/msmarco-data/msmarco-doctrain-queries.tsv.gz
MS_DEV_SET=resources/datasets/raw/msmarco-data/msmarco-docdev-queries.tsv.gz
OUTPUT_DEV=resources/datasets/processed/msmarco-data/pyserini_data/msmarco_dev_bm25tuned.txt
OUTPUT_TRAIN=resources/datasets/processed/msmarco-data/pyserini_data/msmarco_train_bm25tuned.txt
QRELS_DEV=resources/datasets/raw/msmarco-data/msmarco-docdev-qrels.tsv.gz
QRELS_TRAIN=resources/datasets/raw/msmarco-data/msmarco-doctrain-qrels.tsv.gz



# python -m pyserini.index.lucene \
#   --collection JsonCollection \
#   --input $INPUT \
#   --index $INDEX \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 16 \
#   --storePositions --storeDocvectors --storeRaw

# python -m pyserini.search.lucene \
#   --index  $INDEX \
#   --topics  $MS_TRAIN_SET \
#   --output $OUTPUT_TRAIN \
#   --output-format msmarco \
#   --hits 1000 \
#   --bm25 --k1 0.82 --b 0.68 \
#   --threads 16 --batch-size 16

# python -m pyserini.search.lucene \
#   --index  $INDEX \
#   --topics  $MS_DEV_SET \
#   --output $OUTPUT_DEV \
#   --output-format msmarco \
#   --hits 1000 \
#   --bm25 --k1 0.82 --b 0.68 \
#   --threads 16 --batch-size 16

# Evaluation for train set
python -m pyserini.eval.msmarco_doc_eval \
  --judgments $QRELS_TRAIN \
  --run $OUTPUT_TRAIN \

# # Evaluation for dev set
# python -m pyserini.eval.msmarco_doc_eval \
#   --judgments $QRELS_DEV \
#   --run $OUTPUT_DEV \
