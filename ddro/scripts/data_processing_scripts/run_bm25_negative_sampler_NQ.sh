#!/bin/sh
#SBATCH --job-name=negativesampling
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=256gb # memory per GPU
#SBATCH -c 32 # number of CPUs
#SBATCH --output=logs-slurm-sft-nq/other-logs/NQ-NEGATIVE-SAMPLER-TRAIN-SET-%j.out # %j is the job ID

# Set up the environment
source /home/kmekonn/.bashrc
conda activate ddro_env
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

SUBSET="train" # Options: "train" or "dev"

# Run the Python script
python data/data_preprocessing/bm25_negative_Sampling_NQ.py \
    --relevance_path resources/datasets/processed/nq-data/pyserini_data/nq_${SUBSET}_bm25tuned.txt \
    --qrel_path resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_${SUBSET}.tsv.gz \
    --output_path resources/datasets/processed/nq-data/hard_negatives_from_bm25_top1000_retrieval/nq_${SUBSET}_triples.tsv \
    --num_negative_per_query 8 \
    --query_path resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_${SUBSET}.tsv.gz \
    --docs_path resources/datasets/processed/nq-data/nq-merged/nq_docs.tsv
 