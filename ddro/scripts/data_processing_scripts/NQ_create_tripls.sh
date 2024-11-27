#!/bin/sh
#SBATCH --job-name=NQ_Triples
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm-sft/other-logs/nq_triples-%j.out # %j is the job ID

# Set up the environment
source /home/kmekonn/.bashrc
conda activate pyserini

# Navigate to the project directory
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

# Run the script
python data/data_preprocessing/create_nq_triples.py \
    --relevance_path resources/datasets/processed/nq-data/nq_msmarco_format/nq_dev_bm25tuned.txt \
    --qrel_path resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_train.tsv.gz \
    --output_path resources/datasets/processed/nq-data/nq_hard_negatives_format \
    --num_negative_per_query 1 \
    --query_path resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_dev.tsv.gz \
    --docs_path resources/datasets/processed/nq-data/nq-merged-json/nq-docs-sents.json
