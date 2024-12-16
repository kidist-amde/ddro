#!/bin/sh
#SBATCH --job-name=bm25eval
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=256gb # memory per GPU 
#SBATCH -c 32 # number of CPUs
#SBATCH --output=logs-slurm-sft-nq/other-logs/NQ_dataset_BM25_PYTREC_DEV_Eval-%j.out # %j is the job ID

# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro_env
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro



RUN_FILE_DEV=resources/datasets/processed/nq-data/pyserini_data/nq_dev_bm25tuned.txt
RUN_FILE_TRAIN=resources/datasets/processed/nq-data/pyserini_data/nq_train_bm25tuned.txt

QRELS_DEV=resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_dev.tsv.gz
QRELS_TRAIN=resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_train.tsv.gz

# python data/data_preprocessing/pytrec_eval_eval.py
python data/data_preprocessing/eval_pytrec_eval.py \
  --qrels_file $QRELS_TRAIN \
  --run_file $RUN_FILE_TRAIN