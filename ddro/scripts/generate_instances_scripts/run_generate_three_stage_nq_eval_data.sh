#!/bin/sh
#SBATCH --job-name=nq_dataset_processing
##SBATCH --partition=staging # Accessible partition
#SBATCH --partition=cbuild
#SBATCH --ntasks=1          # Number of tasks
#SBATCH --time=1:00:00      # d-h:m:s
#SBATCH --mem=64gb         # Memory per job
#SBATCH -c 1             # Number of CPUs
#SBATCH --output=logs-slurm/other-logs/T5_url_title_eval_data_generation-%j.out # %j is the job ID


# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro

# Change to the directory where your script and data are located.
HOME_DIR="/gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro"

OUTPUT_DIR="$HOME_DIR/resources/datasets/processed/nq-data/test_data"
# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Variables and parameters for the script
ENCODING="url_title"  # 'url_title' or 'pq' or 'atomic'
MODEL="t5"  # Fixed to 't5' for all experiments
QRELS_PATH="$HOME_DIR/resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_dev.tsv.gz"
QUERY_PATH="$HOME_DIR/resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_dev.tsv.gz"
PRETRAIN_MODEL_PATH="$HOME_DIR/resources/transformer_models/t5-base"  # Fixed to 't5-base' for all experiments             
MSMARCO_OR_NQ="nq"  # Choose between 'msmarco' or 'nq'
SAMPLE_FOR_ONE_DOC=1  # Fixed to 1 for all experiments
DOCID_PATH="$HOME_DIR/resources/datasets/processed/nq-data/encoded_docid/t5_512_${ENCODING}_docids.txt"
DATA_PATH="$HOME_DIR/resources/datasets/processed/nq-data/nq-merged-json/nq-docs-sents.json"
MAX_SEQ_LENGTH=128  # Fixed to 128 for all experiments
OUTPUT_PATH="$OUTPUT_DIR/query_dev.${MODEL}_${MAX_SEQ_LENGTH}_${SAMPLE_FOR_ONE_DOC}.${ENCODING}_${MSMARCO_OR_NQ}.json"
CURRENT_DATA="query_dev"  # Fixed to 'query_dev' for all experiments

# Run the Python script with the specified arguments
python data/generate_instances/generate_three_stage_nq_eval_data.py \
    --encoding $ENCODING \
    --qrels_path $QRELS_PATH \
    --query_path $QUERY_PATH \
    --pretrain_model_path $PRETRAIN_MODEL_PATH \
    --msmarco_or_nq $MSMARCO_OR_NQ \
    --max_seq_length $MAX_SEQ_LENGTH \
    --data_path $DATA_PATH \
    --docid_path $DOCID_PATH \
    --output_path $OUTPUT_PATH \
    --current_data $CURRENT_DATA

# Print success message when the job completes
echo "T5 evaluation data generation completed successfully"
