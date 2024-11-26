#!/bin/sh
#SBATCH --job-name=nq_dataset_processing
##SBATCH --partition=staging # Accessible partition
#SBATCH --partition=cbuild
#SBATCH --ntasks=1          # Number of tasks
#SBATCH --time=24:00:00      # d-h:m:s
#SBATCH --mem=64gb         # Memory per job
#SBATCH -c 1             # Number of CPUs
#SBATCH --output=logs-slurm/other-logs/T5_url_title_general_pretrain_data_generation-%j.out # %j is the job ID


# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro

# Change to the directory where your script and data are located.
HOME_DIR="/gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro"

# Variables and parameters for the script
ENCODING="url_title"  # 'url_title' or 'pq' or 'atomic'
MODEL="t5"  # Fixed to 't5' for all experiments
QRELS_PATH="$HOME_DIR/resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_train.tsv.gz"
QUERY_PATH="$HOME_DIR/resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_train.tsv.gz"
PRETRAIN_MODEL_PATH="$HOME_DIR/resources/transformer_models/t5-base"  # Fixed to 't5-base' for all experiments             
MSMARCO_OR_NQ="nq"  # Choose 'nq'
SAMPLE_FOR_ONE_DOC=1  # Fixed to 1 for all experiments
DOCID_PATH="$HOME_DIR/resources/datasets/processed/nq-data/encoded_docid/t5_512_${ENCODING}_docids.txt"
DATA_PATH="$HOME_DIR/resources/datasets/processed/nq-data/nq-merged-json/nq-docs-sents.json"
MAX_SEQ_LENGTH=128  # Fixed to 128 for all experiments
FAKE_QUERY_PATH="$HOME_DIR/resources/datasets/processed/nq-data/nq_pseudo_query_10_3epoch.txt"
CURRENT_DATA="general_pretrain"  # Set to 'general_pretrain', 'search_pretrain', or 'finetune'

# Run the Python script with the specified arguments
python data/generate_instances/generate_three_stage_nq_training_data.py \
    --encoding $ENCODING \
    --cur_data $CURRENT_DATA \
    --query_path $QUERY_PATH \
    --qrels_path $QRELS_PATH \
    --pretrain_model_path $PRETRAIN_MODEL_PATH \
    --msmarco_or_nq $MSMARCO_OR_NQ \
    --max_seq_length $MAX_SEQ_LENGTH \
    --data_path $DATA_PATH \
    --docid_path $DOCID_PATH \
    --fake_query_path $FAKE_QUERY_PATH \
    --sample_for_one_doc $SAMPLE_FOR_ONE_DOC \
    --model $MODEL

# Print success message when the job completes
echo "T5 training data generation completed successfully"
 