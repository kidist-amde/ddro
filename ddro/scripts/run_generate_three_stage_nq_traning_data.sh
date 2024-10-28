#!/bin/sh
#SBATCH --job-name=t5_train_data_gen
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 32 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/t5_atomic_search_pretrain_gen-%j.out # %j is the job ID

# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro

# Change to the base directory where the code and data are located.
HOME_DIR="/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro"

# Variables and parameters for the script
ENCODING="atomic"  # 'url' or 'pq' or 'atomic'
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
CURRENT_DATA="search_pretrain"  # Set to 'general_pretrain', 'search_pretrain', or 'finetune'

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
 