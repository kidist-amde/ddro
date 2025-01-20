#!/bin/sh
#SBATCH --job-name=t5_eval_data_gen
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 32 # number of CPUs
##SBATCH --output=logs-slurm/other-logs/generate_three_stage_msmarco_pq_eval_data-%j.out # %j is the job ID
#SBATCH --output=logs-slurm-summaries/generate_three_stage_NQ_summary_eval_data-%j.out # %j is the job ID


# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro_env
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

# Change to the directory where your script and data are located.
HOME_DIR="/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro"


OUTPUT_DIR="$HOME_DIR/resources/datasets/processed/nq-data/test_data"
# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Variables and parameters for the script
ENCODING="summary"  # 'url_title' or 'pq' or 'atomic' or 'summary'
MODEL="t5"  # Fixed to 't5' for all experiments
QRELS_PATH="$HOME_DIR/resources/datasets/processed/nq-data/nq_msmarco_format/nq_qrels_dev.tsv.gz"
QUERY_PATH="$HOME_DIR/resources/datasets/processed/nq-data/nq_msmarco_format/nq_queries_dev.tsv.gz"
PRETRAIN_MODEL_PATH="$HOME_DIR/resources/transformer_models/t5-base"  # Fixed to 't5-base' for all experiments             
MSMARCO_OR_NQ="nq"  # Choose between 'msmarco' or 'nq'
SAMPLE_FOR_ONE_DOC=1  # Fixed to 1 for all experiments
DOCID_PATH="$HOME_DIR/resources/datasets/processed/nq-data/encoded_docid/t5_512_${ENCODING}_docids.txt"
DATA_PATH="$HOME_DIR/resources/datasets/processed/nq-data/nq-merged-json/msmarco_sents_pyserni_format.json"
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
