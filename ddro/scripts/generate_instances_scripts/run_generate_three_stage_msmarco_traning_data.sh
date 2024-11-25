#!/bin/sh
#SBATCH --job-name=t5_train_data_gen
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 32 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/generate_general_pretrain_stage_msmarco_url_train_data-%j.out # %j is the job ID

# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro

# Change to the base directory where the code and data are located.
HOME_DIR="/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro"

# Variables and parameters for the script
ENCODING="url"  # 'url' or 'pq' or 'atomic'
QRELS_PATH="$HOME_DIR/resources/datasets/raw/msmarco-data/msmarco-doctrain-qrels.tsv.gz"
QUERY_PATH="$HOME_DIR/resources/datasets/raw/msmarco-data/msmarco-doctrain-queries.tsv.gz"
PRETRAIN_MODEL_PATH="$HOME_DIR/resources/transformer_models/t5-base"  # Fixed to 't5-base' for all experiments             
FAKE_QUERY_PATH="$HOME_DIR/resources/datasets/processed/msmarco-data/msmarco_pseudo_query_10.txt"
CUR_DATA="general_pretrain"  # general_pretrain/search_pretrain/finetune

# Run the Python script with the specified arguments
python data/generate_instances/generate_three_stage_msmarco_train_data.py \
                --encoding $ENCODING \
                --qrels_path $QRELS_PATH \
                --query_path $QUERY_PATH \
                --pretrain_model_path $PRETRAIN_MODEL_PATH \
                --fake_query_path $FAKE_QUERY_PATH \
                --scale "top_300k" \
                --cur_data $CUR_DATA


# Print success message when the job completes
echo "T5 training data generation completed successfully"
 