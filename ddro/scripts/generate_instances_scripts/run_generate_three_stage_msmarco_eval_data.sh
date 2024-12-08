#!/bin/sh
#SBATCH --job-name=t5_eval_data_gen
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 32 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/generate_three_stage_msmarco_pq_eval_data-%j.out # %j is the job ID

# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

# Change to the directory where your script and data are located.
HOME_DIR="/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro"


ENCODING="atomic"  # 'atomic' or 'pq' or 'atomic'
QRELS_PATH="$HOME_DIR/resources/datasets/raw/msmarco-data/msmarco-docdev-qrels.tsv.gz"
QUERY_PATH="$HOME_DIR/resources/datasets/raw/msmarco-data/msmarco-docdev-queries.tsv.gz"
PRETRAIN_MODEL_PATH="$HOME_DIR/resources/transformer_models/t5-base"  # Fixed to 't5-base' for all experiments  

# Run the Python script with the specified arguments
python data/generate_instances/generate_three_stage_msmarco_eval_data.py \
                --encoding $ENCODING \
                --qrels_path $QRELS_PATH \
                --query_path $QUERY_PATH \
                --pretrain_model_path $PRETRAIN_MODEL_PATH \
                --scale "top_300k" \

# Print success message when the job completes
echo "T5 evaluation data generation completed successfully"
