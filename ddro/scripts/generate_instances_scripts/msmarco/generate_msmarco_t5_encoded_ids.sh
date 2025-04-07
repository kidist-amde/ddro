#!/bin/sh
#SBATCH --job-name=generate_msmarco_semantic_ids
#SBATCH --partition=gpu 
##SBATCH --gres=gpu:nvidia_rtx_a6000:2
##SBATCH --gres=gpu:nvidia_l40:2
#SBATCH --gres=gpu:tesla_p40:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
##SBATCH --output=logs-slurm/other-logs/generate_msmarco_t5_summary_encoded_ids-%j.out # %j is the job ID
#SBATCH --output=logs-slurm-summaries/generate_msmarco_t5_summary_encoded_ids-%j.out # %j is the job ID

# Set up the environment.

# Set up the environment.
source /home/kmekonn/.bashrc

conda activate ddro_env
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

# Set the model parameters
ENCODING="summary"  # Options: atomic/pq/url/summary
SEQ_LENGTH=512
MODEL_NAME="t5"
SCALE="300k" # Options: top/rand_100k/200k/300k
top_or_rand="top" # Options: top/rand

# Set the input and output paths
INPUT_EMBEDDINGS="resources/datasets/processed/msmarco-data/doc_embedding/t5_512_doc_top_300k.txt" # For PQ encoding
OUTPUT_PATH="resources/datasets/processed/msmarco-data/encoded_docid/${MODEL_NAME}_${SEQ_LENGTH}_${ENCODING}_${top_or_rand}.${SCALE}.txt"
DATA_PATH="resources/datasets/processed/msmarco-data/msmarco-docs-sents.${top_or_rand}.${SCALE}.json" # For atomic and URL encoding
LEADING_SENTENCES_PATH="resources/datasets/processed/msmarco-data/msmarco-docs-leading-sents.json" # For summary encoding

# Run the Python script with appropriate arguments
python data/generate_instances/generate_t5_encoded_mamarco_docids.py \
    --encoding $ENCODING \
    --input_doc_path $DATA_PATH \
    --input_embed_path $INPUT_EMBEDDINGS \
    --output_path $OUTPUT_PATH \
    --summary_path $LEADING_SENTENCES_PATH \
    --batch_size 1024 \
    --pretrain_model_path resources/transformer_models/t5-base
echo "Done encoding the docids"