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
#SBATCH --output=logs-slurm-summaries/generate_mNQ_t5_summary_encoded_ids-%j.out # %j is the job ID


# Set up the environment.
source /home/kmekonn/.bashrc

conda activate ddro_env
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

# Set the model parameters
ENCODING=summary # pq / url_title / atomic / summary
SEQ_LENGTH=512
MODEL_NAME=t5


# Set the input and output paths
INPUT_EMBEDDINGS=resources/datasets/processed/nq-data/doc_embedding/t5_512_nq_doc_embedding.txt # pass this for PQ-based document IDs
OUTPUT_PATH=resources/datasets/processed/nq-data/encoded_docid/${MODEL_NAME}_${SEQ_LENGTH}_${ENCODING}_docids.txt
NQ_DATA_PATH=resources/datasets/processed/nq-data/nq-merged/nq_docs.tsv.gz # pass this for atomic and URL-based document IDs
LEADING_SENTENCES_PATH=resources/datasets/processed/nq-data/nq-merged/nq_docs_leading_sents.json # pass this for Summary-based document IDs  
# Run the Python script
python data/generate_instances/generate_nq_t5_encoded_docids.py \
                --encoding $ENCODING \
                --nq_data_path $NQ_DATA_PATH \
                --output_path $OUTPUT_PATH \
                --summary_path $LEADING_SENTENCES_PATH \
                --batch_size 1024 \
                --pretrain_model_path resources/transformer_models/t5-base
echo "Done encoding the docids"