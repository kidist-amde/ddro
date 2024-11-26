#!/bin/sh
#SBATCH --job-name=nq_dataset_processing
##SBATCH --partition=staging # Accessible partition
#SBATCH --partition=cbuild
#SBATCH --ntasks=1          # Number of tasks
#SBATCH --time=24:00:00      # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
##SBATCH -c 1 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/generate_nq_url_title_encoded_docIDs_ids-%j.out # %j is the job ID
# Set up the environment.


# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro


# Set the model parameters
ENCODING=url_title # pq / url_title / atomic
SEQ_LENGTH=512
MODEL_NAME=t5


# Set the input and output paths
INPUT_EMBEDDINGS=resources/datasets/processed/nq-data/doc_embedding/t5_512_nq_doc_embedding.txt # pass this for PQ-based document IDs
OUTPUT_PATH=resources/datasets/processed/nq-data/encoded_docid/${MODEL_NAME}_${SEQ_LENGTH}_${ENCODING}_docids.txt
NQ_DATA_PATH=resources/datasets/processed/nq-data/nq-merged/nq_docs.tsv.gz # pass this for atomic and URL-based document IDs
SUMMARY_PATH=resources/datasets/processed/nq-data/NQ_llama3_summaries.tsv.gz # pass this for Summary-based document IDs  
# Run the Python script
python data/generate_instances/generate_nq_t5_encoded_docids.py \
                --encoding $ENCODING \
                --nq_data_path $NQ_DATA_PATH \
                --output_path $OUTPUT_PATH \
                --batch_size 1024