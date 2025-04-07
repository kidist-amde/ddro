#!/bin/sh
#SBATCH --job-name=compute_msmarco_embeddings
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
##SBATCH --gres=gpu:nvidia_l40:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=128gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/compute_msmarco_embeddings-%j.out # %j is the job ID
# Set up the environment.

source /home/kmekonn/.bashrc
conda activate ddro
cd ..
nvidia-smi 

# Set the paths for the input and output files
INPUT_PATH="resources/datasets/processed/msmarco-data/msmarco-docs-sents.top.300k.json"
OUTPUT_PATH="resources/datasets/processed/msmarco-data/doc_embedding/t5_512_doc_top_300k.txt"

# Run the Python script and pass the arguments
python3 data/data_preprocessing/compute_t5_msmarco_doc_embeddings.py \
                --input_path "$INPUT_PATH" \
                --output_path "$OUTPUT_PATH" \
                --batch_size  256

