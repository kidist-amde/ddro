#!/bin/sh
#SBATCH --job-name=compute_nq_embeddings
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:8
##SBATCH --gres=gpu:nvidia_l40:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=256gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/other-logs/compute_nq_embeddings-%j.out # %j is the job ID
# Set up the environment.

source /home/kmekonn/.bashrc
conda activate ddro
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

python data/data_preprocessing/compute_nq_t5_embeddings.py \
    --merged_file resources/datasets/raw/nq-data/nq_merged.tsv.gz \
    --output_file resources/datasets/processed/nq-data/doc_embedding/t5_512_nq_doc_embedding.txt \
    --model_name sentence-transformers/gtr-t5-base \
    --batch_size 256
