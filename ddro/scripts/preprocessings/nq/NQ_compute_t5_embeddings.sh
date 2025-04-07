#!/bin/sh
#SBATCH --job-name=NQ_Precomputed_embeddings
#SBATCH --partition gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
##SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=128gb #120gb #180gb
#SBATCH -c 4
#SBATCH --output=logs-slurm/other-logs/NQ_Precomputed_embeddings-%j.out # %j is the job ID
# Set up the environment.

# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro

python data/data_preprocessing/compute_nq_t5_embeddings.py \
    --merged_file resources/datasets/processed/nq-data/nq-merged/nq_docs.tsv.gz \
    --output_file resources/datasets/processed/nq-data/doc_embedding/t5_512_nq_doc_embedding.txt \
    --model_name sentence-transformers/gtr-t5-base \
    --batch_size 256
