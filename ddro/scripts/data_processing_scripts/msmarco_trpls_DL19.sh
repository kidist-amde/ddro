#!/bin/sh
#SBATCH --job-name=MS_Triples
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=256gb # memory per GPU 
#SBATCH -c 32 # number of CPUs
#SBATCH --output=logs-slurm-msmarco-sft/other-logs/MS_dev_triples_DL19-%j.out # %j is the job ID

# Set up the environment
source /home/kmekonn/.bashrc
conda activate pyserini

# Navigate to the project directory
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro

# Run the script
python data/data_preprocessing/create_msmarco_triples_DL19.py 
   