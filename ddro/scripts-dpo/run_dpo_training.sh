#!/bin/sh

#SBATCH --job-name=DDRO-PQ
#SBATCH --partition=gpu 
##SBATCH --gres=gpu:nvidia_rtx_a6000:1
##SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=64gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm-dpo/DDRO-PQ_7epoch_lr5e-6_beta_049_logs-%j.out # %j is the job ID
# Set up the environment.
source /home/kmekonn/.bashrc
conda activate ddro
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 
FILE_PATH=resources/datasets/processed/nq-data

# Run the Python training script
python pretrain/train_dpo_model.py \
    --train_file $FILE_PATH/nq_hard_negatives_format/nq_train_triples_with_hard_negatives.txt \
    --dev_file $FILE_PATH/nq_hard_negatives_format/nq_dev_triples_with_hard_negatives.txt \
    --docid_path $FILE_PATH/encoded_docid/t5_512_pq_docids.txt \
    --output_dir outputs-nq-dpo/pq/dpo_ckp_7epoch_lr5e-6_beta_049 \
    --pretrain_model_path resources/transformer_models/t5-base \
    --use_origin_head False \
    --checkpoint_path outputs-nq/t5_128_1_pq_pretrain_search_finetune/model_final.pkl \
    --max_prompt_length 64 \
  