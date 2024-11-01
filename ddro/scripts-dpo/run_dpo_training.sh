#!/bin/sh
#SBATCH --job-name=DDRO-PQ-3epo
#SBATCH --partition gpu_h100
#SBATCH --gres=gpu:1
##SBATCH --partition gpu
##SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64gb #120gb #180gb
#SBATCH -c 16
#SBATCH --output=logs-slurm-dpo/DDRO-10epochPQ_3epoch_lr5e-4_beta_06_logs-%j.out # %j is the job ID

# Set up the environment.
# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate dsi-env
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

FILE_PATH=resources/datasets/processed/nq-data



# Run the Python training script
python pretrain/train_dpo_model.py \
    --train_file $FILE_PATH/nq_hard_negatives_format/nq_train_triples_with_hard_negatives.txt \
    --dev_file $FILE_PATH/nq_hard_negatives_format/nq_dev_triples_with_hard_negatives.txt \
    --docid_path $FILE_PATH/encoded_docid/t5_512_pq_docids.txt \
    --output_dir outputs-nq-dpo/pq/dpo_ckp_10epoPQ_3epoch_lr5e-4_beta_06 \
    --pretrain_model_path resources/transformer_models/t5-base \
    --use_origin_head False \
    --checkpoint_path outputs-nq/t5_128_1_pq_pretrain_search_finetune_ULTRON_10epoch/model_final.pkl \
    --max_prompt_length 64 \
  