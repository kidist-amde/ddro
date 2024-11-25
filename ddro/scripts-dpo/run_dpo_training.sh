#!/bin/sh
#SBATCH --job-name=DDRO-atomic-3epo
##SBATCH --partition gpu_h100
##SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=16gb #120gb #180gb
#SBATCH -c 2
#SBATCH --output=logs-slurm-sft/DDRO-url_2epoch_lr5e-7_beta_049_128_logs-%j.out # %j is the job ID

# Set up the environment.
# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate ddro
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

# --max_prompt_length 64 \

FILE_PATH=resources/datasets/processed/nq-data

# Run the Python training script
python pretrain/train_dpo_model.py \
    --train_file $FILE_PATH/nq_hard_negatives_format/nq_train_triples_with_hard_negatives.txt \
    --dev_file $FILE_PATH/nq_hard_negatives_format/nq_dev_triples_with_hard_negatives.txt \
    --docid_path $FILE_PATH/encoded_docid/t5_512_url_docids.txt \
    --output_dir outputs-sft/dpo/url/dpo_ckp_url_2epoch_lr5e-7 \
    --pretrain_model_path resources/transformer_models/t5-base \
    --use_origin_head False \
    --checkpoint_path outputs-sft/t5_128_10_top_300k_nq_url_title/model_final.pkl \
    --max_prompt_length 128 \
  