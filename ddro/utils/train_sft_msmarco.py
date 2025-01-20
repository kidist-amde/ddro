import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="pq", type=str, help="docid method atomic/pq/url")
parser.add_argument("--scale", default="top_300k", type=str, help="docid method atomic/pq/url")

args = parser.parse_args()

config_file = json.load(open("scripts/config.json", "r"))
config_file["atomic"]["add_doc_num"] = config_file["doc_num"][args.scale]
config = config_file[args.encoding]
encoding, add_doc_num, max_docid_length, use_origin_head = config["encoding"], config["add_doc_num"], config["max_docid_length"], config["use_origin_head"]


code_dir = "/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro"
top_or_rand, scale = args.scale.split("_")

######################################################################
print("start SFT traning...")
model = "t5_128_10"  # the data for current training
operation = "training"  # training / pair_training
max_seq_length = 128
epoch = 10
dataset="msmarco" # msmarco / nq
load_ckpt = "True"  # True if load checkpoint, go to load_ckpt_path



os.system(f"cd {code_dir}/utils && python sft_T5.py \
    --epoch {epoch} \
    --per_gpu_batch_size  100 \
    --learning_rate 1e-3 \
    --save_path {code_dir}/outputs-summaries-msmarco/{model}_{top_or_rand}_{scale}_{dataset}_{encoding}_resumed_FromUltron/ \
    --log_path {code_dir}/logs-summaries/{model}.{top_or_rand}.{scale}.{dataset}.{encoding}.log \
    --doc_file_path {code_dir}/resources/datasets/processed/msmarco-data/msmarco-docs-sents.{top_or_rand}.{scale}.json \
    --pretrain_model_path {code_dir}/resources/transformer_models/t5-base \
    --docid_path {code_dir}/resources/datasets/processed/msmarco-data/encoded_docid/t5_512_{encoding}_{top_or_rand}.{scale}.txt \
    --train_file_path {code_dir}/resources/datasets/processed/msmarco-data/sft_training_datasets_{encoding} \
    --dataset_script_dir {code_dir}/data/data_scripts \
    --dataset_cache_dir ../negs_tutorial_cache \
    --add_doc_num {add_doc_num} \
    --max_seq_length {max_seq_length} \
    --max_docid_length {max_docid_length} \
    --use_origin_head {use_origin_head} \
    --load_ckpt {load_ckpt} \
    --load_ckpt_path {code_dir}/outputs-summaries-msmarco/t5_128_10_top_300k_summary_pretrain_search/model_final.pkl \
    --output_every_n_step 5000 \
    --save_every_n_epoch 2 \
    --operation {operation}")


