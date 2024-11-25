import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="url", type=str, help="docid method atomic/pq/url")
parser.add_argument("--scale", default="top_300k", type=str, help="docid method atomic/pq/url")
args = parser.parse_args()

config_file = json.load(open("scripts/config.json", "r"))
config_file["atomic"]["add_doc_num"] = config_file["doc_num"][args.scale]
config = config_file[args.encoding]
encoding, add_doc_num, max_docid_length, use_origin_head = config["encoding"], config["add_doc_num"], config["max_docid_length"], config["use_origin_head"]


code_dir = "/gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro"
top_or_rand, scale = args.scale.split("_")

######################################################################
print("start general pretrain...")
model = "t5_128_10"  # the data for current training
all_data = "pretrain"  # all data used for training so far
cur_data = "pretrain"  # the data used for current training  # pretrain/rank_pretrain/finetune
stage = "pretrain"  # pretrain / post_pretrain / finetune
operation = "training"  # training / pair_training
max_seq_length = 128
epoch = 10

# os.system(f"cd {code_dir}/utils && python runT5.py \
#     --epoch {epoch} \
#     --per_gpu_batch_size  100 \
#     --learning_rate 1e-3 \
#     --save_path {code_dir}/outputs-msmarco/{model}_{top_or_rand}_{scale}_{encoding}_{all_data}_ULTRON/ \
#     --log_path {code_dir}/logs-msmarco/{stage}.{model}.{top_or_rand}.{scale}.{encoding}.{all_data}.log \
#     --doc_file_path {code_dir}/resources/datasets/processed/msmarco-data/msmarco-docs-sents.{top_or_rand}.{scale}.json \
#     --pretrain_model_path {code_dir}/resources/transformer_models/t5-base \
#     --docid_path {code_dir}/resources/datasets/processed/msmarco-data/encoded_docid/t5_512_{encoding}_{top_or_rand}.{scale}.txt \
#     --train_file_path {code_dir}/resources/datasets/processed/msmarco-data/train_data_{top_or_rand}_{scale}/{cur_data}.{model}.{encoding}.{scale}.json \
#     --test_file_path {code_dir}/resources/datasets/processed/msmarco-data/test_data_{top_or_rand}_{scale}/ \
#     --dataset_script_dir {code_dir}/data/data_scripts \
#     --dataset_cache_dir {code_dir}/negs_tutorial_cache \
#     --add_doc_num {add_doc_num} \
#     --max_seq_length {max_seq_length} \
#     --max_docid_length {max_docid_length} \
#     --use_origin_head {use_origin_head} \
#     --output_every_n_step 5000 \
#     --save_every_n_epoch 2 \
#     --operation {operation}")


# ######################################################################
# print("start search pretrain...")
# model = "t5_128_10"  # the data for current training
# load_model = "t5_128_10"  # the data to be loaded
# all_data = "pretrain_search"  # all data used for training  # pretrain_post_finetune
# cur_data = "search_pretrain"  # the data used for current training  # pretrain / rank_pretrain / finetune
# stage = "search_pretrain"  # pretrain / post_pretrain / finetune
# load_ckpt = "True"  # True if load checkpoint, go to load_ckpt_path
# operation = "training"  # training
# max_seq_length = 64
# epoch = 20

# os.system(f"cd {code_dir}/utils && python runT5.py \
#     --epoch {epoch} \
#     --per_gpu_batch_size  100 \
#     --learning_rate 1e-3 \
#     --save_path {code_dir}/outputs-msmarco/{model}_{top_or_rand}_{scale}_{encoding}_{all_data}_ULTRON/ \
#     --log_path {code_dir}/logs-msmarco/{stage}.{model}.{top_or_rand}.{scale}.{encoding}.{all_data}.log \
#     --doc_file_path {code_dir}/resources/datasets/processed/msmarco-data/msmarco-docs-sents.{top_or_rand}.{scale}.json \
#     --pretrain_model_path {code_dir}/resources/transformer_models/t5-base \
#     --docid_path {code_dir}/resources/datasets/processed/msmarco-data/encoded_docid/t5_512_{encoding}_{top_or_rand}.{scale}.txt \
#     --train_file_path {code_dir}/resources/datasets/processed/msmarco-data/train_data_{top_or_rand}_{scale}/{cur_data}.{model}.{encoding}.{scale}.json \
#     --test_file_path {code_dir}/resources/datasets/processed/msmarco-data/test_data_{top_or_rand}_{scale}/ \
#     --dataset_script_dir {code_dir}/data/data_scripts \
#     --dataset_cache_dir {code_dir}/negs_tutorial_cache \
#     --add_doc_num {add_doc_num} \
#     --max_seq_length {max_seq_length} \
#     --max_docid_length {max_docid_length} \
#     --use_origin_head {use_origin_head} \
#     --load_ckpt {load_ckpt} \
#     --load_ckpt_path {code_dir}/outputs-msmarco/{load_model}_{top_or_rand}_{scale}_{encoding}_pretrain_ULTRON/model_final.pkl \
#     --output_every_n_step 5000 \
#     --save_every_n_epoch 4 \
#     --operation {operation}")


######################################################################
print("start finetune...")
model = "t5_128_1"  # the data for current training
load_model = "t5_128_10"  # the data to be loaded
all_data = "pretrain_search_finetune"  # all data used for training  # pretrain_post_finetune
cur_data = "finetune"  # the data used for current training  # pretrain / rank_pretrain / finetune
stage = "finetune"  # pretrain / post_pretrain / finetune
load_ckpt = "True"  # True if load checkpoint, go to load_ckpt_path
operation = "training"  # training / pair_training
max_seq_length = 64
epoch = 10

os.system(f"cd {code_dir}/utils && python runT5.py \
    --epoch {epoch} \
    --per_gpu_batch_size 100 \
    --learning_rate 1e-3 \
    --save_path {code_dir}/outputs-msmarco/{model}_{top_or_rand}_{scale}_{encoding}_{all_data}_ULTRON/ \
    --log_path {code_dir}/logs-msmarco/{stage}.{model}.{top_or_rand}.{scale}.{encoding}.{all_data}.log \
    --doc_file_path {code_dir}/resources/datasets/processed/msmarco-data/msmarco-docs-sents.{top_or_rand}.{scale}.json \
    --pretrain_model_path {code_dir}/resources/transformer_models/t5-base \
    --docid_path {code_dir}/resources/datasets/processed/msmarco-data/encoded_docid/t5_512_{encoding}_{top_or_rand}.{scale}.txt \
    --train_file_path {code_dir}/resources/datasets/processed/msmarco-data/train_data_{top_or_rand}_{scale}/{cur_data}.{model}.{encoding}.{scale}.json \
    --test_file_path {code_dir}/resources/datasets/processed/msmarco-data/test_data_{top_or_rand}_{scale}/ \
    --dataset_script_dir {code_dir}/data/data_scripts \
    --dataset_cache_dir {code_dir}/negs_tutorial_cache \
    --add_doc_num {add_doc_num} \
    --max_seq_length {max_seq_length} \
    --max_docid_length {max_docid_length} \
    --use_origin_head {use_origin_head} \
    --load_ckpt {load_ckpt} \
    --load_ckpt_path {code_dir}/outputs-msmarco/{load_model}_{top_or_rand}_{scale}_{encoding}_pretrain_search_ULTRON/model_final.pkl \
    --output_every_n_step 5000 \
    --save_every_n_epoch 2 \
    --operation {operation}")

print("write success")