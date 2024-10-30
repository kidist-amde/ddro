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

## test settings
print("start evaluation...")
model = "t5_128_1"  # the data for current training
load_model = "t5_128_1"  # the data to be loaded
all_data = "pretrain_search_finetune"  # all data used for training
cur_data = "query_dev"  # the data used for current training
stage = "inference"  # pretrain / finetune
num_beams = 100
use_docid_rank = "True"  # True to discriminate different docs with the same docid
operation = "testing"
max_seq_length = 64


def main():
    # for epoch in [1,3,5,7,9]:
    epoch = 9
    os.system(f"""cd {code_dir}/utils && python runT5.py \
        --epoch 10 \
        --per_gpu_batch_size 2 \
        --learning_rate 1e-3 \
        --save_path {code_dir}/outputs-nq/t5_128_10_url_title_pretrain_search/model_final.pkl \
        --log_path {code_dir}/logs-nq/inference.t5_128_1.url_title.pretrain_search.log \
        --doc_file_path {code_dir}/resources/datasets/processed/nq-data/nq-merged-json/nq-docs-sents.json \
        --pretrain_model_path {code_dir}/resources/transformer_models/t5-base \
        --docid_path {code_dir}/resources/datasets/processed/nq-data/encoded_docid/t5_512_url_docids.txt \
        --train_file_path {code_dir}/resources/datasets/processed/nq-data/train_data/{cur_data}.{model}.url.json \
        --test_file_path {code_dir}/resources/datasets/processed/nq-data/test_data/query_dev.t5_128_1.url_nq.json \
        --dataset_script_dir {code_dir}/data/data_scripts \
        --dataset_cache_dir {code_dir}/negs_tutorial_cache \
        --num_beams {num_beams} \
        --add_doc_num {add_doc_num} \
        --max_seq_length {max_seq_length} \
        --max_docid_length {max_docid_length} \
        --output_every_n_step 1000 \
        --save_every_n_epoch 2 \
        --operation {operation} \
        --use_docid_rank {use_docid_rank}""")

    print("write success")

if __name__ == '__main__':
    main()