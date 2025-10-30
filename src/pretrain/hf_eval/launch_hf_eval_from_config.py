# File: src/pretrain/hf_eval/launch_hf_eval_from_config.py

import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="pq", type=str, help="docid method atomic/pq/url")
parser.add_argument("--scale", default="top_300k", type=str, help="scale: top_300k, rand_100k, etc.")
parser.add_argument("--dataset", default="msmarco", type=str, help="dataset: msmarco/nq")
args = parser.parse_args()

# Load correct config file based on dataset
config_file_path = (
    f"src/scripts/configs/config_{args.dataset}.json" if args.dataset == "nq" 
    else "src/scripts/configs/config.json"
)
with open(config_file_path, "r") as f:
    config_file = json.load(f)

# Get config for the specified encoding
config = config_file[args.encoding]
encoding = config["encoding"]
add_doc_num = config["add_doc_num"]
max_docid_length = config["max_docid_length"]
use_origin_head = config["use_origin_head"]

code_dir = "/ivi/ilps/personal/kmekonn/projects/LDDRO/ddro"

# Determine num_beams based on dataset and encoding
num_beams = (
    80  if (args.dataset == "msmarco" and args.encoding == "pq") else
    100 if (args.dataset == "nq" and args.encoding == "pq") else
    50  if (args.dataset == "nq" and args.encoding == "url") else
    15
)

model_suffix = "tu" if encoding == "url_title" else "pq"

# Logging info
print(f"=== Evaluation Configuration ===")
print(f"Dataset: {args.dataset}")
print(f"Encoding: {args.encoding}")
print(f"Config encoding: {encoding}")
print(f"Scale: {args.scale}")
print(f"Num beams: {num_beams}")
print(f"================================")

# Static settings
model = "t5_128_1"
cur_data = "query_dev"
use_docid_rank = "True"
operation = "testing"
max_seq_length = 64
model_name = "DDRO"
top_or_rand, scale = args.scale.split("_")

def main():
    # Create log directory
    log_dir = f"{code_dir}/logs/{args.dataset}"
    os.makedirs(log_dir, exist_ok=True)

    # Setup paths
    base_path = "/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro/resources"

    if args.dataset == "nq":
        test_file = f"{base_path}/datasets/processed/nq-data/test_data/{cur_data}.{model}.{encoding}_nq.json"
        docid_file = f"{base_path}/datasets/processed/nq-data/encoded_docid/t5_512_{encoding}_docids.txt"
    else:
        encoding_filename = "url" if encoding == "url_title" else encoding
        test_file = f"{base_path}/datasets/processed/msmarco-data/test_data_{args.scale}/{cur_data}.{model}.{encoding_filename}.{scale}.json"
        encoding_for_docid = "url" if encoding == "url_title" else args.encoding
        docid_file = f"{base_path}/ENCODED_DOC_IDs/t5_{encoding_for_docid}_msmarco.txt"

    checkpoint_path = f"resources/checkpoints/{args.dataset}/dpo_ckpt_{encoding}/dpo_model_final.pkl"

    # Check required files exist
    files_to_check = {
        "Test file": test_file,
        "Docid file": docid_file,
        "Checkpoint": checkpoint_path
    }

    print("\nFile validation:")
    all_files_exist = True
    for desc, path in files_to_check.items():
        if os.path.exists(path):
            print(f"[FOUND] {desc}: {path}")
        else:
            print(f"[NOT FOUND] {desc}: {path}")
            all_files_exist = False

    if not all_files_exist:
        print("\nERROR: Some required files are missing. Please check the paths above.")
        return

    print(f"\nStarting evaluation...\n")

    # Build eval command
    eval_cmd = f"""
    python src/pretrain/hf_eval/eval_hf_docid_ranking.py \
        --per_gpu_batch_size 4 \
        --pretrain_model_path kiyam/ddro-{args.dataset}-{model_suffix} \
        --log_path logs/{args.dataset}/dpo_{model_name}_{encoding}.log \
        --test_file_path {test_file} \
        --docid_path {docid_file} \
        --dataset_script_dir src/data/data_scripts \
        --dataset_cache_dir {code_dir}/negs_tutorial_cache \
        --num_beams {num_beams} \
        --add_doc_num {add_doc_num} \
        --max_seq_length {max_seq_length} \
        --max_docid_length {max_docid_length} \
        --use_docid_rank {use_docid_rank} \
        --docid_format {args.dataset} \
        --lookup_fallback True \
        --assert_strict True \

    """

    os.system(eval_cmd)

    print("\nEvaluation completed successfully")
    print("=== Evaluation Completed ===")


if __name__ == '__main__':
    main()
