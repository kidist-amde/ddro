import os
import json
import argparse

# Why: let launcher emit HF URIs for docids/tests; evaluator will download.

DOCID_FILENAME_MAP = {
    ("msmarco", "pq"): "pq_msmarco_docids.txt",
    ("msmarco", "url"): "tu_msmarco_docids.txt",       # url_title -> tu
    ("msmarco", "url_title"): "tu_msmarco_docids.txt",
    ("nq", "pq"): "pq_nq_docids.txt",
    ("nq", "url"): "tu_nq_docids.txt",
    ("nq", "url_title"): "tu_nq_docids.txt",
}

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="pq", type=str, help="docid method pq/url/url_title")
parser.add_argument("--scale", default="top_300k", type=str, help="scale: top_300k, rand_100k, etc.")
parser.add_argument("--dataset", default="msmarco", type=str, help="dataset: msmarco/nq")

# New: HF repos (override if you used different repo names)
parser.add_argument("--hf_docids_repo", default="kiyam/ddro-docids", type=str)
parser.add_argument("--hf_tests_repo", default="kiyam/ddro-testsets", type=str)

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
use_origin_head = config.get("use_origin_head", False)  # kept for completeness

# Determine num_beams based on dataset and encoding
num_beams = (
    80  if (args.dataset == "msmarco" and args.encoding == "pq") else
    100 if (args.dataset == "nq" and args.encoding == "pq") else
    50  if (args.dataset == "nq" and args.encoding in {"url", "url_title"}) else
    15
)

model_suffix = "tu" if encoding == "url_title" else "pq"
model = "t5_128_1"
cur_data = "query_dev"
use_docid_rank = "True"
operation = "testing"
max_seq_length = 64
model_name = "DDRO"

# Paths (logs/cache remain local)
code_dir = os.getcwd()

def main():
    top_or_rand, scale = args.scale.split("_")

    # HF docids filename mapping
    enc_key = "url_title" if encoding == "url_title" else ("url" if encoding == "url" else "pq")
    docid_filename = DOCID_FILENAME_MAP.get((args.dataset, enc_key))
    if not docid_filename:
        raise ValueError(f"No docid mapping for dataset={args.dataset}, encoding={enc_key}")

    # Build HF URIs
    # Test file naming mirrors your previous local layout; ensure same names on HF.
    if args.dataset == "nq":
        test_rel = f"nq/test_data/{cur_data}.{model}.{encoding}_nq.json"
    else:
        encoding_filename = "url" if encoding == "url_title" else encoding
        test_rel = f"msmarco/test_data_{args.scale}/{cur_data}.{model}.{encoding_filename}.{args.scale}.json"

    docid_uri = f"hf:dataset:{args.hf_docids_repo}:{docid_filename}"
    test_uri = f"hf:dataset:{args.hf_tests_repo}:{test_rel}"

    # Logging info
    print(f"=== Evaluation Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Encoding: {args.encoding} (config encoding: {encoding})")
    print(f"Scale: {args.scale}")
    print(f"Num beams: {num_beams}")
    print(f"DocIDs: {docid_uri}")
    print(f"Test file: {test_uri}")
    print(f"================================")

    # Build eval command (pass HF URIs directly)
    eval_cmd = f"""
    python src/pretrain/hf_eval/eval_hf_docid_ranking.py \
        --per_gpu_batch_size 4 \
        --pretrain_model_path kiyam/ddro-{args.dataset}-{model_suffix} \
        --log_path logs/{args.dataset}/dpo_{model_name}_{encoding}.log \
        --test_file_path "{test_uri}" \
        --docid_path "{docid_uri}" \
        --dataset_script_dir src/data/data_scripts \
        --dataset_cache_dir {code_dir}/negs_tutorial_cache \
        --num_beams {num_beams} \
        --add_doc_num {add_doc_num} \
        --max_seq_length {max_seq_length} \
        --max_docid_length {max_docid_length} \
        --use_docid_rank {use_docid_rank} \
        --docid_format {args.dataset} \
        --lookup_fallback True \
        --assert_strict True
    """.strip()

    os.makedirs(f"{code_dir}/logs/{args.dataset}", exist_ok=True)
    print("\nStarting evaluation...\n")
    rc = os.system(eval_cmd)
    if rc != 0:
        raise SystemExit(rc)

    print("\nEvaluation completed")
    print("=== Evaluation Completed ===")


if __name__ == '__main__':
    main()
