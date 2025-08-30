import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["msmarco", "nq"], required=True, help="Dataset name")
    parser.add_argument("--encoding", default="pq", type=str, help="docid method: pq, or url")
    parser.add_argument("--scale", default="top_300k", type=str, help="Document scale: top_300k")
    args = parser.parse_args()

    # Load configuration
    config_path = f"src/scripts/config_{args.dataset}.json" if args.dataset == "nq" else "src/scripts/configs/config.json"
    config_file = json.load(open(config_path, "r"))
    config = config_file[args.encoding]

    encoding = config["encoding"]
    add_doc_num = config["add_doc_num"]
    max_docid_length = config["max_docid_length"]
    top_300k, scale = args.scale.split("_")
    use_origin_head = config["use_origin_head"]  # Get from config

    # Static configs - Updated paths based on your project structure
    model = "t5_128_1"
    model_name = "DDRO"
    num_beams = 50 if args.dataset == "nq" else 15
    max_seq_length = 64
    cur_data = "query_dev"

    # Construct command with correct paths
    run_cmd = f"""python src/pretrain/eval_ddro_docid_ranking.py \
        --per_gpu_batch_size {2 if args.dataset == "nq" else 4} \
        --save_path resources/checkpoints/{args.dataset}/dpo_ckpt_{encoding}/dpo_model_final.pkl \
        --log_path logs/{args.dataset}/dpo_{model_name}_{encoding}.log \
        --pretrain_model_path t5-base \
        --docid_path resources/datasets/processed/{args.dataset}-data/encoded_docid/{"pq_docid.txt" if encoding == "pq" else "url_title_docid.txt" if encoding == "url" else f"{encoding}_docid.txt"} \
        --test_file_path resources/datasets/processed/{args.dataset}-data/eval_data/{cur_data}_{encoding}.jsonl \
        --dataset_script_dir src/data/data_scripts \
        --dataset_cache_dir ./cache \
        --num_beams {num_beams} \
        --add_doc_num {add_doc_num} \
        --max_seq_length {max_seq_length} \
        --max_docid_length {max_docid_length} \
        --use_origin_head False \
        --use_docid_rank True"""

    print("Running evaluation...")
    print(f"Command: {run_cmd}")
    
    # Create necessary directories
    os.makedirs(f"logs/{args.dataset}", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    os.system(run_cmd)
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()