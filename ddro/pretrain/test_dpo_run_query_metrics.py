import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["msmarco", "nq"], required=True, help="Dataset name")
    parser.add_argument("--encoding", default="pq", type=str, help="docid method: atomic, pq, or url_title")
    parser.add_argument("--scale", default="top_300k", type=str, help="Document scale: top_300k or rand_300k")
    args = parser.parse_args()

    # Load configuration
    config_path = f"scripts/config_{args.dataset}.json" if args.dataset == "nq" else "scripts/config.json"
    config_file = json.load(open(config_path, "r"))
    config_file["atomic"]["add_doc_num"] = config_file["doc_num"][args.scale]
    config = config_file[args.encoding]

    encoding = config["encoding"]
    add_doc_num = config["add_doc_num"]
    max_docid_length = config["max_docid_length"]
    use_origin_head = config["use_origin_head"]
    top_or_rand, scale = args.scale.split("_")

    # Static configs
    code_dir = "ddro"
    model = "t5_128_1"
    model_name = "DDRO"
    num_beams = 80 if args.dataset == "nq" else 5
    operation = "testing"
    max_seq_length = 64
    cur_data = "query_dev"

    # Construct command
    run_cmd = f"""cd {code_dir} && python run_dpo_evaluation.py \
        --epoch 10 \
        --per_gpu_batch_size {2 if args.dataset == "nq" else 4} \
        --save_path {code_dir}/outputs/{args.dataset}/dpo_ckpt_{encoding}_final/dpo_model_final.pkl \
        --log_path {code_dir}/logs/{args.dataset}/dpo_{model_name}_{encoding}.log \
        --doc_file_path {code_dir}/resources/datasets/processed/{args.dataset}-data/{'nq-merged-json/nq-docs-sents.json' if args.dataset == 'nq' else f'msmarco-docs-sents.{top_or_rand}.{scale}.json'} \
        --pretrain_model_path {code_dir}/resources/transformer_models/t5-base \
        --docid_path {code_dir}/resources/datasets/processed/{args.dataset}-data/encoded_docid/t5_512_{encoding}_docids.txt \
        --test_file_path {code_dir}/resources/datasets/processed/{args.dataset}-data/test_data/{cur_data}.{model}.{encoding}_{args.dataset}.json \
        --dataset_script_dir {code_dir}/data/data_scripts \
        --dataset_cache_dir {code_dir}/negs_tutorial_cache \
        --num_beams {num_beams} \
        --add_doc_num {add_doc_num} \
        --max_seq_length {max_seq_length} \
        --max_docid_length {max_docid_length} \
        --output_every_n_step 1000 \
        --save_every_n_epoch 2 \
        --operation {operation} \
        --use_docid_rank True"""

    print("Running evaluation...")
    os.system(run_cmd)
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()
