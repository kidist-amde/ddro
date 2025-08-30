import os
import json
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge MS MARCO training instances by stage.")
    parser.add_argument("--encoding", default="url_title", choices=["pq", "url_title"], help="Document ID encoding method")
    parser.add_argument("--cur_data", default="general_pretrain", choices=["general_pretrain", "search_pretrain", "finetune"], help="Training stage")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length")
    return parser.parse_args()


def main():
    args = parse_arguments()

    model = "t5"
    cur_data = args.cur_data
    encoding = args.encoding
    max_seq_length = args.max_seq_length
    dataset = "msmarco"
    scale ="300k"
    encoding = args.encoding
    cur_data = "query_dev"
    base_path = "resources/datasets/processed"
    output_dir = f"{base_path}/{dataset}-data/eval_data"

    os.makedirs(output_dir, exist_ok=True)

    os.system(f" python src/data/data_prep/build_t5_data/generate_eval_instances.py \
        --max_seq_length {max_seq_length} \
        --pretrain_model_path {model}-base \
        --data_path resources/datasets/processed/{dataset}-docs-sents.top.300k.json \
        --docid_path resources/datasets/processed/{dataset}-data/encoded_docid/{encoding}_docid.txt \
        --query_path resources/datasets/raw/{dataset}-data/{dataset}-docdev-queries.tsv.gz \
        --qrels_path resources/datasets/raw/{dataset}-data/{dataset}-docdev-qrels.tsv.gz \
        --output_path {output_dir}/{cur_data}.{encoding}.jsonl \
        --current_data {cur_data}")

    print("write success")

if __name__ == '__main__':
    main()