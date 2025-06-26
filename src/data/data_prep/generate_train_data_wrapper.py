
import os
import json
import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate and merge MS MARCO training instances for multiple stages.")
    parser.add_argument("--encoding", default="url", choices=["atomic", "pq", "url"], help="Document ID encoding method")
    parser.add_argument("--scale", default="top_300k", choices=["top_300k", "rand_300k"], help="Dataset scale")
    parser.add_argument("--cur_data", default="general_pretrain", choices=["general_pretrain", "search_pretrain", "finetune"], help="Training stage")
    parser.add_argument("--query_path", default="data/msmarco-doctrain-queries.tsv.gz", help="Query file path")
    parser.add_argument("--qrels_path", default="data/msmarco-doctrain-qrels.tsv.gz", help="Qrels file path")
    parser.add_argument("--pretrain_model_path", default="transformer_models/t5-base", help="Pretrained model path")
    parser.add_argument("--fake_query_path", default="data/msmarco_fake_query_10.txt", help="Synthetic query file path")
    return parser.parse_args()

def main():
    args = parse_arguments()

    model = "t5"
    cur_data = args.cur_data
    encoding = args.encoding
    source_docid = "url_title" if encoding == "pq" else "pq"
    max_seq_length = 128
    top_or_rand, scale = args.scale.split("_")
    dataset = "msmarco"

    base_path = "resources/datasets/processed"
    data_path = f"{base_path}/{dataset}-data/{dataset}-docs-sents.{top_or_rand}.{scale}.json"
    docid_path = f"{base_path}/{dataset}-data/encoded_docid/{model}_512_{encoding}_{top_or_rand}.{scale}.txt"
    source_path = f"{base_path}/{dataset}-data/encoded_docid/{model}_512_{source_docid}_{top_or_rand}.{scale}.txt"
    output_dir = f"{base_path}/{dataset}-data/train_data_{top_or_rand}_{scale}"
    os.makedirs(output_dir, exist_ok=True)

    stage_config = {
        "general_pretrain": [("passage", 10), ("sampled_terms", 1), ("enhanced_docid", 1)],
        "search_pretrain": [("synthetic_query", 10)],
        "finetune": [("query", 1)]
    }

    if cur_data not in stage_config:
        raise ValueError(f"Unsupported cur_data value: {cur_data}")

    for data_name, sample_for_one_doc in stage_config[cur_data]:
        print(f"Generating {data_name} ...")
        output_file = f"{output_dir}/{data_name}.{model}_{max_seq_length}_{sample_for_one_doc}.{encoding}.{scale}.json"
        command = f"python generate_train_instances.py " \
                  f"--max_seq_length {max_seq_length} " \
                  f"--pretrain_model_path {args.pretrain_model_path} " \
                  f"--data_path {data_path} " \
                  f"--docid_path {docid_path} " \
                  f"--source_docid_path {source_path} " \
                  f"--query_path {args.query_path} " \
                  f"--qrels_path {args.qrels_path} " \
                  f"--output_path {output_file} " \
                  f"--fake_query_path {args.fake_query_path} " \
                  f"--sample_for_one_doc {sample_for_one_doc} " \
                  f"--current_data {data_name}"
        os.system(command)

    merged_output = f"{output_dir}/{cur_data}.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
    input_files = [
        f"{output_dir}/{name}.{model}_{max_seq_length}_{count}.{encoding}.{scale}.json"
        for name, count in stage_config[cur_data]
    ]

    print(f"Merging into {merged_output}")
    total_count = 0
    with open(merged_output, "w") as fout:
        for input_file in input_files:
            with open(input_file, "r") as fin:
                for line in tqdm(fin, desc=f"Merging {os.path.basename(input_file)}"):
                    fout.write(line)
                    total_count += 1
            if cur_data in ["search_pretrain", "finetune"]:
                os.remove(input_file)

    print(f"Total number of {cur_data} samples: {total_count}")
    print("Training data generation complete.")

if __name__ == "__main__":
    main()
