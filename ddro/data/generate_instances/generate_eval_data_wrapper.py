import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate MS MARCO evaluation data instances.")
    parser.add_argument("--encoding", default="url", choices=["atomic", "pq", "url"], help="Document ID encoding method")
    parser.add_argument("--scale", default="top_300k", choices=["top_300k", "rand_300k"], help="Dataset scale")
    parser.add_argument("--qrels_path", default="data/msmarco-docdev-qrels.tsv.gz", help="Path to qrels file")
    parser.add_argument("--query_path", default="data/msmarco-docdev-queries.tsv.gz", help="Path to query file")
    parser.add_argument("--pretrain_model_path", default="transformer_models/t5-base", help="Path to pretrained model")
    return parser.parse_args()

def main():
    args = parse_arguments()

    model = "t5"
    max_seq_length = 128
    sample_for_one_doc = 1
    current_data = "query_dev"

    top_or_rand, scale = args.scale.split("_")
    dataset_name = "msmarco"

    base_dir = "resources/datasets/processed"
    data_path = f"{base_dir}/{dataset_name}-data/{dataset_name}-docs-sents.{top_or_rand}.{scale}.json"
    docid_path = f"{base_dir}/{dataset_name}-data/encoded_docid/{model}_512_{args.encoding}_{top_or_rand}.{scale}.txt"
    output_path = f"{base_dir}/{dataset_name}-data/test_data_{top_or_rand}_{scale}/query_dev.{model}_{max_seq_length}_{sample_for_one_doc}.{args.encoding}.{scale}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = (
        f"python generate_eval_instances.py "
        f"--max_seq_length {max_seq_length} "
        f"--pretrain_model_path {args.pretrain_model_path} "
        f"--data_path {data_path} "
        f"--docid_path {docid_path} "
        f"--query_path {args.query_path} "
        f"--qrels_path {args.qrels_path} "
        f"--output_path {output_path} "
        f"--current_data {current_data}"
    )

    print("Running:", command)
    os.system(command)
    print("Evaluation data generation completed.")

if __name__ == "__main__":
    main()