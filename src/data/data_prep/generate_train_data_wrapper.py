import os
import json
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge MS MARCO training instances by stage.")
    parser.add_argument("--encoding", default="url", choices=["pq", "url"], help="Document ID encoding method")
    parser.add_argument("--scale", type=str, required=True, help="Dataset scale (e.g., 300k or custom string)")
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
    scale = args.scale

    base_path = "resources/datasets/processed"
    output_dir = f"{base_path}/{dataset}-data/train_data_top_{scale}"

    os.makedirs(output_dir, exist_ok=True)

    stage_config = {
        "general_pretrain": ["passage", "sampled_terms", "enhanced_docid"],
        "search_pretrain": ["fake_query"],
        "finetune": ["query"]
    }

    if cur_data not in stage_config:
        raise ValueError(f"Unsupported cur_data value: {cur_data}")

    merged_output = f"{output_dir}/{cur_data}.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
    input_files = [f"{output_dir}/{name}.jsonl" for name in stage_config[cur_data]]

    print(f"Merging files into {merged_output}...")
    total_count = 0
    with open(merged_output, "w") as fout:
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"Warning: {input_file} not found, skipping.")
                continue
            with open(input_file, "r") as fin:
                for line in tqdm(fin, desc=f"Merging {os.path.basename(input_file)}"):
                    fout.write(line)
                    total_count += 1
            # if cur_data in ["search_pretrain", "finetune"]:
            #     os.remove(input_file)

    print(f"Total number of {cur_data} samples: {total_count}")
    print("Training data merge complete.")


if __name__ == "__main__":
    main()
