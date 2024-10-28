import os
import argparse
import subprocess

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()

    # Add arguments for all the variables
    parser.add_argument("--encoding", default="url", type=str, help="docid method: atomic/pq/url")
    parser.add_argument("--qrels_path", default="../data/raw/msmarco-docdev-qrels.tsv.gz", type=str, help="Path to qrels file")
    parser.add_argument("--query_path", default="../data/raw/msmarco-docdev-queries.tsv.gz", type=str, help="Path to queries file")
    parser.add_argument("--pretrain_model_path", default="/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro/resources/transformer_models/t5-base", type=str, help="Path to pre-trained T5 model")
    parser.add_argument("--msmarco_or_nq", default="nq", type=str, help="Dataset type: msmarco or nq")
    parser.add_argument("--max_seq_length", default=128, type=int, help="Maximum sequence length")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--docid_path", type=str, required=True, help="Path to the document ID file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for saving results")
    parser.add_argument("--current_data", default="query_dev", type=str, help="Current data type")
    parser.add_argument("--sample_for_one_doc", default=1, type=int, help="Number of samples for one document")
    parser.add_argument("--model", default="t5", type=str, help="Model type: t5 or bert")

    args = parser.parse_args()

    # Parameters (fixed in the code)
    model = args.model
    cur_data = args.current_data 
    msmarco_or_nq = args.msmarco_or_nq

    # Base directory
    code_dir = "/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro"

    # Command to generate evaluation data
    command = [
        "python", f"{code_dir}/data/generate_instances/generate_{model}_{msmarco_or_nq}_eval_data.py",
        "--max_seq_length", str(args.max_seq_length),
        "--pretrain_model_path", args.pretrain_model_path,
        "--data_path", args.data_path,
        "--docid_path", args.docid_path,
        "--query_path", args.query_path,
        "--qrels_path", args.qrels_path,
        "--output_path", args.output_path,
        "--current_data", cur_data
    ]

    try:
        # First change directory to where the Python script exists
        os.chdir(f"{code_dir}/data/generate_instances")

        # Run the command with subprocess
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("Write success")
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
    except Exception as e:
        print(f"General error: {str(e)}")

if __name__ == '__main__':
    main()
