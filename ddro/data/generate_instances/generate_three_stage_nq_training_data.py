import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="url", type=str, help="docid method atomic/pq/url")
parser.add_argument("--cur_data", default="general_pretrain", type=str, help="current stage: general_pretrain/search_pretrain/finetune")
parser.add_argument("--query_path", default="../data/raw/msmarco-doctrain-queries.tsv.gz", type=str, help='data path')
parser.add_argument("--qrels_path", default="../data/raw/msmarco-doctrain-qrels.tsv.gz", type=str, help='data path')
parser.add_argument("--pretrain_model_path", default="./transformer_models/t5-base", type=str, help='bert model path')
parser.add_argument("--msmarco_or_nq", default="nq", type=str, help="Dataset type: msmarco or nq")
parser.add_argument("--max_seq_length", default=128, type=int, help="Maximum sequence length")
parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset JSON file")
parser.add_argument("--docid_path", type=str, required=True, help="Path to the document ID file")
parser.add_argument("--fake_query_path", default="", type=str, help="Path to the fake query file")
parser.add_argument("--current_data", default="query_dev", type=str, help="Current data type")
parser.add_argument("--sample_for_one_doc", default=1, type=int, help="Number of samples for one document")
parser.add_argument("--model", default="t5", type=str, help="Model type: t5 or bert")

args = parser.parse_args()

model = args.model
cur_data = args.cur_data
encoding = args.encoding # atomic/pq/url
source_docid = "url" if encoding == "pq" else "pq" # label_source_docid
max_seq_length = args.max_seq_length
msmarco_or_nq = args.msmarco_or_nq


def main():
    code_dir = "/gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro"
    output_dir = os.path.join(code_dir, "resources/datasets/processed/nq-data/train_data")
    os.makedirs(output_dir, exist_ok=True)
    if cur_data == "general_pretrain":
        target_data = [("passage", 10), ("sampled_terms", 1), ("enhanced_docid", 1)]
    elif cur_data == "search_pretrain":
        target_data = [("fake_query", 10)]
    elif cur_data == "finetune":
        target_data = [("query", 1)]
     
    for data_name, sample_for_one_doc in target_data:
         print(f"generating {data_name} ...")
         os.system(f"cd {code_dir}/data/generate_instances/ && python generate_{model}_{msmarco_or_nq}_train_data.py "
         f"--max_seq_length {max_seq_length} "
         f"--pretrain_model_path {args.pretrain_model_path} "
         f"--data_path {args.data_path} "
         f"--docid_path {args.docid_path} "
         f"--source_docid_path {code_dir}/resources/datasets/processed/nq-data/encoded_docid/{model}_512_{source_docid}_docids.txt "
         f"--query_path {args.query_path} "
         f"--qrels_path {args.qrels_path} "
         f"--output_path {output_dir}/{data_name}.{model}_{max_seq_length}_{sample_for_one_doc}.{encoding}_{msmarco_or_nq}.json "
         f"--fake_query_path {args.fake_query_path} "
         f"--sample_for_one_doc {sample_for_one_doc} "
         f"--current_data {data_name}")

    if cur_data == "general_pretrain":
        passage_input = f"{code_dir}/resources/datasets/processed/nq-data/train_data/passage.{model}_{max_seq_length}_10.{encoding}_{msmarco_or_nq}.json"
        sampled_input = f"{code_dir}/resources/datasets/processed/nq-data/train_data/sampled_terms.{model}_{max_seq_length}_1.{encoding}_{msmarco_or_nq}.json"
        docid_input = f"{code_dir}/resources/datasets/processed/nq-data/train_data/enhanced_docid.{model}_{max_seq_length}_1.{encoding}_{msmarco_or_nq}.json"
        merge_output = f"{code_dir}/resources/datasets/processed/nq-data/train_data/pretrain.{model}_{max_seq_length}_10.{encoding}_{msmarco_or_nq}.json"
        fout = open(merge_output, "w")
        total_count = 0
        with open(passage_input, "r") as fr:
            for line in tqdm(fr, desc="loading passage input"):
                fout.write(line)
                total_count += 1
        with open(sampled_input, "r") as fr:
            for line in tqdm(fr, desc="loading sampled terms input"):
                fout.write(line)
                total_count += 1
        with open(docid_input, "r") as fr:
            for line in tqdm(fr, desc="loading docid input"):
                fout.write(line)
                total_count += 1
        fout.close()
        print("total number of pretrain samples: ", total_count)

    elif cur_data == "search_pretrain":
        fakequery_input = f"{code_dir}/resources/datasets/processed/nq-data/train_data/fake_query.{model}_{max_seq_length}_10.{encoding}_{msmarco_or_nq}.json"
        merge_output = f"{code_dir}/resources/datasets/processed/nq-data/train_data/search_pretrain.{model}_{max_seq_length}_10.{encoding}_{msmarco_or_nq}.json"
        fout = open(merge_output, "w")
        total_count = 0
        with open(fakequery_input, "r") as fr:
            for line in tqdm(fr, desc="loading fakequery input"):
                fout.write(line)
                total_count += 1
        fout.close()
        print("total number of search pretrain samples: ", total_count)
        os.system(f"rm {fakequery_input}")
        
    elif cur_data == "finetune":
        query_input = f"{code_dir}/resources/datasets/processed/nq-data/train_data/query.{model}_{max_seq_length}_1.{encoding}_{msmarco_or_nq}.json"
        merge_output = f"{code_dir}/resources/datasets/processed/nq-data/train_data/finetune.{model}_{max_seq_length}_1.{encoding}_{msmarco_or_nq}.json"
        fout = open(merge_output, "w")
        total_count = 0
        with open(query_input, "r") as fr:
            for line in tqdm(fr, desc="loading query input"):
                fout.write(line)
                total_count += 1
        fout.close()
        print("total number of finetune samples: ", total_count)
        os.system(f"rm {query_input}")

    print("write success")

if __name__ == '__main__':
    main()