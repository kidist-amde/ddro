import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="url", type=str, help="docid method atomic/pq/url")
parser.add_argument("--scale", default="top_300k", type=str, help="docid method atomic/pq/url")
parser.add_argument("--qrels_path", default="../data/raw/msmarco-docdev-qrels.tsv.gz", type=str, help='data path')
parser.add_argument("--query_path", default="../data/raw/msmarco-docdev-queries.tsv.gz", type=str, help='data path')
parser.add_argument("--pretrain_model_path", default="transformer_models/t5-base", type=str, help='bert model path')
args = parser.parse_args()

model = "t5"
encoding = args.encoding
max_seq_length = 128
sample_for_one_doc = 1
cur_data = "query_dev"
top_or_rand, scale = args.scale.split("_")
msmarco_or_nq = "msmarco" # msmarco or nq

def main():
    code_dir = "/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro"
    os.system(f"cd {code_dir}/data/generate_instances/ && python generate_{model}_{msmarco_or_nq}_eval_data.py \
        --max_seq_length {max_seq_length} \
        --pretrain_model_path {args.pretrain_model_path} \
        --data_path {code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/{msmarco_or_nq}-docs-sents.{top_or_rand}.{scale}.json \
        --docid_path {code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/encoded_docid/{model}_512_{encoding}_{top_or_rand}.{scale}.txt \
        --query_path {args.query_path} \
        --qrels_path {args.qrels_path} \
        --output_path {code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/test_data_{top_or_rand}_{scale}/query_dev.{model}_{max_seq_length}_{sample_for_one_doc}.{encoding}.{scale}.json \
        --current_data {cur_data}")  

    print("write success")

if __name__ == '__main__':
    main()