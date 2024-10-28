import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="url", type=str, help="docid method atomic/pq/url")
parser.add_argument("--scale", default="top_300k", type=str, help="docid method atomic/pq/url")
parser.add_argument("--cur_data", default="general_pretrain", type=str, help="current stage: general_pretrain/search_pretrain/finetune")
parser.add_argument("--query_path", default="../data/raw/msmarco-doctrain-queries.tsv.gz", type=str, help='data path')
parser.add_argument("--qrels_path", default="../data/raw/msmarco-doctrain-qrels.tsv.gz", type=str, help='data path')
parser.add_argument("--pretrain_model_path", default="./transformer_models/t5-base", type=str, help='bert model path')
parser.add_argument("--fake_query_path", default="./dataset/msmarco-data/msmarco_fake_query_10.txt", type=str, help='fake query path')
args = parser.parse_args()

model = "t5"
cur_data = args.cur_data
encoding = args.encoding # atomic/pq/url
source_docid = "url" if encoding == "pq" else "pq" # label_source_docid
max_seq_length = 128
top_or_rand, scale = args.scale.split("_")
msmarco_or_nq = "msmarco" # msmarco/nq

def main():
    code_dir = "/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro"
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
          f"--data_path {code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/{msmarco_or_nq}-docs-sents.{top_or_rand}.{scale}.json "
          f"--docid_path {code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/encoded_docid/{model}_512_{encoding}_{top_or_rand}.{scale}.txt "
          f"--source_docid_path {code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/encoded_docid/{model}_512_{source_docid}_{top_or_rand}.{scale}.txt "
          f"--query_path {args.query_path} "
          f"--qrels_path {args.qrels_path} "
          f"--output_path {code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/train_data_{top_or_rand}_{scale}/{data_name}.{model}_{max_seq_length}_{sample_for_one_doc}.{encoding}.{scale}.json "
          f"--fake_query_path {args.fake_query_path} "
          f"--sample_for_one_doc {sample_for_one_doc} "
          f"--current_data {data_name}")

    
    if cur_data == "general_pretrain":
        passage_input = f"{code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/train_data_{top_or_rand}_{scale}/passage.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
        sampled_input = f"{code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/train_data_{top_or_rand}_{scale}/sampled_terms.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
        docid_input = f"{code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/train_data_{top_or_rand}_{scale}/enhanced_docid.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
        merge_output = f"{code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/train_data_{top_or_rand}_{scale}/pretrain.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
        merge_output_dir = os.path.dirname(merge_output)
        if not os.path.exists(merge_output_dir):
            os.makedirs(merge_output_dir)
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
        fakequery_input = f"{code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/train_data_{top_or_rand}_{scale}/fake_query.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
        merge_output = f"{code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/train_data_{top_or_rand}_{scale}/search_pretrain.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
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
        query_input = f"{code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/train_data_{top_or_rand}_{scale}/query.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
        merge_output = f"{code_dir}/resources/datasets/processed/{msmarco_or_nq}-data/train_data_{top_or_rand}_{scale}/finetune.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
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