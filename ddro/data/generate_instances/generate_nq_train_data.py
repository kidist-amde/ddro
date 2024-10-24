import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="pq", type=str, help="docid method atomic/pq/url")
parser.add_argument("--scale", default="top_300k", type=str, help="docid method atomic/pq/url")
parser.add_argument("--cur_data", default="general_pretrain", type=str, help="current stage: general_pretrain/search_pretrain/finetune")
args = parser.parse_args()

model = "t5"
cur_data = args.cur_data
encoding = args.encoding # atomic/pq/url
source_docid = "url" if encoding == "pq" else "pq" # label_source_docid
max_seq_length = 128
top_or_rand, scale = args.scale.split("_")
msmarco_or_nq = "msmarco"

def main():
    code_dir = "../"
    if cur_data == "general_pretrain":
        target_data = [("passage", 10), ("sampled_terms", 1), ("enhanced_docid", 1)]
    elif cur_data == "search_pretrain":
        target_data = [("fake_query", 10)]
    elif cur_data == "finetune":
        target_data = [("query", 1)]
    
    for data_name, sample_for_one_doc in target_data:
        print(f"generating {data_name} ...")
        os.system(f"cd {code_dir}/gen_instance/ && python gen_{model}_train_data.py \
            --max_seq_length {max_seq_length} \
            --pretrain_model_path {code_dir}/transformer_models/{model}-base \
            --data_path {code_dir}/dataset/{msmarco_or_nq}-data/{msmarco_or_nq}-docs-sents.{top_or_rand}.{scale}.json \
            --docid_path {code_dir}/dataset/encoded_docid/{model}_{encoding}_{top_or_rand}_{scale}.txt \
            --source_docid_path {code_dir}/dataset/encoded_docid/{model}_{source_docid}_{top_or_rand}_{scale}.txt \
            --query_path {code_dir}/dataset/{msmarco_or_nq}-data/{msmarco_or_nq}-doctrain-queries.tsv \
            --qrels_path {code_dir}/dataset/{msmarco_or_nq}-data/{msmarco_or_nq}-doctrain-qrels.tsv \
            --output_path {code_dir}/dataset/train_data_{top_or_rand}_{scale}/{data_name}.{model}_{max_seq_length}_{sample_for_one_doc}.{encoding}.{scale}.json \
            --fake_query_path {code_dir}/dataset/{msmarco_or_nq}-data/{msmarco_or_nq}_fake_query_10.txt\
            --sample_for_one_doc {sample_for_one_doc} \
            --current_data {data_name}")
    
    if cur_data == "general_pretrain":
        passage_input = f"{code_dir}/dataset/train_data_{top_or_rand}_{scale}/passage.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
        sampled_input = f"{code_dir}/dataset/train_data_{top_or_rand}_{scale}/sampled_terms.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
        docid_input = f"{code_dir}/dataset/train_data_{top_or_rand}_{scale}/enhanced_docid.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
        merge_output = f"{code_dir}/dataset/train_data_{top_or_rand}_{scale}/pretrain.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
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
        fakequery_input = f"{code_dir}/dataset/train_data_{top_or_rand}_{scale}/fake_query.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
        merge_output = f"{code_dir}/dataset/train_data_{top_or_rand}_{scale}/search_pretrain.{model}_{max_seq_length}_10.{encoding}.{scale}.json"
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
        query_input = f"{code_dir}/dataset/train_data_{top_or_rand}_{scale}/query.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
        merge_output = f"{code_dir}/dataset/train_data_{top_or_rand}_{scale}/finetune.{model}_{max_seq_length}_1.{encoding}.{scale}.json"
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