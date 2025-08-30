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

    base_path = "resources/datasets/processed"
    output_dir = f"{base_path}/{dataset}-data/train_data_top_{scale}"

    os.makedirs(output_dir, exist_ok=True)


    if cur_data == "general_pretrain":
        target_data = [("passage", 10), ("sampled_terms", 1), ("enhanced_docid", 1)]
    elif cur_data == "search_pretrain":
        target_data = [("fake_query", 10)]
    elif cur_data == "finetune":
        target_data = [("query", 1)]
    
    for data_name, sample_for_one_doc in target_data:
        print(f"generating {data_name} ...")
        os.system(f"python src/data/data_prep/build_t5_data/generate_train_instances.py \
            --max_seq_length {max_seq_length} \
            --pretrain_model_path {model}-base \
            --data_path resources/datasets/processed/{dataset}-docs-sents.top.300k.json \
            --docid_path resources/datasets/processed/{dataset}-data/encoded_docid/{encoding}_docid.txt \
            --source_docid_path resources/datasets/processed/{dataset}-data/encoded_docid/{encoding}_docid.txt \
            --query_path resources/datasets/raw/{dataset}-data/{dataset}-doctrain-queries.tsv.gz \
            --qrels_path resources/datasets/raw/{dataset}-data/{dataset}-doctrain-qrels.tsv.gz \
            --output_path {output_dir}/{data_name}.{model}_{max_seq_length}_{sample_for_one_doc}.{encoding}.json \
            --fake_query_path /resources/datasets/processed/{dataset}-data/{dataset}_pseudo_query_10.txt \
            --sample_for_one_doc {sample_for_one_doc} \
            --current_data {data_name}")
    
    if cur_data == "general_pretrain":
        passage_input = f"{output_dir}/passage.{model}_{max_seq_length}_10.{encoding}.json"
        sampled_input = f"{output_dir}/sampled_terms.{model}_{max_seq_length}_1.{encoding}.json"
        docid_input = f"{output_dir}/enhanced_docid.{model}_{max_seq_length}_1.{encoding}.json"
        merge_output = f"{output_dir}/pretrain.{model}_{max_seq_length}_10.{encoding}.json"
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
        fakequery_input =  f"{output_dir}/fake_query.{model}_{max_seq_length}_10.{encoding}.json"
        merge_output =  f"{output_dir}/search_pretrain.{model}_{max_seq_length}_10.{encoding}.json"
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
        query_input =  f"{output_dir}/query.{model}_{max_seq_length}_1.{encoding}.json"
        merge_output =  f"{output_dir}/finetune.{model}_{max_seq_length}_1.{encoding}.json"
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