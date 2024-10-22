import os
import nltk
import json
import random
from tqdm import tqdm
from collections import defaultdict

def generate_top_documents(doc_click_count, scale = "300k"):  # all clicked documents in the train set, almost 10%
    print(f"generating top {scale} dataset.")
    input_path = "../dataset/msmarco-data/msmarco-docs-sents-all.json"
    output_path = f"../dataset/msmarco-data/msmarco-docs-sents.top.{scale}.json"
    count = 0
    with open(input_path, "r") as fr:
        with open(output_path, "w") as fw:
            for line in fr:
                docid = json.loads(line)["docid"]
                if doc_click_count[docid] <= 0:
                    continue
                fw.write(line)
                count += 1
    print(f"count of top {scale}: ", count)

def generate_random_documents(scale = "300k"):  # 10% docs from the whole corpus
    print(f"generating rand {scale} dataset.")
    input_path = "../dataset/msmarco-data/msmarco-docs-sents-all.json"
    output_path = f"../dataset/msmarco-data/msmarco-docs-sents.rand.{scale}.json"

    rand_300k_docids = []  # the same docids as our used rand 300k dataset
    with open("../dataset/msmarco-data/msmarco-docids.rand.300k.txt", "r") as fr:
        for line in fr:
            rand_300k_docids.append(line.strip())

    count = 0    
    with open(input_path, "r") as fr:
        with open(output_path, "w") as fw:
            for line in tqdm(fr, desc="reading all docs"):
                docid = json.loads(line)["docid"]
                if docid in rand_300k_docids:
                    fw.write(line)
                    count += 1
    print(f"count of random {scale}: ", count)

if __name__ == '__main__':
    doc_file_path = "../dataset/msmarco-data/msmarco-docs.tsv"
    qrels_train_path = "../dataset/msmarco-data/msmarco-doctrain-qrels.tsv"
    qrels_dev_path = "../dataset/msmarco-data/msmarco-docdev-qrels.tsv"
    fout = open("../dataset/msmarco-data/msmarco-docs-sents-all.json", "w")
    id_to_content = {}
    doc_click_count = defaultdict(int)
    content_to_id = {}

    with open(doc_file_path) as fin:
        for i, line in tqdm(enumerate(fin)):
            cols = line.split("\t")
            if len(cols) != 4:
                continue
            docid, url, title, body = cols
            sents = nltk.sent_tokenize(body)
            id_to_content[docid] = {"docid": docid, "url": url, "title": title, "body": body, "sents": sents}
            doc_click_count[docid] = 0

    print("Total number of unique documents: ", len(doc_click_count))

    with open(qrels_train_path, "r") as fr:
        for line in tqdm(fr):
            queryid, _, docid, _ = line.strip().split()
            doc_click_count[docid] += 1

    # 所有doc按照点击query的数量(popularity)由高到低选择，优先使用点击次数多的doc  
    sorted_click_count = sorted(doc_click_count.items(), key=lambda x:x[1], reverse=True)
    print("sorted_click_count: ", sorted_click_count[:100])
    for docid, count in sorted_click_count:
        if docid not in id_to_content:
            continue
        fout.write(json.dumps(id_to_content[docid])+"\n")

    fout.close()

    # generate top/random 100k/200k/300k dataset
    generate_top_documents(doc_click_count, scale = "300k")
    generate_random_documents(scale = "300k")