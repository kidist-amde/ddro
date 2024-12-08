import json

file_path = "/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro/resources/datasets/processed/msmarco-data/msmarco-docs-sents.top.300k.json"

with open(file_path, "r") as f:
    # read one line
    line = f.readline()
    print(line)