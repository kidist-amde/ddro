# -*- coding: utf-8 -*-
import csv
import random
import gzip
import os
from collections import defaultdict
from typing import Dict, List, Tuple
from multiprocessing import Pool, Manager
from tqdm import tqdm

subset = "dev"  # "train" or "dev"

def ensure_directory_exists(filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def load_query_strings(path: str) -> Dict[str, str]:
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return {row[0]: row[1] for row in csv.reader(f, delimiter="\t")}

def load_doc_offsets(path: str) -> Dict[str, int]:
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return {row[0]: int(row[2]) for row in csv.reader(f, delimiter="\t")}

def load_qrels(path: str) -> Dict[str, List[str]]:
    qrel = defaultdict(list)
    with gzip.open(path, 'rt', encoding='utf8') as f:
        for row in csv.reader(f, delimiter=" "):
            qrel[row[0]].append(row[2])
    return qrel

def process_line(args: Tuple[str, Dict, Dict, Dict, int, set]) -> Tuple[str, str, str, str]:
    line, querystring, qrel, docoffset, rank_filter, processed = args
    try:
        topicid, _, unjudged_docid, rank, _, _ = line.strip().split()
        if topicid in processed or int(rank) != rank_filter:
            return None
        if topicid not in qrel or unjudged_docid in qrel[topicid]:
            return None
        pos_docid = random.choice(qrel[topicid])
        processed.add(topicid)
        return topicid, querystring[topicid], pos_docid, unjudged_docid
    except:
        return None

def generate_triples_parallel(outfile: str, triples_to_generate: int):
    stats = defaultdict(int)
    manager = Manager()
    processed = manager.dict()
    rank_filter = random.randint(1, 100)

    querystring = load_query_strings(f"resources/datasets/raw/msmarco-data/msmarco-doc{subset}-queries.tsv.gz")
    docoffset = load_doc_offsets("resources/datasets/raw/msmarco-data/msmarco-docs-lookup.tsv.gz")
    qrel = load_qrels(f"resources/datasets/raw/msmarco-data/msmarco-doc{subset}-qrels.tsv.gz")

    with gzip.open(f"resources/datasets/raw/msmarco-data/msmarco-doc{subset}-top100.gz", 'rt', encoding='utf8') as top100f, \
         open("resources/datasets/raw/msmarco-data/msmarco-docs.tsv", 'r', encoding="utf8") as docs_f, \
         open(outfile, 'w', encoding="utf8") as out:

        def get_url(docid):
            docs_f.seek(docoffset[docid])
            return docs_f.readline().split("\t")[1]

        lines = list(top100f)
        args = [(line, querystring, qrel, docoffset, rank_filter, processed) for line in lines]

        with Pool(os.cpu_count() // 2) as pool:
            for result in tqdm(pool.imap_unordered(process_line, args), total=len(args), desc="Processing lines"):
                if result:
                    topicid, query, pos_id, neg_id = result
                    stats['kept'] += 1
                    out.write(f"{topicid}\t{query}\t{get_url(pos_id)}\t{pos_id}\t{get_url(neg_id)}\t{neg_id}\n")
                    if stats['kept'] >= triples_to_generate:
                        break

    return stats

if __name__ == "__main__":
    outfile = f"resources/datasets/processed/msmarco-data/hard_negatives_DL19/msmarco_{subset}_triples"
    logfile = f"logs-slurm-msmarco-sft/other-logs/triples_{subset}_generation.log"
    ensure_directory_exists(logfile)

    stats = generate_triples_parallel(outfile, triples_to_generate=float('inf'))

    with open(logfile, 'w', encoding='utf8') as log:
        for key, val in stats.items():
            log.write(f"{key}\t{val}\n")
        log.write("\nInterpretation of Statistics:\n")
        log.write("- skipped: query already processed or rank mismatch\n")
        log.write("- kept: valid triples generated\n")

    print(f"Log saved at: {logfile}")
