import csv
import random
import gzip
import os
from collections import defaultdict
from typing import Dict, List, Tuple
from multiprocessing import Pool, Manager
from tqdm import tqdm
from multiprocessing import Lock


subset = "dev"  # "train" or "dev"

def ensure_directory_exists(filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def load_query_strings(file_path: str) -> Dict[str, str]:
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        return {row[0]: row[1] for row in csv.reader(f, delimiter="\t")}

def load_doc_offsets(file_path: str) -> Dict[str, int]:
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        return {row[0]: int(row[2]) for row in csv.reader(f, delimiter="\t")}

def load_relevance_judgments(file_path: str) -> Dict[str, List[str]]:
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        qrel = defaultdict(list)
        for row in csv.reader(f, delimiter=" "):
            qrel[row[0]].append(row[2])
        return qrel


def process_line_with_lock(args: Tuple[str, Dict, Dict, Dict, int, List, Lock]):
    line, querystring, qrel, docoffset, unjudged_rank_to_keep, processed_docs, lock = args
    try:
        topicid, _, unjudged_docid, rank, _, _ = line.split()
        with lock:
            processed_docs_set = set(processed_docs)  # Convert list to set for lookups
        if topicid in processed_docs_set or int(rank) != unjudged_rank_to_keep:
            return None

        if topicid not in qrel or unjudged_docid in qrel[topicid]:
            return None

        positive_docid = random.choice(qrel[topicid])
        with lock:
            processed_docs.append(topicid)  # Append to shared list safely
        return topicid, querystring[topicid], positive_docid, unjudged_docid
    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def generate_triples_parallel(outfile: str, triples_to_generate: int):
    stats = defaultdict(int)
    manager = Manager()
    processed_docs = manager.list()  # Use a Manager list instead of a set
    lock = manager.Lock()  # Lock for safe access to shared `processed_docs`
    unjudged_rank_to_keep = random.randint(1, 100)

    querystring = load_query_strings(f"resources/datasets/raw/msmarco-data/msmarco-doc{subset}-queries.tsv.gz")
    docoffset = load_doc_offsets("resources/datasets/raw/msmarco-data/msmarco-docs-lookup.tsv.gz")
    qrel = load_relevance_judgments(f"resources/datasets/raw/msmarco-data/msmarco-doc{subset}-qrels.tsv.gz")

    with gzip.open(f"resources/datasets/raw/msmarco-data/msmarco-doc{subset}-top100.gz", 'rt', encoding='utf8') as top100f, \
            gzip.open("resources/datasets/raw/msmarco-data/msmarco-docs.tsv.gz", 'rt', encoding="utf8") as docs_f, \
            open(outfile, 'w', encoding="utf8") as out:

        def doc_url(docid):
            docs_f.seek(docoffset[docid])
            return docs_f.readline().split("\t")[1]

        lines = (line for line in top100f)  # Iterator to avoid loading all lines into memory

        args = ((line, querystring, qrel, docoffset, unjudged_rank_to_keep, processed_docs, lock) for line in lines)

        with Pool(os.cpu_count() // 2) as pool:
            for result in tqdm(pool.imap_unordered(process_line_with_lock, args), desc="Processing lines"):
                if result:
                    topicid, query, pos_doc, neg_doc = result
                    stats['kept'] += 1
                    out.write(f"{topicid}\t{query}\t{doc_url(pos_doc)}\t{pos_doc}\t{doc_url(neg_doc)}\t{neg_doc}\n")
                    if stats['kept'] >= triples_to_generate:
                        break

    return stats

outfile = f"resources/datasets/processed/msmarco-data/hard_negatives/msmarco_{subset}_triples"
logfile = f"logs-slurm-msmarco-sft/other-logs/triples_{subset}_generation.log"
ensure_directory_exists(logfile)

triples_to_generate = float('inf')
stats = generate_triples_parallel(outfile, triples_to_generate)

# Save stats to the log file
with open(logfile, 'w', encoding='utf8') as log:
    for key, val in stats.items():
        log.write(f"{key}\t{val}\n")
    log.write("\nInterpretation of Statistics:\n")
    log.write("skipped: Number of entries ignored during triple generation. Skipping occurs if:\n")
    log.write("         - A triple has already been generated for the same topic (topicid) to avoid duplication.\n")
    log.write("         - The document rank does not match the randomly selected rank to ensure dataset diversity.\n\n")
    log.write("kept: Number of triples successfully generated, each containing:\n")
    log.write("      - A query (topicid and its query text).\n")
    log.write("      - A positive document (judged relevant to the query).\n")
    log.write("      - A random negative document (from the top 100 results, not judged relevant).\n\n")
    log.write("docid_collision: Number of times a randomly selected 'unjudged' document was found to be relevant.\n")
    log.write("                 These cases are excluded to ensure the negative example is irrelevant to the query.\n")
print(f"Log file saved at: {logfile}")
