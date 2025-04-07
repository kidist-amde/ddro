import csv
import gzip
import os
import random
import logging
from collections import defaultdict
from typing import Dict, List
from multiprocessing import Pool, cpu_count
from functools import lru_cache
from itertools import islice
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)

# Logging setup
def setup_logging(logfile: str):
    logging.basicConfig(
        filename=logfile,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info("Logging setup complete.")

def ensure_directory_exists(filepath: str):
    """Ensures the directory for the given file path exists."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def load_querystring(queries_file: str) -> Dict[str, str]:
    """Load query strings from a gzipped file."""
    querystring = {}
    with gzip.open(queries_file, 'rt', encoding='utf8') as f:
        print(f"Loading queries from {queries_file}")
        tsvreader = csv.reader(f, delimiter="\t")
        for topicid, querystring_of_topicid in tsvreader:
            querystring[topicid] = querystring_of_topicid
    return querystring

def load_docoffset(doc_offsets_file: str) -> Dict[str, int]:
    """Load document offsets from a gzipped file."""
    docoffset = {}
    with gzip.open(doc_offsets_file, 'rt', encoding='utf8') as f:
        print(f"Loading document offsets from {doc_offsets_file}")
        tsvreader = csv.reader(f, delimiter="\t")
        for docid, _, offset in tsvreader:
            docoffset[docid] = int(offset)
    return docoffset

def load_qrel(qrels_file: str) -> Dict[str, List[str]]:
    """Load Qrels data from a gzipped file."""
    qrel = defaultdict(list)
    with gzip.open(qrels_file, 'rt', encoding='utf8') as f:
        print(f"Loading Qrels from {qrels_file}")
        tsvreader = csv.reader(f, delimiter=" ")
        for topicid, _, docid, rel in tsvreader:
            if rel == "1":
                qrel[topicid].append(docid)
    return qrel

def get_cached_content(docid: str, docs_f) -> str:
    """Retrieve cached content for a given docid."""
    docs_f.seek(docoffset[docid])  # This assumes `docoffset` is accessible
    line = docs_f.readline()
    if not line.startswith(docid + "\t"):
        raise ValueError(f"Mismatch: Expected {docid}, but found {line}")
    return line.rstrip()


def process_query(args):
    topicid, groups, querystring, qrel, docoffset, docs_file, group_sizes, n_negatives = args
    stats = defaultdict(int)
    triples = []

    if topicid not in querystring or topicid not in qrel:
        stats['skipped_queries'] += 1
        return triples, stats

    positive_docid = random.choice(qrel[topicid])
    if positive_docid not in docoffset:
        stats['missing_positive_doc'] += 1
        return triples, stats

    try:
        pos_offset = docoffset[positive_docid]
        pos_content = get_cached_content(positive_docid, docs_file)  # Simplified call
        pos_url = pos_content.split("\t")[1]
    except KeyError:
        stats['missing_positive_doc'] += 1
        return triples, stats
    negatives = []
    for group, size in zip(['top', 'mid', 'lower'], group_sizes):
        candidates = [docid for docid in groups[group] if docid not in qrel[topicid]]
        negatives.extend(random.sample(candidates, min(size, len(candidates))))

    if len(negatives) < n_negatives:
        stats['not_enough_negatives'] += 1
        return triples, stats
    for neg_docid in negatives:
        if neg_docid not in docoffset:
            stats['missing_negative_doc'] += 1
            continue

        try:
            neg_content = get_cached_content(neg_docid, docs_file)
            neg_url = neg_content.split("\t")[1]
            triples.append(f"{topicid}\t{positive_docid}\t{neg_docid}\n")

            stats['triples_generated'] += 1
        except KeyError:
            stats['missing_negative_doc'] += 1

    return triples, stats


def generate_triples_parallel(outfile: str, querystring, qrel, docoffset, pyserini_file, docs_file, n_negatives=32):
    stats = defaultdict(int)
    group_sizes = [n_negatives // 3] * 3
    group_sizes[-1] += n_negatives % 3
    logging.info(f"Group sizes: {group_sizes}")
    grouped_docs = defaultdict(lambda: {'top': [], 'mid': [], 'lower': []})
    with open(pyserini_file, 'rt', encoding='utf8') as pyserini_f:
        for line in pyserini_f:
            topicid, docid, rank = line.strip().split("\t")
            rank = int(rank)
            if 1 <= rank <= 100:
                grouped_docs[topicid]['top'].append(docid)
            elif 101 <= rank <= 500:
                grouped_docs[topicid]['mid'].append(docid)
            elif 501 <= rank <= 1000:
                grouped_docs[topicid]['lower'].append(docid)
    
    docs_file = open(docs_file, 'rt', encoding='utf8')

    tasks = [
        (topicid, groups, querystring, qrel, docoffset, docs_file, group_sizes, n_negatives)
        for topicid, groups in grouped_docs.items()
    ]
    
    print(f"Processing {len(tasks)} queries...")
    with open(outfile, 'w', encoding='utf8') as out:
        for args in tqdm(tasks, desc="Processing queries", total=len(tasks)):
            triples, result_stats = process_query(args)
            out.writelines(triples)
            for key, val in result_stats.items():
                stats[key] += val
    docs_file.close()
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate triples with sampled negatives.")
    parser.add_argument("--queries_file", required=True, help="Path to query strings file (gzipped).")
    parser.add_argument("--qrels_file", required=True, help="Path to Qrels file (gzipped).")
    parser.add_argument("--pyserini_file", required=True, help="Path to Pyserini retrieval file.")
    parser.add_argument("--docs_file", required=True, help="Path to document content file (gzipped).")
    parser.add_argument("--doc_offsets_file", required=True, help="Path to document offsets file.")
    parser.add_argument("--outfile", required=True, help="Path to save generated triples.")
    parser.add_argument("--logfile", required=True, help="Path to save log file.")
    parser.add_argument("--n_negatives", type=int, default=32, help="Number of negatives to sample per query.")
    args = parser.parse_args()

    setup_logging(args.logfile)
    ensure_directory_exists(args.outfile)

    querystring = load_querystring(args.queries_file)
    docoffset = load_docoffset(args.doc_offsets_file)
    qrel = load_qrel(args.qrels_file)

    stats = generate_triples_parallel(
        outfile=args.outfile,
        querystring=querystring,
        qrel=qrel,
        docoffset=docoffset,
        pyserini_file=args.pyserini_file,
        docs_file=args.docs_file,
        n_negatives=args.n_negatives
    )

    for key, val in stats.items():
        print(f"{key}\t{val}")
        logging.info(f"{key}: {val}")
