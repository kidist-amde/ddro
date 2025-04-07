
import os
import gzip
import csv
import json
import argparse
import random
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm.auto import tqdm

random.seed(42)
logging.basicConfig(level=logging.INFO)


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_queries(queries_file: str, gzipped=True) -> Dict[str, str]:
    queries = {}
    opener = gzip.open if gzipped else open
    with opener(queries_file, 'rt', encoding='utf8') as f:
        reader = csv.reader(f, delimiter="\t")
        for qid, query in reader:
            queries[qid] = query
    return queries


def load_qrels(qrels_file: str, gzipped=True) -> Dict[str, str]:
    qrels = {}
    opener = gzip.open if gzipped else open
    with opener(qrels_file, 'rt') as f:
        for line in f:
            qid, _, docid, _ = line.strip().split()
            qrels[qid] = docid
    return qrels


def load_bm25_rankings(rank_file: str) -> Dict[str, List[Tuple[str, float]]]:
    rankings = defaultdict(list)
    with open(rank_file, 'r') as f:
        for line in f:
            qid, docid, score = line.strip().split()
            rankings[qid].append((docid, float(score)))
    return rankings


def sample_negatives(rankings: Dict[str, List[Tuple[str, float]]],
                      qrels: Dict[str, str],
                      num_neg: int) -> List[Tuple[str, str, str]]:
    triples = []
    for qid, pos_doc in qrels.items():
        candidates = [docid for docid, _ in rankings.get(qid, []) if docid != pos_doc]
        if len(candidates) < num_neg:
            continue
        negatives = random.sample(candidates, num_neg)
        for neg_doc in negatives:
            triples.append((qid, pos_doc, neg_doc))
    return triples


def write_triples(triples: List[Tuple[str, str, str]],
                  queries: Dict[str, str],
                  docs: Dict[str, str],
                  output_path: str):
    with open(output_path, 'w') as fout:
        for qid, pos, neg in triples:
            if pos in docs and neg in docs and qid in queries:
                fout.write(f"{qid}\t{queries[qid]}\t{docs[pos]}\t{pos}\t{docs[neg]}\t{neg}\n")


def load_doc_metadata(docs_file: str) -> Dict[str, str]:
    docs = {}
    with open(docs_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=[
            'query', 'id', 'long_answer', 'short_answer', 'title',
            'abstract', 'content', 'document_url', 'doc_tac', 'language'
        ])
        for row in tqdm(reader, desc="Reading document metadata"):
            doc_id = row.get('id', '').strip()
            url = row.get('document_url', '').strip()
            if doc_id and url:
                docs[doc_id] = url
    return docs


def main():
    parser = argparse.ArgumentParser(description="Unified BM25-based hard negative sampler")
    parser.add_argument('--dataset', choices=['msmarco', 'nq'], required=True)
    parser.add_argument('--rank_file', required=True)
    parser.add_argument('--qrels_file', required=True)
    parser.add_argument('--queries_file', required=True)
    parser.add_argument('--docs_file', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--num_negative_per_query', type=int, default=3)
    args = parser.parse_args()

    ensure_dir(args.output_path)

    queries = load_queries(args.queries_file)
    qrels = load_qrels(args.qrels_file)
    rankings = load_bm25_rankings(args.rank_file)
    triples = sample_negatives(rankings, qrels, args.num_negative_per_query)
    docs = load_doc_metadata(args.docs_file)
    write_triples(triples, queries, docs, args.output_path)
    print(f"Generated {len(triples)} triples → {args.output_path}")


if __name__ == '__main__':
    main()
