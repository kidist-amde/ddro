import gzip
import argparse
import json
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser(description='Generate NQ Triples with Hard Negatives')
    parser.add_argument('--relevance_path', type=str, required=True, help='Path to BM25 relevance file')
    parser.add_argument('--qrel_path', type=str, required=True, help='Path to qrels file (gzipped)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output triples file')
    parser.add_argument('--num_negative_per_query', type=int, required=True, help='Number of negatives per query')
    parser.add_argument('--query_path', type=str, required=True, help='Path to query file (gzipped)')
    parser.add_argument('--docs_path', type=str, required=True, help='Path to JSONL file with document URLs')
    args = parser.parse_args()

    triples = generate_triples(
        args.relevance_path,
        args.qrel_path,
        args.num_negative_per_query
    )
    queries = load_queries(args.query_path)
    doc_urls = load_doc_urls(args.docs_path)

    with open(args.output_path, 'w') as fout:
        for query_id, pos_id, neg_id in triples:
            if pos_id not in doc_urls or neg_id not in doc_urls:
                continue
            fout.write(
                f"{query_id}\t{queries[query_id]}\t{doc_urls[pos_id]}\t{pos_id}\t{doc_urls[neg_id]}\t{neg_id}\n"
            )


def generate_triples(relevance_path, qrel_path, num_negatives):
    qrels = load_qrels(qrel_path)
    relevance = {}

    with open(relevance_path, 'r') as f:
        for line in f:
            qid, docid, score = line.strip().split()
            relevance.setdefault(qid, []).append((docid, float(score)))

    triples = []
    for qid, pos_id in qrels.items():
        ranked = sorted(relevance.get(qid, []), key=lambda x: x[1], reverse=True)
        negatives = [docid for docid, _ in ranked if docid != pos_id]
        for neg_id in negatives[:num_negatives]:
            triples.append((qid, pos_id, neg_id))
    return triples


def load_qrels(path):
    qrels = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            qid, _, docid, _ = line.strip().split()
            qrels[qid] = docid
    return qrels


def load_queries(path):
    queries = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            qid, query = line.strip().split('\t')
            queries[qid] = query
    return queries


def load_doc_urls(path):
    doc_urls = {}
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Loading document URLs"):
            doc = json.loads(line)
            doc_urls[str(doc['id'])] = doc['url']
    return doc_urls


if __name__ == '__main__':
    main()
