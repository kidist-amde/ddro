import gzip
import argparse
import numpy as np
import json
from tqdm.auto import tqdm

def main():
    parser = argparse.ArgumentParser(description='Generate NQ Triples with Hard Negatives')
    parser.add_argument('--relevance_path', required=True, type=str, help='Input file path')
    parser.add_argument('--qrel_path', type=str, required=True, help='Qrels file path')
    parser.add_argument('--output_path', type=str, required=True, help='Output file path')
    parser.add_argument('--num_negative_per_query', required=True, type=int, default=1, help='Number of negative samples per query')
    parser.add_argument('--query_path', type=str, required=True, help='Query file path')
    parser.add_argument('--docs_path', type=str, required=True, help='Docs file path')

    args = parser.parse_args()

    # Load data
    triples = create_nq_triple_dataset(args.relevance_path, args.qrel_path, args.num_negative_per_query)
    queries = load_queries(args.query_path)
    docs_urls = load_docs_urls(args.docs_path)

    # Write output
    with open(args.output_path, 'w') as f:
        for query_id, pos_doc_id, neg_doc_id in triples:
            # Check if pos_doc_id exists in docs_urls
            if pos_doc_id not in docs_urls:
                print(f"Missing pos_doc_id: {pos_doc_id} for query_id: {query_id}")
                continue  # Skip this triple if pos_doc_id is missing

            # Check if neg_doc_id exists in docs_urls
            if neg_doc_id not in docs_urls:
                print(f"Missing neg_doc_id: {neg_doc_id} for query_id: {query_id}")
                continue  # Skip this triple if neg_doc_id is missing

            # Write the valid triples to the output file
            f.write(f"{query_id}\t{queries[query_id]}\t{docs_urls[pos_doc_id]}\t{pos_doc_id}\t{docs_urls[neg_doc_id]}\t{neg_doc_id}\n")


def create_nq_triple_dataset(relevance_path, qrel_path, num_negative_per_query):
    qrels = load_qrel(qrel_path)
    relevance = {}

    # Load BM25 results (relevance scores)
    with open(relevance_path, 'r') as f:
        for line in f:
            query_id, doc_id, score = line.strip().split()
            query_id = str(query_id)  # Ensure query_id is a string
            doc_id = str(doc_id)  # Ensure doc_id is a string

            if query_id not in relevance:
                relevance[query_id] = []
            # Store each doc with its score
            relevance[query_id].append((doc_id, float(score)))

    triples = []

    for query_id in qrels:
        pos_doc_id = qrels[query_id]
        # Get the BM25-retrieved documents for the query, sorted by score (descending)
        retrieved_docs = sorted(relevance.get(query_id, []), key=lambda x: x[1], reverse=True)

        # Separate positives and negatives
        negatives = [doc_id for doc_id, score in retrieved_docs if doc_id != pos_doc_id]

        # If no negatives are available, skip this query
        if len(negatives) == 0:
            continue

        # Select the top-ranked irrelevant document(s) as hard negatives
        hard_negatives = negatives[:num_negative_per_query]  # top `num_negative_per_query` as hard negatives

        for neg_doc_id in hard_negatives:
            triples.append((query_id, pos_doc_id, neg_doc_id))

    return triples


def load_qrel(qrel_path):
    qrel = {}
    with gzip.open(qrel_path, 'rt') as f:
        for line in f:
            query_id, _, doc_id, _ = line.strip().split()
            query_id = str(query_id)  # Ensure query_id is a string
            doc_id = str(doc_id)  # Ensure doc_id is a string
            qrel[query_id] = doc_id
    return qrel


def load_queries(queries_path):
    queries = {}
    with gzip.open(queries_path, 'rt') as f:
        for line in f:
            query_id, query = line.strip().split('\t')
            query_id = str(query_id)  # Ensure query_id is a string
            queries[query_id] = query
    return queries


def load_docs_urls(docs_path):
    docs = {}
    with open(docs_path, 'r') as f:
        for line in tqdm(f, total=109739):
            doc = json.loads(line)
            doc_id = str(doc['id'])  
            docs[doc_id] = doc['url']
    return docs


# Debugging function to search for specific document in the docs file
def find_doc_by_id(docs_path, doc_id):
    with open(docs_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            if doc['id'] == str(doc_id):
                return doc
    return None


if __name__ == '__main__':
    main()
