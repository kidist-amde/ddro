import gzip
from tqdm import tqdm


def load_qrels(file_path):
    """
    Load qrels (ground truth) from a file.

    :param file_path: Path to qrels file
    :return: Dictionary {query_id: set of relevant doc_ids}
    """
    qrels = {}
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            query_id, _, doc_id, relevance = parts
            if int(relevance) > 0:
                qrels.setdefault(query_id, set()).add(doc_id)
    return qrels


def load_retrievals(file_path):
    """
    Load retrievals (predictions) from a file.

    :param file_path: Path to retrievals file
    :return: Dictionary {query_id: list of predicted doc_ids}
    """
    retrievals = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            query_id, doc_id = parts[:2]
            retrievals.setdefault(query_id, []).append(doc_id)
    return retrievals


def hit_at_k(truth, pred, k):
    """
    Calculate Hit@k metric.

    :param truth: Set of relevant doc_ids
    :param pred: List of predicted doc_ids
    :param k: Rank threshold
    :return: 1.0 if any relevant doc is in top-k predictions, else 0.0
    """
    return 1.0 if any(doc in truth for doc in pred[:k]) else 0.0


def mrr_at_k(truth, pred, k):
    """
    Calculate MRR@k metric.

    :param truth: Set of relevant doc_ids
    :param pred: List of predicted doc_ids
    :param k: Rank threshold
    :return: Reciprocal rank for the first relevant doc in top-k predictions, else 0.0
    """
    for rank, doc in enumerate(pred[:k]):
        if doc in truth:
            return 1.0 / (rank + 1)
    return 0.0


if __name__ == "__main__":
    qrels_file = "resources/datasets/raw/msmarco-data/msmarco-docdev-qrels.tsv.gz"
    retrievals_file = "resources/datasets/processed/msmarco-data/pyserini_data/msmarco_dev_bm25tuned.txt"

    print("Loading qrels...")
    qrels = load_qrels(qrels_file)

    print("Loading retrievals...")
    retrievals = load_retrievals(retrievals_file)

    # Check query coverage
    print(f"Number of queries in qrels: {len(qrels)}")
    print(f"Number of queries in retrievals: {len(retrievals)}")

    matched_queries = set(qrels.keys()) & set(retrievals.keys())
    print(f"Number of matched queries: {len(matched_queries)}")

    print("Evaluating metrics...")
    total_hit_1, total_hit_5, total_hit_10,  = 0, 0, 0
    total_mrr_10 = 0
    total_mrr_100 = 0
    num_queries = len(matched_queries)

    for query_id in tqdm(matched_queries):
        truth = qrels[query_id]
        pred = retrievals[query_id]

        total_hit_1 += hit_at_k(truth, pred, 1)
        total_hit_5 += hit_at_k(truth, pred, 5)
        total_hit_10 += hit_at_k(truth, pred, 10)
        total_mrr_10 += mrr_at_k(truth, pred, 10)
        total_mrr_100 += mrr_at_k(truth, pred, 100)

    avg_hit_1 = total_hit_1 / num_queries
    avg_hit_5 = total_hit_5 / num_queries
    avg_hit_10 = total_hit_10 / num_queries
    avg_mrr_10 = total_mrr_10 / num_queries
    avg_mrr_100 = total_mrr_100 / num_queries

    print("\nFinal Metrics:")
    print(f"Hit@1: {avg_hit_1:.4f}")
    print(f"Hit@5: {avg_hit_5:.4f}")
    print(f"Hit@10: {avg_hit_10:.4f}")
    print(f"MRR@10: {avg_mrr_10:.4f}")
    print(f"MRR@100: {avg_mrr_100:.4f}")
    

