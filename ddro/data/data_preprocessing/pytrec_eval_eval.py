import gzip
import pytrec_eval

# Paths to files
qrels_file = "resources/datasets/raw/msmarco-data/msmarco-docdev-qrels.tsv.gz"
run_file = "resources/datasets/processed/msmarco-data/pyserini_data/msmarco_dev_bm25tuned.txt"

def load_qrels(file_path):
    """Load qrels into a dictionary, handling the extra column."""
    qrels = {}
    with gzip.open(file_path, 'rt') as f:  # Handles gzipped files
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:  # Ensure there are enough columns
                query_id, _, doc_id, relevance = parts  # Skip the second column
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = int(relevance)
    return qrels


def load_run(file_path):
    """Load retrieval results into a dictionary."""
    runs = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:  # Handle files with only <query_id>, <doc_id>, <rank>
                query_id, doc_id, rank = parts
                score = 1.0 / int(rank)  # Assign a dummy score inversely proportional to rank
            elif len(parts) == 6:  # Handle full-format files
                query_id, _, doc_id, rank, score, _ = parts
                score = float(score)
            else:
                continue  # Skip malformed lines

            if query_id not in runs:
                runs[query_id] = {}
            runs[query_id][doc_id] = score
    return runs


def compute_hit_and_mrr(qrels, runs, hits=[1, 5, 10]):
    """Compute Hit@K and MRR@10 metrics."""
    metrics = {f'hit@{k}': 0 for k in hits}
    metrics['mrr@10'] = 0
    total_queries = 0

    for query_id, relevant_docs in qrels.items():
        if query_id not in runs:
            continue
        retrieved_docs = list(runs[query_id].keys())[:max(hits)]  # Top K documents only

        # Compute Hit@K
        for k in hits:
            if any(doc in relevant_docs for doc in retrieved_docs[:k]):
                metrics[f'hit@{k}'] += 1

        # Compute MRR@10
        for rank, doc_id in enumerate(retrieved_docs[:10], start=1):
            if doc_id in relevant_docs:
                metrics['mrr@10'] += 1 / rank
                break

        total_queries += 1

    # Normalize metrics
    for key in metrics:
        metrics[key] /= total_queries

    return metrics

# Load qrels and run files
qrels = load_qrels(qrels_file)
runs = load_run(run_file)

# Compute metrics
metrics = compute_hit_and_mrr(qrels, runs)
print("\nMetrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
