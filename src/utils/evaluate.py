import pandas as pd
import numpy as np
from tqdm import tqdm


def average_precision(truth, pred):
    """
        Computes the average precision.
        
        This function computes the average precision at k between two lists of items.

        Parameters
        ----------
        truth: set
                    A set of ground-truth elements (order doesn't matter)
        pred: list
                    A list of predicted elements (order does matter)
        Returns
        -------
        score: double
                    The average precision over the input lists 
    """
    if not truth:
        return 0.0
    
    score, hits_num = 0.0, 0
    for idx, doc in enumerate(pred):
        if doc in truth and doc not in pred[:idx]:
            hits_num += 1.0
            score += hits_num / (idx + 1.0)
    return score / max(1.0, len(truth))


# def recall(truth, pred):
#     if not truth:
#         return 0.0
    
#     score, hits_num = 0.0, 0
#     for idx, doc in enumerate(pred):
#         if doc in truth:
#             return 1.0
#     return 0.0

def recall_at_k(truth, pred, k=None):
    # If no ground truth is provided, 
    if not truth:
        return 0.0
    
    # If k is specified, limit predictions to the top k
    if k is not None:
        pred = pred[:k]
    
    # Calculate the number of relevant items in the predictions
    relevant_in_pred = [doc for doc in pred if doc in truth]
    # # Debugging: Log the number of relevant docs and top k predictions
    # print(f"Truth (Relevant docs): {len(truth)}")
    # print(f"Top-{k} Predictions: {len(pred)}")
    # print(f"Relevant docs found in top {k}: {len(relevant_in_pred)}")
  

    
    # Return the proportion of relevant documents found in the predictions
    # Number of relevant items found / Total number of relevant items in truth
    return len(relevant_in_pred) / len(truth) if truth else 0.0



def hit_at_k(truth, pred, k):
    return 1.0 if any(item in truth for item in pred[:k]) else 0.0



def NDCG(truth, pred, use_graded_scores=False):
    score = 0.0
    for rank, item in enumerate(pred):
        if item in truth:
            if use_graded_scores:
                grade = 1.0 / (truth.index(item) + 1)
            else:
                grade = 1.0
            score += grade / np.log2(rank + 2)
    
    norm = 0.0
    for rank in range(len(truth)):
        if use_graded_scores:
            grade = 1.0 / (rank + 1)
        else:
            grade = 1.0
        norm += grade / np.log2(rank + 2)
    return score / max(0.3, norm)


def metrics(truth, pred, metrics_map):
    """
        Return a numpy array containing metrics specified by metrics_map.
        truth: set
                    A set of ground-truth elements (order doesn't matter)
        pred: list
                    A list of predicted elements (order does matter)
    """
    out = np.zeros((len(metrics_map),), np.float32)

    if "MAP@20" in metrics_map:
        avg_precision = average_precision(truth, pred[:20])
        out[metrics_map.index('MAP@20')] = avg_precision
    
    if "P@1" in metrics_map:
        intersec = len(truth & set(pred[:1]))
        out[metrics_map.index('P@1')] = intersec / max(1., float(len(pred[:1])))

    if "P@10" in metrics_map:
        intersec = len(truth & set(pred[:10]))
        out[metrics_map.index('P@10')] = intersec / max(1., float(len(pred[:10])))
    
    if "P@20" in metrics_map:
        intersec = len(truth & set(pred[:20]))
        out[metrics_map.index('P@20')] = intersec / max(1., float(len(pred[:20])))

    if "P@100" in metrics_map:
        intersec = len(truth & set(pred[:100]))
        out[metrics_map.index('P@100')] = intersec / max(1., float(len(pred[:100])))

    if "R@1" in metrics_map:
        res = recall_at_k(truth, pred[:1])
        out[metrics_map.index('R@1')] = res
    
    if "R@10" in metrics_map:
        res = recall_at_k(truth, pred[:10])
        out[metrics_map.index('R@10')] = res
    
    if "R@100" in metrics_map:
        res = recall_at_k(truth, pred[:100])
        out[metrics_map.index('R@100')] = res

    if "R@1000" in metrics_map:
        res = recall_at_k(truth, pred[:1000])
        out[metrics_map.index('R@1000')] = res
    
    if "MRR" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred):
            if item in truth:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR')] = score
        
    if "MRR@10" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred[:10]):
            if item in truth:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR@10')] = score
   
    if "MRR@100" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred[:100]):
            if item in truth:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR@100')] = score
    
    if "NDCG@10" in metrics_map:
         out[metrics_map.index('NDCG@10')] = NDCG(truth, pred[:10])

    if "NDCG@20" in metrics_map:
         out[metrics_map.index('NDCG@20')] = NDCG(truth, pred[:20])

    if "NDCG@100" in metrics_map:
         out[metrics_map.index('NDCG@100')] = NDCG(truth, pred[:100])

        # Add hit@k metrics
    if "Hit@1" in metrics_map:
        out[metrics_map.index('Hit@1')] = hit_at_k(truth, pred, 1)

    if "Hit@5" in metrics_map:
        out[metrics_map.index('Hit@5')] = hit_at_k(truth, pred, 5)

    if "Hit@10" in metrics_map:
        out[metrics_map.index('Hit@10')] = hit_at_k(truth, pred, 10)

    if "Hit@100" in metrics_map:
        out[metrics_map.index('Hit@100')] = hit_at_k(truth, pred, 100)
    
    return out


class evaluator:
    def __init__(self):
        self.METRICS_MAP = [
            'MRR@10', 'MRR', 'NDCG@10', 'NDCG@20', 'NDCG@100', 'MAP@20', 
            'P@1', 'P@10', 'P@20', 'P@100', 'R@1', 'R@10', 'R@100', 'R@1000',
            'Hit@1', 'Hit@5', 'Hit@10', 'Hit@100'  # Added Hit metrics here
        ]
    
    def evaluate_ranking(self, docid_truth, all_doc_probs, doc_idxs=None, query_ids=None, match_scores=None):
        map_list = []
        mrr_list, mrr_10_list = [], []
        ndcg_10_list, ndcg_20_list, ndcg_100_list = [], [], []
        p_1_list, p_10_list, p_20_list ,p_100_list = [], [], [], []
        r_1_list, r_10_list, r_100_list, r_1000_list = [], [], [], []
        hit_1_list, hit_5_list, hit_10_list, hit_100_list = [], [], [], []  # Added lists for hits

        for docid, probability in tqdm(zip(docid_truth, all_doc_probs)):
            click_doc = set(docid)
            sorted_docs = probability
            metrics_results = metrics(truth=click_doc, pred=sorted_docs, metrics_map=self.METRICS_MAP)

            _mrr10, _mrr, _ndcg10, _ndcg20, _ndcg100, _map20, _p1, _p10, _p20, _p100, _r1, _r10, _r100, _r1000, _hit1, _hit5, _hit10, _hit100 = metrics_results
                
            mrr_10_list.append(_mrr10)
            mrr_list.append(_mrr)

            ndcg_10_list.append(_ndcg10)
            ndcg_20_list.append(_ndcg20)
            ndcg_100_list.append(_ndcg100)
                
            p_1_list.append(_p1)
            p_10_list.append(_p10)
            p_20_list.append(_p20)
            p_100_list.append(_p100)

            r_1_list.append(_r1)
            r_10_list.append(_r10)
            r_100_list.append(_r100)
            r_1000_list.append(_r1000)

            hit_1_list.append(_hit1)
            hit_5_list.append(_hit5)
            hit_10_list.append(_hit10)
            hit_100_list.append(_hit100)

            map_list.append(_map20)

        # Create a dictionary with the results
        results_dict = {
            'MRR@10': np.mean(mrr_10_list), 'MRR': np.mean(mrr_list), 
            'NDCG@10': np.mean(ndcg_10_list), 'NDCG@20': np.mean(ndcg_20_list), 
            'NDCG@100': np.mean(ndcg_100_list), 'MAP@20': np.mean(map_list),
            'P@1': np.mean(p_1_list), 'P@10': np.mean(p_10_list), 
            'P@20': np.mean(p_20_list), 'P@100': np.mean(p_100_list),
            'R@1': np.mean(r_1_list), 'R@10': np.mean(r_10_list), 
            'R@100': np.mean(r_100_list), 'R@1000': np.mean(r_1000_list),
            'Hit@1': np.mean(hit_1_list), 'Hit@5': np.mean(hit_5_list),
            'Hit@10': np.mean(hit_10_list), 'Hit@100': np.mean(hit_100_list)
        }

        # Convert the results dictionary into a pandas DataFrame for better visualization
        results_df = pd.DataFrame([results_dict])
        return results_df