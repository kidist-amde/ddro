import os
import json
import gzip
import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.data.path.append('/ivi/ilps/personal/kmekonn/projects')

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


def recall(truth, pred):
    if not truth:
        return 0.0

    score, hits_num = 0.0, 0
    for idx, doc in enumerate(pred):
        if doc in truth:
            return 1.0
    return 0.0

def bleu(truth, pred, num=1):
    if not truth:
        return 0.0
    reference = [i.split(',') for i in truth]
    candidate = pred[0].split(',')
    if num==1:
        bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        return bleu1
    if num==2:
        bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
        return bleu2


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

    if "P@1" in metrics_map: # 第1个结果是否命中
        intersec = len(truth & set(pred[:1]))
        out[metrics_map.index('P@1')] = intersec / max(1., float(len(pred[:1])))

    if "P@10" in metrics_map: # 前10个返回的结果中有多少个命中的
        intersec = len(truth & set(pred[:10]))
        out[metrics_map.index('P@10')] = intersec / max(1., float(len(pred[:10])))

    if "P@20" in metrics_map: # 前20个返回的结果中有多少个命中的
        intersec = len(truth & set(pred[:20]))
        out[metrics_map.index('P@20')] = intersec / max(1., float(len(pred[:20])))
        # print(intersec, max(1., float(len(pred[:20]))))

    if "P@100" in metrics_map: # 前100个返回的结果中有多少个命中的
        intersec = len(truth & set(pred[:100]))
        out[metrics_map.index('P@100')] = intersec / max(1., float(len(pred[:100])))

    if "R@1" in metrics_map: # 前1个返回的结果是否命中了正确答案
        res = recall(truth, pred[:1])
        out[metrics_map.index('R@1')] = res

    if "R@5" in metrics_map: # 前10个返回的结果是否命中了正确答案
        res = recall(truth, pred[:5])
        out[metrics_map.index('R@5')] = res

    if "R@10" in metrics_map: # 前10个返回的结果是否命中了正确答案
        res = recall(truth, pred[:10])
        out[metrics_map.index('R@10')] = res

    if "R@100" in metrics_map: # 前100个返回的结果是否命中了正确答案
        res = recall(truth, pred[:100])
        out[metrics_map.index('R@100')] = res

    if "R@1000" in metrics_map: # 前1000个返回的结果是否命中了正确答案
        res = recall(truth, pred[:1000])
        out[metrics_map.index('R@1000')] = res

    if "MRR" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred):
            if item in truth:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR')] = score

    if "MRR@5" in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred[:5]):
            if item in truth:
                score = 1.0 / (rank + 1.0)
                break
        out[metrics_map.index('MRR@5')] = score

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

    if "BLEU-1" in metrics_map:
        out[metrics_map.index('BLEU-1')] = bleu(truth, pred[:1], num=1)
    
    if "BLEU-2" in metrics_map:
        out[metrics_map.index('BLEU-2')] = bleu(truth, pred[:1], num=2)

    return out


class evaluator:
    def __init__(self):
        self.METRICS_MAP = ['MRR@5','MRR@10', 'MRR', 'NDCG@10', 'NDCG@20', 'NDCG@100', 'MAP@20', 'P@1', 'P@10', 'P@20', 'P@100', 'R@1', 'R@5', 'R@10', 'R@100', "BLEU-1", "BLEU-2"]

    def evaluate_ranking(self, docid_truth, all_doc_probs, doc_idxs=None, query_ids=None, match_scores=None, operation="generate_doc"):
        mrr_list = []
        mrr_10_list = []
        mrr_5_list = []
        ndcg_10_list = []
        ndcg_20_list = []
        ndcg_100_list = []
        map_list = []
        p_1_list = []
        p_10_list = []
        p_20_list = []
        p_100_list = []
        r_1_list = []
        r_5_list = []
        r_10_list = []
        r_100_list = []
        r_1000_list = []
        bleu_1_list = []
        bleu_2_list = []

        if operation == "sorted_generate_doc":
            for docid, probability in tqdm(zip(docid_truth, all_doc_probs)):
                click_doc = set(docid)
                sorted_docs = probability
                _mrr5, _mrr10, _mrr, _ndcg10, _ndcg20, _ndcg100, _map20, _p1, _p10, _p20, _p100, _r1, _r5, _r10, _r100, bleu_1, bleu_2 = metrics(truth=click_doc, pred=sorted_docs, metrics_map=self.METRICS_MAP)

                mrr_5_list.append(_mrr5)
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
                r_5_list.append(_r5)
                r_10_list.append(_r10)
                r_100_list.append(_r100)

                bleu_1_list.append(bleu_1)
                bleu_2_list.append(bleu_2)

                map_list.append(_map20)

        if operation == "top_generate_doc":
            for docid, probability, doc_idx in tqdm(zip(docid_truth, all_doc_probs, doc_idxs)):
                click_doc = set([docid])
                probs = [[doc_idx[idx], probability[idx]] for idx in range(len(probability))]
                probs = sorted(probs, key=lambda x:x[1], reverse=True)

                sorted_docs = [probs[idx][0] for idx in range(len(probs))]
                _mrr10, _mrr, _ndcg10, _ndcg20, _ndcg100, _map20, _p1, _p10, _p20, _p100, _r1, _r10, _r100, _r1000 = metrics(truth=click_doc, pred=sorted_docs, metrics_map=self.METRICS_MAP)

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

                map_list.append(_map20)

        if operation == "generate_doc":
            for docid, probability in tqdm(zip(docid_truth, all_doc_probs)):
                click_doc = set([docid])
                probs = [[idx, probability[idx]] for idx in range(len(probability))]
                probs = sorted(probs, key=lambda x:x[1], reverse=True)

                sorted_docs = [probs[idx][0] for idx in range(len(probs))]

                _mrr10, _mrr, _ndcg10, _ndcg20, _ndcg100, _map20, _p1, _p10, _p20, _p100, _r1, _r10, _r100, _r1000 = metrics(truth=click_doc, pred=sorted_docs, metrics_map=self.METRICS_MAP)

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

                map_list.append(_map20)

        return [np.mean(mrr_5_list), np.mean(mrr_10_list), np.mean(mrr_list), np.mean(ndcg_10_list), np.mean(ndcg_20_list), np.mean(ndcg_100_list), np.mean(map_list), np.mean(p_1_list), np.mean(p_10_list), np.mean(p_20_list), np.mean(p_100_list), np.mean(r_1_list), np.mean(r_5_list), np.mean(r_10_list), np.mean(r_100_list), np.mean(bleu_1_list), np.mean(bleu_2_list)]