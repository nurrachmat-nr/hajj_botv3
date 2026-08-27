"""Retrieval evaluation matching the paper's protocol:
nDCG@10 on BEIR datasets, nDCG@10 and Recall@1000 on TREC-DL 2019/2020.
"""
from __future__ import annotations


def evaluate_run(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k_values: tuple[int, ...] = (10,),
    recall_k_values: tuple[int, ...] = (1000,),
) -> dict[str, float]:
    """
    Args:
        run: query_id -> {doc_id: score} produced by a retriever/fusion.
        qrels: query_id -> {doc_id: relevance} ground truth.
        k_values: cutoffs to report nDCG@k for.
        recall_k_values: cutoffs to report Recall@k for.

    Returns:
        Dict of metric name -> value, averaged over queries present in both
        ``run`` and ``qrels``.
    """
    import pytrec_eval

    measures = {f"ndcg_cut.{k}" for k in k_values} | {f"recall.{k}" for k in recall_k_values}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)

    shared_query_ids = set(run.keys()) & set(qrels.keys())
    scores = evaluator.evaluate({qid: run[qid] for qid in shared_query_ids})

    results: dict[str, float] = {}
    for k in k_values:
        key = f"ndcg_cut_{k}"
        results[f"nDCG@{k}"] = _mean(scores, key)
    for k in recall_k_values:
        key = f"recall_{k}"
        results[f"Recall@{k}"] = _mean(scores, key)
    return results


def _mean(scores: dict[str, dict[str, float]], key: str) -> float:
    values = [q_scores[key] for q_scores in scores.values() if key in q_scores]
    return sum(values) / len(values) if values else 0.0
