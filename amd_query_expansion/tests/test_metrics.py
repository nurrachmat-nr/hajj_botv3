import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pytrec_eval")

from amd.eval.metrics import evaluate_run


def test_perfect_ranking_gets_ndcg_one():
    qrels = {"q1": {"d1": 1, "d2": 0}}
    run = {"q1": {"d1": 2.0, "d2": 1.0}}

    metrics = evaluate_run(run, qrels, k_values=(2,))
    assert metrics["nDCG@2"] == 1.0


def test_inverted_ranking_scores_lower_than_perfect():
    qrels = {"q1": {"d1": 1, "d2": 0}}
    perfect_run = {"q1": {"d1": 2.0, "d2": 1.0}}
    inverted_run = {"q1": {"d1": 1.0, "d2": 2.0}}

    perfect = evaluate_run(perfect_run, qrels, k_values=(2,))
    inverted = evaluate_run(inverted_run, qrels, k_values=(2,))
    assert inverted["nDCG@2"] < perfect["nDCG@2"]


def test_recall_at_k_counts_relevant_docs_retrieved():
    qrels = {"q1": {"d1": 1, "d2": 1, "d3": 0}}
    run = {"q1": {"d1": 3.0, "d3": 2.0, "d2": 1.0}}

    metrics = evaluate_run(run, qrels, k_values=(10,), recall_k_values=(1,))
    assert metrics["Recall@1"] == 0.5  # only d1 (rank 1) is relevant within top-1
