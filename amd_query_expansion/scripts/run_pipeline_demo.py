#!/usr/bin/env python
"""Fully offline demo of the AMD pipeline mechanics: runs the three agents
(Socratic Questioning -> Dialogic Answering -> Reflective Feedback) with a
deterministic mock LLM, then shows a toy retrieval + fusion + evaluation
pass over a handful of hand-written documents. No downloads, GPU, or API
keys required.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amd.eval.metrics import evaluate_run
from amd.llm import MockLLM
from amd.pipeline import AMDPipeline
from amd.retrieval.dense import DenseRetriever
from amd.retrieval.fusion import reciprocal_rank_fusion
from amd.retrieval.sparse import BM25Retriever

TOY_CORPUS = {
    "d1": "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.",
    "d2": "Chlorophyll in plant leaves absorbs sunlight, driving the light-dependent reactions of photosynthesis.",
    "d3": "The Calvin cycle uses ATP and NADPH to fix carbon dioxide into organic molecules during photosynthesis.",
    "d4": "Cellular respiration breaks down glucose to release energy, the reverse process of photosynthesis.",
    "d5": "The stock market saw significant volatility this week amid interest rate uncertainty.",
}
TOY_QUERY = "How does photosynthesis work?"
TOY_QRELS = {"q1": {"d1": 2, "d2": 1, "d3": 1}}


def main() -> None:
    print(f"Query: {TOY_QUERY!r}\n")

    llm = MockLLM()
    pipeline = AMDPipeline(llm, relevance_threshold=0.4, original_query_weight=1)
    result = pipeline.expand(TOY_QUERY)

    print("--- Socratic Questioning Agent: sub-questions ---")
    for sq in result.sub_questions:
        print(f"  [{sq.dimension}] {sq.text}")

    print("\n--- Dialogic Answering Agent: pseudo-answers ---")
    for pa in result.pseudo_answers:
        print(f"  [{pa.dimension}] {pa.text}")

    print("\n--- Reflective Feedback Agent: refined & filtered answers ---")
    for ra in result.refined_answers:
        print(f"  [{ra.dimension}] score={ra.score:.2f} -> {ra.text}")

    print(f"\n--- Final expanded query ---\n{result.expanded_query}\n")

    print("--- Toy retrieval: sparse (BM25) vs. dense (e5) vs. RRF fusion ---")
    bm25 = BM25Retriever(TOY_CORPUS)
    try:
        dense = DenseRetriever(TOY_CORPUS)
        dense_available = True
    except Exception as exc:  # pragma: no cover - depends on optional heavy deps
        print(f"  (dense retriever unavailable in this environment: {exc})")
        dense_available = False

    for label, query_text in [("baseline", TOY_QUERY), ("AMD-expanded", result.expanded_query)]:
        sparse_ranking = bm25.search(query_text, top_k=5)
        run: dict[str, dict[str, float]] = {"q1": dict(sparse_ranking)}
        if dense_available:
            dense_ranking = dense.search(query_text, top_k=5)
            fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking], top_k=5)
            run = {"q1": dict(fused)}

        try:
            metrics = evaluate_run(run, TOY_QRELS, k_values=(3,))
            print(f"  [{label}] top doc ids: {list(run['q1'].keys())} | nDCG@3={metrics['nDCG@3']:.4f}")
        except ImportError as exc:
            print(f"  [{label}] top doc ids: {list(run['q1'].keys())} | (nDCG unavailable: {exc})")


if __name__ == "__main__":
    main()
