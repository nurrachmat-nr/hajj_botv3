"""Reciprocal Rank Fusion (RRF) of multiple rankings.

RRF(d) = sum over rankings r that contain d of  1 / (k + rank_r(d))

Fusing on rank rather than raw score sidesteps the score-scale mismatch
between sparse (BM25) and dense (cosine similarity) retrievers, which is
exactly why the paper uses RRF to combine them.
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 1000,
) -> list[tuple[str, float]]:
    """
    Args:
        rankings: one ranked list per retriever, each a list of
            (doc_id, score) pairs already sorted best-first.
        k: RRF damping constant (standard default: 60).
        top_k: number of fused results to return.

    Returns:
        Fused ranking as a list of (doc_id, rrf_score) sorted best-first.
    """
    fused_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]
