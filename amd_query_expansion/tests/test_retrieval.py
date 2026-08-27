import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amd.retrieval.fusion import reciprocal_rank_fusion
from amd.retrieval.sparse import BM25Retriever

CORPUS = {
    "d1": "the cat sat on the mat",
    "d2": "dogs are loyal companions",
    "d3": "cats and dogs can be friends",
    "d4": "the weather today is sunny and warm",
}


def test_bm25_ranks_relevant_document_first():
    retriever = BM25Retriever(CORPUS)
    ranking = retriever.search("cat", top_k=4)

    assert len(ranking) == 4
    top_doc_id, _score = ranking[0]
    assert top_doc_id in {"d1", "d3"}


def test_bm25_scores_are_sorted_descending():
    retriever = BM25Retriever(CORPUS)
    ranking = retriever.search("dogs friends", top_k=4)
    scores = [score for _doc_id, score in ranking]
    assert scores == sorted(scores, reverse=True)


def test_reciprocal_rank_fusion_rewards_consensus_docs():
    ranking_a = [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)]
    ranking_b = [("d2", 5.0), ("d3", 4.0), ("d1", 1.0)]

    fused = reciprocal_rank_fusion([ranking_a, ranking_b], k=60, top_k=3)
    fused_ids = [doc_id for doc_id, _score in fused]

    # d2 (rank 2 then rank 1) and d3 (rank 3 then rank 2) beat d1
    # (rank 1 then rank 3) once both rankings are fused.
    assert fused_ids[0] in {"d2", "d3"}
    assert set(fused_ids) == {"d1", "d2", "d3"}


def test_reciprocal_rank_fusion_respects_top_k():
    ranking = [(f"d{i}", float(10 - i)) for i in range(10)]
    fused = reciprocal_rank_fusion([ranking], top_k=3)
    assert len(fused) == 3
