"""Load TREC Deep Learning 2019 / 2020 passage-ranking data via ``ir_datasets``.

The full MS MARCO passage collection has ~8.8M passages; ``max_docs`` lets
you cap the corpus (e.g. for a smoke test) at the cost of no longer matching
the paper's reported numbers, which use the full collection.
"""
from __future__ import annotations

from amd.data.types import IRDataset

_IR_DATASETS_ID = {
    "trec-dl-2019": "msmarco-passage/trec-dl-2019/judged",
    "trec-dl-2020": "msmarco-passage/trec-dl-2020/judged",
}


def load_trec_dl(name: str, max_docs: int | None = None) -> IRDataset:
    """
    Args:
        name: "trec-dl-2019" or "trec-dl-2020".
        max_docs: optional cap on the number of corpus passages loaded
            (useful for local smoke tests; omit for the full collection).
    """
    import ir_datasets

    if name not in _IR_DATASETS_ID:
        raise ValueError(f"Unknown TREC-DL dataset: {name!r}. Expected one of {list(_IR_DATASETS_ID)}")

    dataset = ir_datasets.load(_IR_DATASETS_ID[name])

    queries = {q.query_id: q.text for q in dataset.queries_iter()}

    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    corpus: dict[str, str] = {}
    for i, doc in enumerate(dataset.docs_iter()):
        if max_docs is not None and i >= max_docs:
            break
        corpus[doc.doc_id] = doc.text

    return IRDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)
