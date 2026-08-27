"""BM25 sparse retriever, built on ``rank_bm25`` for a lightweight,
dependency-minimal reproduction (no JVM / Pyserini required)."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, corpus: dict[str, str]):
        """
        Args:
            corpus: mapping of doc_id -> document text.
        """
        self.doc_ids = list(corpus.keys())
        tokenized_docs = [_tokenize(corpus[doc_id]) for doc_id in self.doc_ids]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = 1000) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
