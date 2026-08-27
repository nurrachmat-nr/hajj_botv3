"""Shared dataset container used by both BEIR and TREC-DL loaders."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IRDataset:
    name: str
    corpus: dict[str, str]  # doc_id -> text
    queries: dict[str, str]  # query_id -> text
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}
