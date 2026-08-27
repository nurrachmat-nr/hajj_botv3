"""Dense bi-encoder retriever, defaulting to ``intfloat/multilingual-e5-base``
(the dense retriever used in the paper's experiments).

E5 models require ``"query: "`` / ``"passage: "`` prefixes on inputs and
L2-normalized embeddings for cosine similarity -- both are handled here.
"""
from __future__ import annotations

import numpy as np


class DenseRetriever:
    def __init__(
        self,
        corpus: dict[str, str],
        model_name: str = "intfloat/multilingual-e5-base",
        batch_size: int = 64,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.doc_ids = list(corpus.keys())

        passages = [f"passage: {corpus[doc_id]}" for doc_id in self.doc_ids]
        embeddings = self.model.encode(
            passages,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.doc_embeddings = embeddings.astype(np.float32)

    def search(self, query: str, top_k: int = 1000) -> list[tuple[str, float]]:
        query_embedding = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        scores = self.doc_embeddings @ query_embedding
        top_k = min(top_k, len(self.doc_ids))
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]
