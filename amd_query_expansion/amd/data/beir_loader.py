"""Load a BEIR benchmark dataset (corpus, queries, qrels).

Requires the ``beir`` package, which downloads and caches the dataset zip
from the official BEIR host on first use.
"""
from __future__ import annotations

import os

from amd.data.types import IRDataset

BEIR_URL_TEMPLATE = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)


def load_beir_dataset(name: str, data_dir: str = "./beir_datasets", split: str = "test") -> IRDataset:
    """
    Args:
        name: BEIR dataset name, e.g. "scifact", "nfcorpus", "trec-covid",
            "fiqa", "scidocs", "webis-touche2020".
        data_dir: local directory to download/cache the dataset into.
        split: which qrels split to evaluate on ("test" for most BEIR
            datasets; "dev" for a few, e.g. "msmarco").
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    os.makedirs(data_dir, exist_ok=True)
    url = BEIR_URL_TEMPLATE.format(name=name)
    dataset_path = util.download_and_unzip(url, data_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split=split)
    corpus_text = {doc_id: f"{doc.get('title', '')} {doc.get('text', '')}".strip() for doc_id, doc in corpus.items()}

    return IRDataset(name=name, corpus=corpus_text, queries=queries, qrels=qrels)
