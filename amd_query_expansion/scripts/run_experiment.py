#!/usr/bin/env python
"""Reproduce the paper's evaluation protocol on a BEIR or TREC-DL dataset.

Example:
    python scripts/run_experiment.py --dataset scifact --llm-backend mock \
        --fusion rrf --top-k 10 --limit-queries 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from amd.data.beir_loader import load_beir_dataset
from amd.data.trec_dl_loader import load_trec_dl
from amd.eval.metrics import evaluate_run
from amd.llm import build_llm
from amd.pipeline import AMDPipeline
from amd.retrieval.dense import DenseRetriever
from amd.retrieval.fusion import reciprocal_rank_fusion
from amd.retrieval.sparse import BM25Retriever

TREC_DL_NAMES = {"trec-dl-2019", "trec-dl-2020"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="BEIR dataset name, or trec-dl-2019 / trec-dl-2020")
    parser.add_argument("--data-dir", default="./beir_datasets")
    parser.add_argument("--max-docs", type=int, default=None, help="cap corpus size (TREC-DL only)")

    parser.add_argument("--llm-backend", default="mock", choices=["hf", "openai", "mock"])
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dense-model", default="intfloat/multilingual-e5-base")

    parser.add_argument("--relevance-threshold", type=float, default=0.4)
    parser.add_argument("--original-query-weight", type=int, default=1)

    parser.add_argument("--fusion", default="rrf", choices=["sparse", "dense", "rrf"])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=10, help="nDCG@k cutoff to report")
    parser.add_argument("--depth", type=int, default=1000, help="how many docs each retriever returns before fusion")

    parser.add_argument("--limit-queries", type=int, default=None, help="evaluate only the first N queries (smoke test)")
    parser.add_argument("--no-baseline", action="store_true", help="skip the unexpanded-query baseline run")
    return parser.parse_args()


def load_dataset(args: argparse.Namespace):
    if args.dataset in TREC_DL_NAMES:
        return load_trec_dl(args.dataset, max_docs=args.max_docs)
    return load_beir_dataset(args.dataset, data_dir=args.data_dir)


def build_retrievers(args: argparse.Namespace, corpus: dict[str, str]):
    need_sparse = args.fusion in {"sparse", "rrf"}
    need_dense = args.fusion in {"dense", "rrf"}

    sparse_retriever = BM25Retriever(corpus) if need_sparse else None
    dense_retriever = DenseRetriever(corpus, model_name=args.dense_model) if need_dense else None
    return sparse_retriever, dense_retriever


def retrieve(args: argparse.Namespace, query_text: str, sparse_retriever, dense_retriever) -> dict[str, float]:
    sparse_ranking = sparse_retriever.search(query_text, top_k=args.depth) if sparse_retriever else None
    dense_ranking = dense_retriever.search(query_text, top_k=args.depth) if dense_retriever else None

    if args.fusion == "sparse":
        ranking = sparse_ranking
    elif args.fusion == "dense":
        ranking = dense_ranking
    else:
        ranking = reciprocal_rank_fusion([sparse_ranking, dense_ranking], k=args.rrf_k, top_k=args.depth)

    return {doc_id: score for doc_id, score in ranking}


def main() -> None:
    args = parse_args()

    print(f"Loading dataset {args.dataset!r} ...")
    dataset = load_dataset(args)
    print(f"  corpus={len(dataset.corpus)} queries={len(dataset.queries)} qrels={len(dataset.qrels)}")

    query_items = list(dataset.queries.items())
    if args.limit_queries:
        query_items = query_items[: args.limit_queries]

    print(f"Building retrievers (fusion={args.fusion}) ...")
    sparse_retriever, dense_retriever = build_retrievers(args, dataset.corpus)

    print(f"Loading LLM backend={args.llm_backend} ...")
    llm_kwargs = {"model_name": args.llm_model} if args.llm_backend != "mock" else {}
    llm = build_llm(args.llm_backend, **llm_kwargs)
    pipeline = AMDPipeline(
        llm,
        relevance_threshold=args.relevance_threshold,
        original_query_weight=args.original_query_weight,
    )

    amd_run: dict[str, dict[str, float]] = {}
    baseline_run: dict[str, dict[str, float]] = {}

    for query_id, query_text in tqdm(query_items, desc="AMD query expansion + retrieval"):
        if not args.no_baseline:
            baseline_run[query_id] = retrieve(args, query_text, sparse_retriever, dense_retriever)

        result = pipeline.expand(query_text)
        amd_run[query_id] = retrieve(args, result.expanded_query, sparse_retriever, dense_retriever)

    recall_k = (1000,) if args.dataset in TREC_DL_NAMES else ()
    amd_metrics = evaluate_run(amd_run, dataset.qrels, k_values=(args.top_k,), recall_k_values=recall_k)

    print("\n=== AMD (Agent-Mediated Dialogic) expanded query ===")
    for name, value in amd_metrics.items():
        print(f"  {name}: {value:.4f}")

    if not args.no_baseline:
        baseline_metrics = evaluate_run(baseline_run, dataset.qrels, k_values=(args.top_k,), recall_k_values=recall_k)
        print("\n=== Baseline (original query, no expansion) ===")
        for name, value in baseline_metrics.items():
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
