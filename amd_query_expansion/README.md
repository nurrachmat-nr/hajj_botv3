# AMD: Agent-Mediated Dialogic Query Expansion

A from-scratch reproduction of:

> Wonduk Seo, Hyunjin An, Seunghyun Lee. *A New Query Expansion Approach via
> Agent-Mediated Dialogic Inquiry.* arXiv:2502.08557 (ACM SIGKDD 2025 Workshop
> on AI Agent for Information Retrieval, Agent4IR). Also published as
> *A New Query Expansion Approach for Enhancing Information Retrieval via
> Agent-Mediated Dialogic Inquiry*, WSDM'26.

## Method

Classic LLM query-expansion methods (Query2Doc, HyDE-style single-prompt
pseudo-document generation, etc.) ask one model call to expand a query and
tend to produce **homogeneous, narrow** expansions. AMD instead runs a small
**multi-agent dialogue** so the expansion covers several distinct facets of
the query and filters out noise before it reaches the retriever:

1. **Socratic Questioning Agent** — reformulates the original query into
   **three sub-questions**, one per Socratic questioning dimension:
   - *Clarification* — "What exactly is being asked / what does this term mean?"
   - *Assumption probing* — "What is being taken for granted by this query?"
   - *Implication probing* — "What follows from / what are the consequences
     or downstream effects of the topic?"
2. **Dialogic Answering Agent** — answers each sub-question independently
   with a short pseudo-answer, so the three answers surface different
   perspectives on the query rather than one narrow expansion.
3. **Reflective Feedback Agent** — scores each pseudo-answer for relevance
   and faithfulness to the original query's intent, rewrites/condenses it to
   keep only the informative content, and drops answers that fall below a
   relevance threshold.

The surviving refined answers are concatenated with the original query to
form the **expanded query**, which is then run through retrieval:

- **Sparse retrieval**: BM25 over the expanded query.
- **Dense retrieval**: bi-encoder embeddings (`multilingual-e5-base` by
  default, matching the paper) over the expanded query.
- **Fusion**: **Reciprocal Rank Fusion (RRF)** of the sparse and dense
  rankings, which sidesteps score-scale mismatches by fusing on rank alone.

Evaluation follows the paper: **nDCG@10** on BEIR-style datasets and
**nDCG@10 / Recall@1000** on TREC-DL 2019/2020, using `pytrec_eval`.

## Repo layout

```
amd_query_expansion/
├── amd/
│   ├── llm.py                  # LLM backends (HF transformers, OpenAI-compatible, mock)
│   ├── agents/
│   │   ├── base.py             # Agent base class + prompt templates
│   │   ├── socratic_questioning.py
│   │   ├── dialogic_answering.py
│   │   └── reflective_feedback.py
│   ├── pipeline.py             # AMDPipeline: orchestrates the 3 agents -> expanded query
│   ├── retrieval/
│   │   ├── sparse.py           # BM25
│   │   ├── dense.py            # e5 bi-encoder + cosine search
│   │   └── fusion.py           # Reciprocal Rank Fusion
│   ├── eval/
│   │   └── metrics.py          # nDCG@10, Recall@1000 via pytrec_eval
│   └── data/
│       ├── beir_loader.py      # BEIR corpus/queries/qrels loading
│       └── trec_dl_loader.py   # TREC DL 2019/2020 via ir_datasets
├── configs/default.yaml
├── scripts/
│   ├── run_pipeline_demo.py    # offline demo with the mock LLM (no downloads)
│   └── run_experiment.py       # full BEIR/TREC-DL experiment + evaluation table
└── tests/                      # unit tests, all run offline with the mock LLM
```

## Installation

```bash
cd amd_query_expansion
pip install -r requirements.txt
```

Real experiments additionally need network access to download datasets
(`beir`, `ir_datasets`) and model weights (`Qwen/Qwen2.5-7B-Instruct`,
`intfloat/multilingual-e5-base`) from the Hugging Face Hub.

## Quick, fully-offline demo

Runs the full 3-agent pipeline with a deterministic mock LLM (no GPU, no
downloads, no API keys) so you can see the mechanics end-to-end:

```bash
python scripts/run_pipeline_demo.py
```

## Reproducing the paper's experiments

```bash
python scripts/run_experiment.py \
    --dataset scifact \
    --llm-backend hf --llm-model Qwen/Qwen2.5-7B-Instruct \
    --dense-model intfloat/multilingual-e5-base \
    --fusion rrf --top-k 10
```

`--dataset` accepts any BEIR dataset name (`trec-covid`, `nfcorpus`, `fiqa`,
`scifact`, `scidocs`, `webis-touche2020`, ...) or `trec-dl-2019` /
`trec-dl-2020` for the TREC Deep Learning passage tracks. `--llm-backend`
also accepts `openai` (any OpenAI-compatible chat-completions endpoint, set
`OPENAI_API_KEY`/`OPENAI_BASE_URL`) or `mock` for offline smoke testing.

## Notes / honest limitations

- The paper does not release the exact agent prompts or the reflective
  scoring rubric; the prompts here are a faithful reconstruction from the
  paper's description of the three roles and the three Socratic dimensions
  it names (clarification, assumption probing, implication probing).
- Default generation model (`Qwen2.5-7B-Instruct`) and dense retriever
  (`multilingual-e5-base`) match those reported in the paper's experiments.
- BM25 here uses `rank_bm25` for a dependency-light, pure-Python sparse
  retriever; swap in Pyserini/Lucene BM25 for exact score parity with
  IR-standard tooling if needed.
