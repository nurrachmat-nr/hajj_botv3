import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amd.llm import MockLLM
from amd.pipeline import AMDPipeline

QUERY = "What causes inflation?"


def test_pipeline_expands_query_and_includes_original():
    pipeline = AMDPipeline(MockLLM(), relevance_threshold=0.0, original_query_weight=1)
    result = pipeline.expand(QUERY)

    assert result.original_query == QUERY
    assert QUERY in result.expanded_query
    assert len(result.expanded_query) > len(QUERY)
    assert len(result.sub_questions) == 3
    assert len(result.pseudo_answers) == 3


def test_original_query_weight_repeats_query_text():
    pipeline_single = AMDPipeline(MockLLM(), relevance_threshold=1.01, original_query_weight=1)
    pipeline_triple = AMDPipeline(MockLLM(), relevance_threshold=1.01, original_query_weight=3)

    single = pipeline_single.expand(QUERY).expanded_query
    triple = pipeline_triple.expand(QUERY).expanded_query

    # With an unreachable relevance threshold, no expansion text survives,
    # so the expanded query is just the (possibly repeated) original query.
    assert single == QUERY
    assert triple == " ".join([QUERY] * 3)


def test_high_threshold_falls_back_to_original_query_only():
    pipeline = AMDPipeline(MockLLM(), relevance_threshold=1.01)
    result = pipeline.expand(QUERY)

    assert result.refined_answers == []
    assert result.expanded_query == QUERY
