"""AMDPipeline: orchestrates the three agents into a full query expansion."""
from __future__ import annotations

from dataclasses import dataclass, field

from amd.agents.dialogic_answering import DialogicAnsweringAgent, PseudoAnswer
from amd.agents.reflective_feedback import ReflectiveFeedbackAgent, RefinedAnswer
from amd.agents.socratic_questioning import SocraticQuestioningAgent, SubQuestion
from amd.llm import BaseLLM


@dataclass
class AMDResult:
    """Full trace of one query's run through the AMD pipeline."""

    original_query: str
    sub_questions: list[SubQuestion]
    pseudo_answers: list[PseudoAnswer]
    refined_answers: list[RefinedAnswer]
    expanded_query: str = field(default="")


class AMDPipeline:
    """Agent-Mediated Dialogic query expansion pipeline (arXiv:2502.08557).

    Given a raw search query, produces an *expanded query* by running it
    through three cooperating LLM agents:

        query -> [Socratic Questioning] -> 3 sub-questions
               -> [Dialogic Answering]   -> 3 pseudo-answers
               -> [Reflective Feedback]  -> filtered/refined answers
               -> concatenation with the original query -> expanded_query
    """

    def __init__(
        self,
        llm: BaseLLM,
        relevance_threshold: float = 0.4,
        original_query_weight: int = 1,
    ):
        """
        Args:
            llm: shared LLM backend used by all three agents.
            relevance_threshold: minimum Reflective-Feedback score [0,1] a
                pseudo-answer must reach to be kept in the expansion.
            original_query_weight: number of times the original query text
                is repeated in the final expanded query, a common technique
                (also used by Query2Doc) to keep the original query's terms
                from being diluted by the appended expansion text.
        """
        self.socratic_agent = SocraticQuestioningAgent(llm)
        self.answering_agent = DialogicAnsweringAgent(llm)
        self.feedback_agent = ReflectiveFeedbackAgent(llm, relevance_threshold=relevance_threshold)
        self.original_query_weight = max(1, original_query_weight)

    def expand(self, query: str) -> AMDResult:
        sub_questions = self.socratic_agent.run(query)
        pseudo_answers = self.answering_agent.run_all(query, sub_questions)
        refined_answers = self.feedback_agent.run_all(query, pseudo_answers)

        expanded_query = self._compose(query, refined_answers)
        return AMDResult(
            original_query=query,
            sub_questions=sub_questions,
            pseudo_answers=pseudo_answers,
            refined_answers=refined_answers,
            expanded_query=expanded_query,
        )

    def _compose(self, query: str, refined_answers: list[RefinedAnswer]) -> str:
        query_part = " ".join([query] * self.original_query_weight)
        if not refined_answers:
            return query_part
        expansion_part = " ".join(r.text for r in refined_answers)
        return f"{query_part} {expansion_part}".strip()
