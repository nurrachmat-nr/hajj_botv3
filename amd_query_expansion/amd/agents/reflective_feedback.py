"""Reflective Feedback Agent.

Evaluates each pseudo-answer for relevance/faithfulness to the original
query's intent, condenses it to its informative content, and drops answers
that fall below a relevance threshold -- so only the most useful expansion
content is kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from amd.agents.base import Agent
from amd.agents.dialogic_answering import PseudoAnswer

_SCORE_RE = re.compile(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_REFINED_RE = re.compile(r"refined\s*[:=]\s*(.*)", re.IGNORECASE | re.DOTALL)


@dataclass
class RefinedAnswer:
    dimension: str
    sub_question: str
    text: str
    score: float


class ReflectiveFeedbackAgent(Agent):
    system_prompt = (
        "You are the Reflective Feedback Agent in a multi-agent information "
        "retrieval system. You will be given the user's original query and "
        "a candidate pseudo-answer generated for a sub-question derived from "
        "it. Judge how relevant and informative the pseudo-answer is for "
        "retrieving documents that satisfy the original query's intent, on "
        "a scale from 0.0 (irrelevant/hallucinated) to 1.0 (highly relevant "
        "and on-topic). Then rewrite the pseudo-answer keeping only the "
        "informative, on-topic content (remove filler, hedging, or "
        "off-topic material).\n"
        "Respond in exactly two lines:\n"
        "Score: <float between 0 and 1>\n"
        "Refined: <the condensed, on-topic answer text>"
    )

    def __init__(self, llm, relevance_threshold: float = 0.4):
        super().__init__(llm)
        self.relevance_threshold = relevance_threshold

    def run(self, query: str, pseudo_answer: PseudoAnswer) -> RefinedAnswer:
        prompt = f"Original query: {query}\nPseudo-answer: {pseudo_answer.text}"
        response = self._call(prompt)
        score, refined_text = self._parse(response, fallback_text=pseudo_answer.text)
        return RefinedAnswer(
            dimension=pseudo_answer.dimension,
            sub_question=pseudo_answer.sub_question,
            text=refined_text,
            score=score,
        )

    def run_all(self, query: str, pseudo_answers: list[PseudoAnswer]) -> list[RefinedAnswer]:
        refined = [self.run(query, pa) for pa in pseudo_answers]
        return [r for r in refined if r.score >= self.relevance_threshold]

    @staticmethod
    def _parse(response: str, fallback_text: str) -> tuple[float, str]:
        score_match = _SCORE_RE.search(response)
        refined_match = _REFINED_RE.search(response)

        score = float(score_match.group(1)) if score_match else 0.5
        score = max(0.0, min(1.0, score))

        refined_text = refined_match.group(1).strip() if refined_match else fallback_text
        return score, refined_text
