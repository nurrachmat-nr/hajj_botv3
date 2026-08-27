"""Dialogic Answering Agent.

Generates a pseudo-answer for each Socratic sub-question, enriching the
query representation with multiple perspectives aligned to the user's
underlying intent.
"""
from __future__ import annotations

from dataclasses import dataclass

from amd.agents.base import Agent
from amd.agents.socratic_questioning import SubQuestion


@dataclass
class PseudoAnswer:
    dimension: str
    sub_question: str
    text: str


class DialogicAnsweringAgent(Agent):
    system_prompt = (
        "You are the Dialogic Answering Agent in a multi-agent information "
        "retrieval system. You will be given the user's original query and "
        "one sub-question derived from it. Write a short, factual "
        "pseudo-answer (2-4 sentences) as if it were an excerpt from a "
        "relevant document: use concrete terminology, entities, and facts "
        "that a real, on-topic passage would contain. Do not mention that "
        "this is a hypothetical answer, and do not restate the question."
    )

    def run(self, query: str, sub_question: SubQuestion) -> PseudoAnswer:
        prompt = f"Original query: {query}\nSub-question: {sub_question.text}"
        text = self._call(prompt)
        return PseudoAnswer(dimension=sub_question.dimension, sub_question=sub_question.text, text=text)

    def run_all(self, query: str, sub_questions: list[SubQuestion]) -> list[PseudoAnswer]:
        return [self.run(query, sq) for sq in sub_questions]
