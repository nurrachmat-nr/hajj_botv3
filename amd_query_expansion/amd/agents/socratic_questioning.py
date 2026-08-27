"""Socratic Questioning Agent.

Reformulates the initial query into three sub-questions, one per Socratic
questioning dimension used in the paper: clarification, assumption probing,
and implication probing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from amd.agents.base import Agent

SOCRATIC_DIMENSIONS = ("clarification", "assumption_probing", "implication_probing")

_DIMENSION_DESCRIPTIONS = {
    "clarification": (
        "Clarification: ask what the query precisely means, what key terms "
        "refer to, or what exactly is being asked."
    ),
    "assumption_probing": (
        "Assumption probing: ask what the query takes for granted — the "
        "premises, definitions or context it silently assumes."
    ),
    "implication_probing": (
        "Implication probing: ask what follows from the topic — its "
        "consequences, effects, or what it leads to or depends on."
    ),
}


@dataclass
class SubQuestion:
    dimension: str
    text: str


class SocraticQuestioningAgent(Agent):
    system_prompt = (
        "You are the Socratic Questioning Agent in a multi-agent information "
        "retrieval system. Given a user's search query, reformulate it into "
        "exactly three sub-questions, each grounded in a distinct Socratic "
        "questioning dimension:\n"
        f"1. {_DIMENSION_DESCRIPTIONS['clarification']}\n"
        f"2. {_DIMENSION_DESCRIPTIONS['assumption_probing']}\n"
        f"3. {_DIMENSION_DESCRIPTIONS['implication_probing']}\n"
        "Respond with exactly three numbered lines, each starting with the "
        "dimension name, a colon, and the sub-question. Do not answer the "
        "questions yourself."
    )

    def run(self, query: str) -> list[SubQuestion]:
        response = self._call(f"Query: {query}")
        return self._parse(response, query)

    def _parse(self, response: str, query: str) -> list[SubQuestion]:
        sub_questions: list[SubQuestion] = []
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            matched_dim = None
            for dim in SOCRATIC_DIMENSIONS:
                label = dim.replace("_", " ")
                if line.lower().startswith(label):
                    matched_dim = dim
                    line = line[len(label) :].lstrip(":- ").strip()
                    break
            if matched_dim is None:
                # Fall back to positional assignment if the model didn't label the line.
                matched_dim = SOCRATIC_DIMENSIONS[len(sub_questions) % len(SOCRATIC_DIMENSIONS)]
            if line:
                sub_questions.append(SubQuestion(dimension=matched_dim, text=line))

        if not sub_questions:
            # Degenerate fallback: guarantee three sub-questions even on a parse failure.
            sub_questions = [
                SubQuestion(dimension=dim, text=f"{_DIMENSION_DESCRIPTIONS[dim].split(':')[0]} of: {query}")
                for dim in SOCRATIC_DIMENSIONS
            ]
        return sub_questions[:3]
