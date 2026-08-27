"""Shared base class for the three AMD agents."""
from __future__ import annotations

import abc

from amd.llm import BaseLLM


class Agent(abc.ABC):
    """An LLM-backed role in the AMD dialogic-inquiry pipeline."""

    system_prompt: str

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def _call(self, user_prompt: str) -> str:
        return self.llm.chat(self.system_prompt, user_prompt)
