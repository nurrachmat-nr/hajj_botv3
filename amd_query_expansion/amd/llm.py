"""LLM backends used by the AMD agents.

Every backend implements the same minimal interface:

    llm.chat(system_prompt: str, user_prompt: str) -> str

so the agents in ``amd.agents`` never need to know which backend is behind
them. Three backends are provided:

- ``HFChatLLM``   — local Hugging Face ``transformers`` model
                     (default: Qwen2.5-7B-Instruct, as in the paper).
- ``OpenAIChatLLM`` — any OpenAI-compatible chat-completions endpoint.
- ``MockLLM``     — deterministic, offline, dependency-free stand-in used by
                     the demo script and the unit tests.
"""
from __future__ import annotations

import abc
import hashlib
import textwrap


class BaseLLM(abc.ABC):
    """Common interface for all chat-style LLM backends."""

    @abc.abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Return the assistant's text response for a single-turn chat call."""


class HFChatLLM(BaseLLM):
    """Local Hugging Face causal LM served through ``transformers``.

    Defaults to ``Qwen/Qwen2.5-7B-Instruct``, the generator used in the
    paper's experiments.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        dtype: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype if dtype != "auto" else torch.bfloat16,
        )
        self.model.eval()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class OpenAIChatLLM(BaseLLM):
    """Any OpenAI-compatible chat-completions endpoint (OpenAI, vLLM, TGI, ...)."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 256,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()


class MockLLM(BaseLLM):
    """Deterministic, offline stand-in used for tests and the demo script.

    It does not call any real model; instead it produces plausible-looking,
    deterministic text derived from the prompt content so that the full
    pipeline (agents -> expansion -> retrieval -> evaluation) is exercisable
    without network access, API keys, or GPUs.
    """

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        digest = hashlib.sha256((system_prompt + user_prompt).encode()).hexdigest()[:6]
        sp = system_prompt.lower()

        if "socratic questioning agent" in sp:
            query = _extract_between(user_prompt, "Query:", "\n")
            return "\n".join(
                [
                    f"1. Clarification: What precisely does '{query}' refer to, and how is it defined? [{digest}]",
                    f"2. Assumption probing: What assumptions does the query '{query}' take for granted? [{digest}]",
                    f"3. Implication probing: What are the consequences or downstream effects related to '{query}'? [{digest}]",
                ]
            )

        if "dialogic answering agent" in sp:
            sub_q = _extract_between(user_prompt, "Sub-question:", "\n")
            return textwrap.shorten(
                f"Regarding '{sub_q}', relevant background suggests key facts, terminology and "
                f"context ({digest}) that a well-informed source would mention when addressing it.",
                width=200,
            )

        if "reflective feedback agent" in sp:
            answer = _extract_between(user_prompt, "Pseudo-answer:", "\n")
            score = 0.5 + (int(digest, 16) % 50) / 100.0
            refined = textwrap.shorten(answer, width=120)
            return f"Score: {score:.2f}\nRefined: {refined}"

        return f"[mock response {digest}]"


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    if start_marker not in text:
        return text.strip()[:80]
    after = text.split(start_marker, 1)[1]
    return after.split(end_marker, 1)[0].strip()


def build_llm(backend: str, **kwargs) -> BaseLLM:
    """Factory used by scripts/CLI to build an LLM backend from a config string."""

    backend = backend.lower()
    if backend == "hf":
        return HFChatLLM(**kwargs)
    if backend == "openai":
        return OpenAIChatLLM(**kwargs)
    if backend == "mock":
        return MockLLM()
    raise ValueError(f"Unknown LLM backend: {backend!r}")
