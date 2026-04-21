import re
import ast
import math
import os
import subprocess
import tempfile
from collections import Counter
from typing import Optional

from utils import call_model_chat_completions, extract_final_answer

class ReasoningAgent:
    def __init__(self, max_calls: int = 20):
        self.max_calls = max_calls
        self.call_count = 0

    def _remaining(self) -> int:
        return self.max_calls - self.call_count

    def _call_llm(self, prompt: str, system: Optional[str] = None,
                  temperature: float = 0.0, max_tokens: int = 512) -> str:
        if self.call_count >= self.max_calls:
            raise RuntimeError(f"Exceeded maximum LLM calls ({self.max_calls})")
        self.call_count += 1
        if system is None:
            system = (
                "You are a precise reasoning assistant. "
                "Reason step by step and end with 'Answer: <final answer>'."
            )
        resp = call_model_chat_completions(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        if not resp["ok"]:
            raise RuntimeError(f"API error: {resp['error']}")
        return resp["text"] or ""

    # Technique 1: Chain-of-Thought
    def chain_of_thought(self, question: str, temperature: float = 0.0) -> str:
        prompt = (
            f"Question: {question}\n\n"
            f"Think step by step, then write 'Answer: <final answer>' on its own line."
        )
        return extract_final_answer(
            self._call_llm(prompt, temperature=temperature, max_tokens=1024)
        )

    # Technique 2: Self-Consistency
    def self_consistency(self, question: str, num_samples: int = 3) -> str:
        samples = []
        for _ in range(num_samples):
            if self._remaining() < 1:
                break
            try:
                samples.append(self.chain_of_thought(question, temperature=0.7))
            except RuntimeError:
                break
        if not samples:
            return ""
        counts = Counter(s.strip().lower() for s in samples if s)
        if not counts:
            return samples[0]
        top, freq = counts.most_common(1)[0]
        for s in samples:
            if s.strip().lower() == top:
                return s
        return samples[0]
