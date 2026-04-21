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
