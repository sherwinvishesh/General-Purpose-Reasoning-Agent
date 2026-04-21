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

    
    # Technique 3: Tree-of-Thought (shallow BFS) 
    def tree_of_thought(self, question: str, breadth: int = 2) -> str:
        init_prompt = (
            f"Question: {question}\n\n"
            f"Propose {breadth} distinct first-step approaches. "
            f"Format:\n1. <approach>\n2. <approach>"
        )
        init = self._call_llm(init_prompt, temperature=0.7, max_tokens=256)
        steps = re.findall(r"\d+[.):]\s*(.+?)(?=\n\d+[.):]|\n\n|$)", init, re.DOTALL)
        steps = [s.strip() for s in steps if s.strip()]
        if not steps:
            steps = [init.strip()]

        best_step, best_score = steps[0], -1
        for step in steps[:breadth]:
            if self._remaining() < 2:
                break
            eval_prompt = (
                f"Question: {question}\n"
                f"Proposed approach: {step}\n"
                f"Rate promise on 1-10. Respond with only the integer."
            )
            score_resp = self._call_llm(eval_prompt, max_tokens=8)
            m = re.search(r"\d+", score_resp)
            score = int(m.group(0)) if m else 5
            if score > best_score:
                best_score, best_step = score, step

        expand_prompt = (
            f"Question: {question}\n"
            f"Use this approach: {best_step}\n"
            f"Work through it fully and end with 'Answer: <final answer>'."
        )
        return extract_final_answer(self._call_llm(expand_prompt, max_tokens=1024))

    # Technique 4: Self-Refine 
    def self_refine(self, question: str) -> str:
        initial = self.chain_of_thought(question)
        if self._remaining() < 1:
            return initial
        critique_prompt = (
            f"Question: {question}\n"
            f"Draft answer: {initial}\n\n"
            f"Carefully check the draft. If correct, restate it. "
            f"If wrong, explain briefly and correct it. "
            f"End with 'Answer: <final answer>'."
        )
        refined = self._call_llm(critique_prompt, max_tokens=512)
        return extract_final_answer(refined)

    # Technique 5: ReAct (iterative Thought/Action/Observation) 
    def _safe_eval(self, expr: str) -> Optional[float]:
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e,
        }
        try:
            tree = ast.parse(expr.strip(), mode="eval")
            for n in ast.walk(tree):
                if isinstance(n, ast.Call):
                    if getattr(n.func, "id", None) not in allowed:
                        return None
                elif isinstance(n, ast.Name) and n.id not in allowed:
                    return None
                elif isinstance(n, ast.Attribute):
                    return None
            return eval(compile(tree, "<string>", "eval"),
                        {"__builtins__": {}}, allowed)
        except Exception:
            return None

    def react(self, question: str, max_steps: int = 4) -> str:
        history = (
            f"Question: {question}\n\n"
            f"Solve by alternating Thought -> Action -> Observation.\n"
            f"Available actions:\n"
            f"  CALC[<expr>]  - evaluate arithmetic, e.g. CALC[2*3+4]\n"
            f"  FINISH[<ans>] - output the final answer\n"
            f"Use at most {max_steps} steps. Begin now.\n"
        )
        last = ""
        for _ in range(max_steps):
            if self._remaining() < 1:
                break
            
            resp = self._call_llm(history, temperature=0.0, max_tokens=512)
            last = resp
            m_fin = re.search(r"FINISH\[(.+?)\]", resp, re.DOTALL)
            if m_fin:
                return m_fin.group(1).strip()
            m_calc = re.search(r"CALC\[([^\]]+)\]", resp)
            if m_calc:
                expr = m_calc.group(1).strip()
                result = self._safe_eval(expr)
                history += f"\n{resp}\nObservation: {expr} = {result}\n"
                continue
            return extract_final_answer(resp)
        return extract_final_answer(last)
    # Technique 6: Decomposition (least-to-most)
    def decomposition(self, question: str) -> str:
        decomp_prompt = (
            f"Question: {question}\n\n"
            f"Split into 2-3 self-contained sub-questions needed to answer it. "
            f"Format:\n1. <sub-question>\n2. <sub-question>"
        )
        resp = self._call_llm(decomp_prompt, max_tokens=256)
        subqs = re.findall(r"\d+[.):]\s*(.+?)(?=\n\d+[.):]|\n\n|$)", resp, re.DOTALL)
        subqs = [s.strip() for s in subqs if s.strip()][:3]
        if len(subqs) < 2:
            return self.chain_of_thought(question)

        pairs = []
        for sub in subqs:
            if self._remaining() < 2:
                break
            try:
                pairs.append((sub, self.chain_of_thought(sub)))
            except RuntimeError:
                break
        if not pairs:
            return self.chain_of_thought(question)

        combine_prompt = (
            f"Original question: {question}\n\n"
            f"Sub-answers:\n"
            + "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)
            + "\n\nUsing these, answer the original question. "
              "End with 'Answer: <final>'."
        )
        return extract_final_answer(self._call_llm(combine_prompt, max_tokens=512))

    # Technique 7: Tool-Augmented (single-shot with inline tool) 
    def tool_augmented(self, question: str) -> str:
        prompt = (
            f"Question: {question}\n\n"
            f"Solve it. When you need arithmetic, write exactly one CALC[<expr>] "
            f"and stop; I will return the result in a follow-up turn. "
            f"Otherwise, end with 'Answer: <final answer>'."
        )
        resp = self._call_llm(prompt, max_tokens=512)
        m = re.search(r"CALC\[([^\]]+)\]", resp)
        if m and self._remaining() >= 1:
            expr = m.group(1).strip()
            result = self._safe_eval(expr)
            followup = (
                f"Prior reasoning:\n{resp}\n\n"
                f"Tool result: {expr} = {result}\n\n"
                f"Now finish and end with 'Answer: <final answer>'."
            )
            return extract_final_answer(self._call_llm(followup, max_tokens=256))
        return extract_final_answer(resp)
    #  Technique 8: PAL (Program-Aided Language Model)
    def pal(self, question: str, timeout_sec: int = 5) -> str:
        prompt = (
            f"Question: {question}\n\n"
            f"Write a short Python snippet that prints the final answer. "
            f"You may `import math`. Do not use input() or network calls. "
            f"Output only the code, inside a ```python``` fenced block."
        )
        code_resp = self._call_llm(prompt, max_tokens=1024)
        m = re.search(r"```(?:python)?\s*\n(.*?)```", code_resp, re.DOTALL)
        code = m.group(1) if m else re.sub(r"```(?:python)?\n?|```", "", code_resp).strip()

        if not code.strip():
            return ""

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False, encoding="utf-8"
            ) as fp:
                fp.write(code)
                tmp_path = fp.name
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True, text=True, timeout=timeout_sec,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().splitlines()[-1].strip()
        except Exception:
            pass
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return ""

