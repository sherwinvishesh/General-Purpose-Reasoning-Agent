from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from agent import ReasoningAgent

INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")

# Write to disk every N questions. On a 6000-question run this means the file
SAVE_EVERY = 25

def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data

def load_existing_progress(expected_count: int) -> List[Dict[str, str]]:
    if not OUTPUT_PATH.exists():
        return []
    try:
        with OUTPUT_PATH.open("r") as fp:
            existing = json.load(fp)
    except Exception:
        return []
    if not isinstance(existing, list) or not existing:
        return []

    is_placeholder = all(
        isinstance(a, dict) and isinstance(a.get("output"), str)
        and "Placeholder answer" in a["output"]
        for a in existing
    )
    if is_placeholder:
        print("Existing file is the course placeholder. Starting fresh.")
        return []

    if len(existing) >= expected_count:
        print(f"Existing file has {len(existing)} answers (complete). "
              f"Delete {OUTPUT_PATH} first if you want a clean re-run.")
        return []

    print(f"Resuming: found {len(existing)} previously saved answers.")
    return existing

def save_answers(answers: List[Dict[str, str]]) -> None:
    tmp = OUTPUT_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)
    tmp.replace(OUTPUT_PATH)

def format_duration(secs: float) -> str:
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"

def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers = load_existing_progress(len(questions))
    start_idx = len(answers)
    n = len(questions)

    if start_idx >= n:
        print("All questions already answered; nothing to do.")
        return answers

    agent = ReasoningAgent(max_calls=20)
    print(f"Processing questions {start_idx + 1} through {n}")
    print("-" * 78)

    run_start = time.time()

    for idx in range(start_idx, n):
        question = questions[idx]
        q_start = time.time()
        try:
            output = agent.answer(question["input"])
        except Exception as exc:
            output = f"ERROR: {exc}"
        output = str(output) if output is not None else ""
        if len(output) >= 5000:
            output = output[:4900]
        answers.append({"output": output})

        done = idx + 1
        q_elapsed = time.time() - q_start
        preview = output.replace("\n", " ")[:80]
        print(f"[{done}/{n}] t={q_elapsed:4.1f}s calls={agent.call_count:2d}  "
              f"=> {preview!r}", flush=True)

        if done % SAVE_EVERY == 0 or done == n:
            save_answers(answers)
            done_in_session = done - start_idx
            total_elapsed = time.time() - run_start
            avg = total_elapsed / done_in_session if done_in_session else 0
            eta = (n - done) * avg
            print(f"  [checkpoint] saved {done}/{n}  "
                  f"avg={avg:.1f}s/q  ETA={format_duration(eta)}", flush=True)
            print("-" * 78)

    return answers

def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs "
            f"{len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: "
                f"{type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars)."
            )

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing {INPUT_PATH} in current directory.")
    questions = load_questions(INPUT_PATH)
    print(f"Loaded {len(questions)} questions")

    answers = build_answers(questions)
    save_answers(answers)

    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"\nWrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )

if __name__ == "__main__":
    main()
