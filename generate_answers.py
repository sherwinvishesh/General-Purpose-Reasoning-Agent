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
