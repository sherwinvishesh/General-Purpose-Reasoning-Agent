# CSE 476 Final Project - General-Purpose Reasoning Agent

A reasoning agent that answers arbitrary questions using eight inference-time
techniques, built on the ASU SOL Research Compute LLM endpoint. Submission
for CSE 476 (Natural Language Processing), Spring 2026.

## What this project does

The agent takes a list of 6208 diverse questions - elementary science, arithmetic
word problems, multi-hop factual questions with context, logic puzzles,
commonsense reasoning, code-writing, symbolic planning (blocksworld, logistics),
and more - and produces a short answer string for each one. Every answer is
generated through one or more calls to the `qwen3-30b-a3b-instruct-2507` model
hosted on ASU's SOL cluster, with a hard cap of 20 model calls per question.

The core idea is that a single prompt pattern doesn't work well for all question
types, so we implement **eight distinct reasoning strategies** (Chain-of-Thought,
Self-Consistency, Tree-of-Thought, Self-Refine, ReAct, Decomposition,
Tool-Augmented, and PAL) and use a **lightweight classifier** to pick the best
strategy for each question. When the call budget runs low, the agent falls back
to cheaper techniques instead of crashing.

For the technical writeup covering the eight techniques, classifier logic, and
evaluation results, see `REPORT.md`. This README focuses on how to set the
project up and run it.



## Repository layout

```
.
├── agent.py                                  
├── utils.py                                 
├── generate_answers.py                     
├── smoke_test.py                            
├── cse_476_final_project_test_data.json     
├── cse_476_final_project_answers.json                                     
├── README.md                                
├── .gitignore
└── .env.example                                     
```

Each file's role:

- **`utils.py`** - Low-level API transport. Wraps the SOL chat-completions
  endpoint with retry/backoff on HTTP 429 (rate limit) and 5xx (server errors),
  loads the API key from `.env`, and exposes `call_model_chat_completions()` and
  the `extract_final_answer()` regex helper.
- **`agent.py`** - The `ReasoningAgent` class. Implements the eight techniques,
  the feature-based `_classify` router, and the top-level `answer()` controller
  that routes questions to techniques and handles budget exhaustion.
- **`generate_answers.py`** - The entry point for the full test run. Reads the
  6208 questions, runs the agent on each one, prints progress per question,
  and saves partial results every 25 questions using an atomic temp-file rename
  (so a crash never leaves the output file corrupted).
- **`smoke_test.py`** - Runs just the first 3 questions and prints them. Use
  this after setup to confirm everything is connected before committing to a
  10+ hour full run.



## Requirements

- **Python 3.11+** (we developed on 3.13).
- **ASU network access** - the SOL LLM endpoint (`openai.rc.asu.edu`) is only
  reachable from inside the ASU network. Install Cisco Secure Client from
  `sslvpn.asu.edu` and connect before running anything.
- **An SOL API key** - create one at ASU Research Computing → LLM Access →
  Create Key.
- Two Python packages: `requests` and `python-dotenv`. That's it. We don't use
  any LLM SDKs, agent frameworks, or extra tooling.



## Setup

Clone the repo, create a virtual environment, and install dependencies:

```bash
git clone 
cd 
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install requests python-dotenv
```

Create a `.env` file in the repo root with your API key:

```bash
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env
```

`.env` is in `.gitignore` and must never be committed. If you accidentally push a
key, rotate it immediately in the Research Computing portal.

Optional environment variables (sensible defaults already set in `utils.py`):

```
API_BASE=https://openai.rc.asu.edu/v1
MODEL=qwen3-30b-a3b-instruct-2507
```



## Running it

### Step 1: confirm the API key works

```bash
python3 -c "
from utils import call_model_chat_completions
r = call_model_chat_completions('What is 2+2? Answer with just the number.')
print('ok:', r['ok'], 'status:', r['status'], 'text:', (r['text'] or '').strip())
"
```

Expected output: `ok: True status: 200 text: 4`

If you see status 401 → bad key. If the request hangs → you're not on VPN.

### Step 2: smoke test the pipeline

```bash
python3 smoke_test.py
```

This runs the first three test questions end-to-end through the full agent and
prints the answers. Takes about 15 seconds. Confirms the whole stack works
before you commit to the long run.

### Step 3: run the full test set

```bash
python3 generate_answers.py
```

This processes all 6208 questions and writes to
`cse_476_final_project_answers.json`. Expect **10–16 hours** depending on
question mix and endpoint load. Progress prints to the terminal as each question
finishes.

On macOS, use `caffeinate` so the machine doesn't sleep mid-run:

```bash
caffeinate -di python3 generate_answers.py 2>&1 | tee run.log
```

### Resuming an interrupted run

`generate_answers.py` saves progress every 25 questions. If the run is killed
(Ctrl-C, laptop closed, network drop, whatever), just re-run the same command
and it will pick up from the last checkpoint. It detects three cases:

- No output file yet → starts fresh.
- Output file contains placeholder text → treats it as empty, starts fresh.
- Output file contains real partial answers → resumes from where it left off.
- Output file is already complete → refuses to clobber; delete the file first
  if you actually want a re-run.



## How it answers a single question

When you call `agent.answer(question_text)`:

1. The call counter resets to zero.
2. `_classify()` inspects the question text (digit count, keywords like "solve"
   or "suppose", presence of a `Context:` block, overall length) and labels the
   question as `math`, `logic`, `multihop`, or `commonsense`. No LLM call.
3. The router picks the first technique from that category's preference list
   that fits in the remaining call budget. For example, `math` prefers PAL
   first, then self-consistency, then tool-augmented, then plain CoT.
4. The technique runs, possibly issuing 1–5 LLM calls. Each call is counted.
5. The raw model output goes through `extract_final_answer()`, which pulls out
   a clean answer string using patterns like `Answer: X`, `#### X`, `\boxed{X}`,
   or "the answer is X".
6. If anything raises `RuntimeError` (only happens on budget exhaustion), the
   top-level handler does one last attempt with plain CoT if any budget remains,
   and returns an empty string if even that fails.

In practice we average around **three LLM calls per question** - well under the
20-call hard cap. See `REPORT.md` for the full routing table and technique
descriptions.




## Constraints we respected

The course imposes several hard rules, all of which this submission complies
with:

- **Only the provided LLM.** No GPT, Claude, Gemini, or other paid API is
  used. The sole model is `qwen3-30b-a3b-instruct-2507` on ASU SOL.
- **No hardcoded delegation.** We don't shortcut questions to external tools.
  The CALC and PAL mechanisms are invoked by the model itself as part of its
  reasoning, not triggered by pattern-matching on the question.
- **Call budget.** Every technique checks remaining budget before each LLM
  call. The hard cap of 20 is never exceeded on any question.
- **At least 8 distinct techniques.** Chain-of-Thought, Self-Consistency,
  Tree-of-Thought, Self-Refine, ReAct, Decomposition, Tool-Augmented, and PAL.



## Team

- **Sherwin Jathanna** 
- **Sounak Ghosh** 
- **Kush Sharma**

## Team and Contributions

| Team Member       | Responsibilities                                                                 |
|-------------------|----------------------------------------------------------------------------------|
| **Sherwin Jathanna** | `utils.py` (API transport and answer extraction); `agent.py` - Chain-of-Thought, Self-Consistency, and PAL techniques; `generate_answers.py` initial implementation; final output files (answers JSON, run logs, CSVs); |
| **Sounak Ghosh**  | `smoke_test.py`; `agent.py` - Tree-of-Thought, Self-Refine, and ReAct techniques; `generate_answers.py` question-handling and answer-saving logic; `README.md`; `.env.example`. |
| **Kush Sharma** | `REPORT.md`; `agent.py` - `ReasoningAgent` class scaffold, Decomposition and Tool-Augmented techniques; `agent.py` classifier and routing controller; `generate_answers.py` `build_answers` function. |
