"""
LESSON 8 — Eval Harness
========================
"If you can't measure it, you can't improve it."

Every time you change your system prompt, add a tool, or upgrade the model,
you need to know: did it get better or worse?

An EVAL HARNESS does this systematically:

  1. Define test cases — question + what the correct answer looks like
  2. Run each case through the agent
  3. Score the result with automated scorers
  4. Print a pass/fail report

This is how AI teams catch regressions before shipping.

WHAT WE BUILD:
  ┌─────────────────────────────────────────────────────────────────┐
  │  EvalCase  ──  question + expected tool calls + expected facts  │
  │  Scorer    ──  ToolCallScorer  |  AnswerScorer                  │
  │  EvalHarness ─  runs all cases, tallies scores, prints table    │
  └─────────────────────────────────────────────────────────────────┘

SCORERS EXPLAINED:
  • ToolCallScorer   — Did the agent call the tools you expected?
                       Partial credit: 1/2 tools called → 0.5 score
  • AnswerScorer     — Does the final answer contain the expected facts?
                       Case-insensitive substring match, partial credit

WHAT YOU LEARN:
  • How to write test cases for agents (assertions are about BEHAVIOUR, not exact strings)
  • The difference between tool-call correctness and answer correctness
  • How to get partial-credit scoring right for noisy LLM outputs
  • How to extend the harness with new scorers (e.g. latency, token cost)

Run:  python3 lessons/08_eval_harness.py
"""

import csv
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from colorama import Fore, Style, init as colorama_init
from openai import OpenAI

colorama_init(autoreset=True)

MODEL = "docker.io/ai/gemma4:E4B"

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CSV_PATH     = os.path.join(PROJECT_ROOT, "market_data.csv")
DB_PATH      = os.path.join(PROJECT_ROOT, "portfolio.db")


# ── TOOLS (same mock data as previous lessons) ────────────────────────────────

def get_market_data(ticker: str) -> str:
    try:
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                if row["ticker"].upper() == ticker.upper():
                    return json.dumps({
                        "ticker":       row["ticker"].upper(),
                        "price":        float(row["price"]),
                        "daily_change": row["daily_change"],
                        "volume":       int(row["volume"]),
                    })
        return json.dumps({"error": f"'{ticker.upper()}' not in CSV."})
    except FileNotFoundError:
        return json.dumps({"error": "market_data.csv not found."})


def get_user_balance(username: str) -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT balance, currency FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()
        if row:
            return json.dumps({"username": username, "balance": row[0], "currency": row[1]})
        return json.dumps({"error": f"User '{username}' not found."})
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})


def calculate(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid characters."})
        result = eval(expression)  # noqa: S307
        return json.dumps({"expression": expression, "result": round(result, 4)})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_FUNCTIONS = {
    "get_market_data":  get_market_data,
    "get_user_balance": get_user_balance,
    "calculate":        calculate,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_data",
            "description": "Get live price, daily change %, and volume for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_balance",
            "description": "Get a user's cash balance from the portfolio database.",
            "parameters": {
                "type": "object",
                "properties": {"username": {"type": "string"}},
                "required": ["username"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a financial assistant. Answer questions using the available tools. "
    "Be concise. Always include the numbers you looked up."
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. AGENT (same pattern as Lesson 6/7, returns answer + tools called)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    """Everything the agent produced for one run."""
    answer:       str
    tools_called: list[str]     # e.g. ["get_market_data", "calculate"]
    duration_s:   float


def run_agent(question: str) -> AgentResult:
    """
    Run the tool-calling agent, collect all tool names called, and return
    the final answer plus metadata.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    tools_called: list[str] = []
    t0 = time.time()

    for _ in range(8):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=1.0,
            extra_body={"chat-template-kwargs": {"enable_thinking": True}},
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                name = tc.function.name
                tools_called.append(name)
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                fn     = TOOL_FUNCTIONS.get(name)
                result = fn(**args) if fn else json.dumps({"error": f"Unknown: {name}"})
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })

        elif choice.finish_reason == "stop":
            content = choice.message.content or ""
            answer  = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return AgentResult(
                answer       = answer,
                tools_called = tools_called,
                duration_s   = time.time() - t0,
            )

    return AgentResult(
        answer       = "Max iterations reached.",
        tools_called = tools_called,
        duration_s   = time.time() - t0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. EVAL CASE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    """
    One test case for the agent.

    id:                    short identifier for the report table
    question:              what to ask the agent
    expected_tools:        tools that MUST be called (subset match — order doesn't matter)
    expected_answer_contains: strings that must appear in the answer (case-insensitive)
    """
    id:                       str
    question:                 str
    expected_tools:           list[str] = field(default_factory=list)
    expected_answer_contains: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SCORERS
# ══════════════════════════════════════════════════════════════════════════════

class ToolCallScorer:
    """
    Did the agent call the expected tools?

    Partial credit:
      All expected tools called  → 1.0
      k out of n expected called → k/n
      None called                → 0.0
    """

    name = "tool_calls"

    def score(self, case: EvalCase, result: AgentResult) -> tuple[float, str]:
        if not case.expected_tools:
            return 1.0, "no tools expected (pass)"

        called_set   = set(result.tools_called)
        expected_set = set(case.expected_tools)
        hits = expected_set & called_set
        score = len(hits) / len(expected_set)

        missing  = expected_set - called_set
        extra    = called_set - expected_set
        detail_parts = []
        if hits:
            detail_parts.append(f"✓ {sorted(hits)}")
        if missing:
            detail_parts.append(f"✗ missing {sorted(missing)}")
        if extra:
            detail_parts.append(f"+ extra {sorted(extra)}")

        return round(score, 2), "  ".join(detail_parts)


class AnswerScorer:
    """
    Does the final answer contain the expected facts?

    Uses case-insensitive substring matching.
    Partial credit: k out of n facts found → k/n
    """

    name = "answer_content"

    def score(self, case: EvalCase, result: AgentResult) -> tuple[float, str]:
        if not case.expected_answer_contains:
            return 1.0, "no answer assertions (pass)"

        answer_lower = result.answer.lower()
        found, missing = [], []
        for expected in case.expected_answer_contains:
            if expected.lower() in answer_lower:
                found.append(expected)
            else:
                missing.append(expected)

        score = len(found) / len(case.expected_answer_contains)
        parts = []
        if found:
            parts.append(f"✓ {found}")
        if missing:
            parts.append(f"✗ missing {missing}")
        return round(score, 2), "  ".join(parts)


class LatencyScorer:
    """
    Did the agent answer in time?

    Bonus scorer to show how to add non-LLM quality metrics.
    """

    name = "latency"

    def __init__(self, max_seconds: float = 30.0):
        self.max_seconds = max_seconds

    def score(self, case: EvalCase, result: AgentResult) -> tuple[float, str]:
        if result.duration_s <= self.max_seconds:
            return 1.0, f"{result.duration_s:.1f}s ≤ {self.max_seconds}s"
        return 0.0, f"{result.duration_s:.1f}s > {self.max_seconds}s (SLOW)"


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVAL HARNESS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CaseReport:
    """Results for one test case."""
    case:         EvalCase
    result:       AgentResult
    scorer_scores: dict[str, float]
    scorer_notes:  dict[str, str]

    @property
    def overall_score(self) -> float:
        """Average of all scorer scores for this case."""
        if not self.scorer_scores:
            return 0.0
        return sum(self.scorer_scores.values()) / len(self.scorer_scores)

    @property
    def passed(self) -> bool:
        return self.overall_score >= 0.8


class EvalHarness:
    """
    Runs a battery of EvalCases through the agent and scores each one.

    Usage:
        harness = EvalHarness(scorers=[ToolCallScorer(), AnswerScorer()])
        harness.add_cases(EVAL_SUITE)
        harness.run()
        harness.report()
    """

    def __init__(self, scorers: list):
        self.scorers:  list                = scorers
        self.cases:    list[EvalCase]      = []
        self.reports:  list[CaseReport]    = []

    def add_cases(self, cases: list[EvalCase]) -> None:
        self.cases.extend(cases)

    def run(self) -> None:
        print(f"\n{'═'*65}")
        print(f"  EVAL HARNESS  —  {len(self.cases)} cases  ×  {len(self.scorers)} scorers")
        print(f"{'═'*65}\n")

        for i, case in enumerate(self.cases, 1):
            print(f"[{i}/{len(self.cases)}]  {Fore.WHITE}{case.id}{Style.RESET_ALL}")
            print(f"  Q: {case.question[:80]}")

            result = run_agent(case.question)

            scorer_scores: dict[str, float] = {}
            scorer_notes:  dict[str, str]   = {}
            for scorer in self.scorers:
                score, note = scorer.score(case, result)
                scorer_scores[scorer.name] = score
                scorer_notes[scorer.name]  = note

            report = CaseReport(
                case          = case,
                result        = result,
                scorer_scores = scorer_scores,
                scorer_notes  = scorer_notes,
            )
            self.reports.append(report)

            # Per-case inline output
            overall = report.overall_score
            colour  = Fore.GREEN if report.passed else Fore.RED
            print(f"  {colour}{'PASS' if report.passed else 'FAIL'}{Style.RESET_ALL}  "
                  f"overall={overall:.2f}  {result.duration_s:.1f}s")
            for sname, note in scorer_notes.items():
                score_val = scorer_scores[sname]
                sc = Fore.GREEN if score_val >= 0.8 else (Fore.YELLOW if score_val >= 0.5 else Fore.RED)
                print(f"    {sname:<18} {sc}{score_val:.2f}{Style.RESET_ALL}  {note}")
            print()

    def report(self) -> None:
        """Print the final summary table."""
        if not self.reports:
            print("No results to report. Call .run() first.")
            return

        passed  = sum(1 for r in self.reports if r.passed)
        total   = len(self.reports)
        accuracy = passed / total * 100

        print(f"\n{'═'*65}")
        print(f"  FINAL REPORT")
        print(f"{'═'*65}")

        # Column widths
        id_w     = max(len(r.case.id) for r in self.reports) + 2
        scorer_names = [s.name for s in self.scorers]

        header = f"{'ID':<{id_w}}  {'PASS':>5}  {'OVERALL':>7}"
        for sn in scorer_names:
            header += f"  {sn[:12]:>12}"
        print(f"\n{header}")
        print("─" * len(header))

        for r in self.reports:
            colour = Fore.GREEN if r.passed else Fore.RED
            row = (
                f"{r.case.id:<{id_w}}"
                f"  {colour}{'PASS' if r.passed else 'FAIL':>5}{Style.RESET_ALL}"
                f"  {r.overall_score:>7.2f}"
            )
            for sn in scorer_names:
                sc_val = r.scorer_scores.get(sn, 0.0)
                sc = Fore.GREEN if sc_val >= 0.8 else (Fore.YELLOW if sc_val >= 0.5 else Fore.RED)
                row += f"  {sc}{sc_val:>12.2f}{Style.RESET_ALL}"
            print(row)

        print("─" * len(header))

        colour = Fore.GREEN if accuracy >= 80 else (Fore.YELLOW if accuracy >= 50 else Fore.RED)
        print(f"\n  Passed: {passed}/{total}  Accuracy: {colour}{accuracy:.1f}%{Style.RESET_ALL}")

        if accuracy < 80:
            failing = [r.case.id for r in self.reports if not r.passed]
            print(f"  {Fore.RED}Failing cases: {failing}{Style.RESET_ALL}")
        print(f"{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVAL SUITE  —  the test cases for this lesson
# ══════════════════════════════════════════════════════════════════════════════
#
# WRITING GOOD EVAL CASES:
#   • expected_tools:  what the agent MUST do (not what it might do)
#   • expected_answer_contains:  facts the answer MUST include
#     — use substrings, not exact answers; LLMs paraphrase
#     — prefer numeric values (they can't drift)
#
EVAL_SUITE: list[EvalCase] = [
    EvalCase(
        id       = "single_price",
        question = "What is the current price of AAPL?",
        expected_tools            = ["get_market_data"],
        expected_answer_contains  = ["AAPL", "185"],
    ),
    EvalCase(
        id       = "balance_lookup",
        question = "What is mayank's current cash balance?",
        expected_tools            = ["get_user_balance"],
        expected_answer_contains  = ["mayank"],
    ),
    EvalCase(
        id       = "multi_tool",
        question = "What is the NVDA price, and what would 5 shares cost?",
        expected_tools            = ["get_market_data", "calculate"],
        expected_answer_contains  = ["NVDA"],
    ),
    EvalCase(
        id       = "affordability",
        question = "Can mayank afford 3 shares of MSFT with his current balance?",
        expected_tools            = ["get_user_balance", "get_market_data"],
        expected_answer_contains  = ["mayank", "MSFT"],
    ),
    EvalCase(
        id       = "no_tool_needed",
        question = "What does P/E ratio mean in investing?",
        expected_tools            = [],           # no tool calls expected
        expected_answer_contains  = ["price", "earnings"],
    ),
]


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    harness = EvalHarness(scorers=[
        ToolCallScorer(),
        AnswerScorer(),
        LatencyScorer(max_seconds=60.0),
    ])
    harness.add_cases(EVAL_SUITE)
    harness.run()
    harness.report()
