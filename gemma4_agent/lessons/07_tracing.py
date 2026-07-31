"""
LESSON 7 — Tracing & Observability
====================================
"You can't debug what you can't see."

In production, your agent will make wrong decisions, call the wrong tool,
loop unnecessarily, or time out. Without a trace, you have no idea why.

A TRACE is a time-ordered log of every decision and action an agent took
to produce a single answer. Production platforms like LangSmith, Arize, and
Braintrust are built around this primitive.

This lesson shows you how to build it yourself:

  ┌────────────────────────────────────────────────────────────────┐
  │                        EXECUTION TRACE                         │
  │                                                                │
  │  [00ms]  Model call  →  "call get_market_data(AAPL)"          │
  │  [84ms]  Tool call   →  get_market_data(AAPL)  →  185.50 USD  │
  │  [85ms]  Model call  →  "Final Answer: ..."                    │
  │  [166ms] Done.  2 model calls, 1 tool call, 166ms total        │
  └────────────────────────────────────────────────────────────────┘

WHAT WE BUILD:
  • Span         — a single timed event (model call or tool call)
  • Tracer       — collects spans for one agent run
  • TracedAgent  — a SubAgent wrapper that emits spans automatically
  • save_trace() — writes the full trace to a JSON file

WHAT YOU LEARN:
  • Where to instrument in the agent loop (before/after API call, before/after tool)
  • What metadata to capture (token counts, latency, finish reason)
  • How to render it readably with colorama colours
  • How to save structured traces for offline debugging / monitoring

Run:  python3 lessons/07_tracing.py
"""

import csv
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

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
TRACE_DIR    = os.path.join(PROJECT_ROOT, "traces")


# ── 1. SPAN — a single timed event ────────────────────────────────────────────
# Every observable event in an agent run becomes a Span.
# Spans nest: a "run" span contains "model_call" and "tool_call" spans.

@dataclass
class Span:
    """One timed event in an agent execution."""
    kind:          str              # "model_call" | "tool_call" | "run"
    name:          str              # human-readable label
    start_ms:      float = 0.0     # wall time since run start (ms)
    duration_ms:   float = 0.0     # how long this event took
    # Model call fields
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    finish_reason:     str = ""
    # Tool call fields
    tool_name:   str = ""
    tool_args:   dict = field(default_factory=dict)
    tool_result: str = ""
    # Run-level fields
    iteration:   int = 0
    error:       str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── 2. TRACER — collects spans for one agent run ──────────────────────────────

class Tracer:
    """
    Accumulates spans for a single agent run and produces:
      • a coloured terminal summary
      • a JSON file in the traces/ directory
    """

    def __init__(self, question: str):
        self.run_id    = str(uuid.uuid4())[:8]
        self.question  = question
        self.spans:    list[Span] = []
        self._wall_start = time.time()
        # Open the run span
        self.spans.append(Span(kind="run", name="agent_run"))

    # ── helpers ─────────────────────────────────────────────────────────────

    def _now_ms(self) -> float:
        """Milliseconds since run start."""
        return (time.time() - self._wall_start) * 1000

    def start_model_call(self, iteration: int) -> float:
        """Call just before client.chat.completions.create().  Returns start time."""
        return time.time()

    def finish_model_call(
        self,
        t_start: float,
        response,
        iteration: int,
    ) -> None:
        """Call immediately after receiving the model response."""
        elapsed = (time.time() - t_start) * 1000
        usage   = response.usage
        choice  = response.choices[0]
        span = Span(
            kind              = "model_call",
            name              = f"model_call#{iteration}",
            start_ms          = self._now_ms() - elapsed,
            duration_ms       = elapsed,
            prompt_tokens     = usage.prompt_tokens     if usage else 0,
            completion_tokens = usage.completion_tokens if usage else 0,
            finish_reason     = choice.finish_reason or "",
            iteration         = iteration,
        )
        self.spans.append(span)
        self._print_model_span(span)

    def start_tool_call(self) -> float:
        return time.time()

    def finish_tool_call(
        self,
        t_start:  float,
        name:     str,
        args:     dict,
        result:   str,
        iteration: int,
    ) -> None:
        elapsed = (time.time() - t_start) * 1000
        span = Span(
            kind        = "tool_call",
            name        = f"tool:{name}#{iteration}",
            start_ms    = self._now_ms() - elapsed,
            duration_ms = elapsed,
            tool_name   = name,
            tool_args   = args,
            tool_result = result[:200],   # truncate long results in trace
            iteration   = iteration,
        )
        self.spans.append(span)
        self._print_tool_span(span)

    def finish_run(self, final_answer: str | None) -> dict:
        """
        Close the run, print the summary, save to disk.
        Returns the full trace dict.
        """
        total_ms  = self._now_ms()
        model_spans = [s for s in self.spans if s.kind == "model_call"]
        tool_spans  = [s for s in self.spans if s.kind == "tool_call"]
        total_prompt     = sum(s.prompt_tokens     for s in model_spans)
        total_completion = sum(s.completion_tokens for s in model_spans)

        # Update the root run span
        self.spans[0].duration_ms = total_ms

        self._print_summary(total_ms, model_spans, tool_spans,
                            total_prompt, total_completion, final_answer)

        trace = {
            "run_id":   self.run_id,
            "question": self.question,
            "total_ms": round(total_ms, 1),
            "model_calls":     len(model_spans),
            "tool_calls":      len(tool_spans),
            "prompt_tokens":   total_prompt,
            "completion_tokens": total_completion,
            "final_answer":    final_answer,
            "spans": [s.to_dict() for s in self.spans],
        }
        self._save(trace)
        return trace

    # ── coloured printing ───────────────────────────────────────────────────

    def _print_model_span(self, s: Span) -> None:
        tokens = f"{s.prompt_tokens}→{s.completion_tokens} tok"
        print(
            f"  {Fore.CYAN}[{s.start_ms:6.0f}ms]{Style.RESET_ALL}"
            f"  {Fore.BLUE}MODEL{Style.RESET_ALL}  "
            f"iter={s.iteration}  {tokens:>16}  "
            f"finish={Fore.YELLOW}{s.finish_reason}{Style.RESET_ALL}  "
            f"{Fore.WHITE}{s.duration_ms:.0f}ms{Style.RESET_ALL}"
        )

    def _print_tool_span(self, s: Span) -> None:
        args_str = json.dumps(s.tool_args)
        print(
            f"  {Fore.CYAN}[{s.start_ms:6.0f}ms]{Style.RESET_ALL}"
            f"  {Fore.GREEN}TOOL {Style.RESET_ALL}  "
            f"{Fore.GREEN}{s.tool_name}{Style.RESET_ALL}"
            f"({Fore.WHITE}{args_str}{Style.RESET_ALL})  "
            f"→  {s.tool_result[:60]!r}  "
            f"{Fore.WHITE}{s.duration_ms:.1f}ms{Style.RESET_ALL}"
        )

    def _print_summary(
        self,
        total_ms:   float,
        model_spans: list,
        tool_spans:  list,
        prompt_tok:  int,
        comp_tok:    int,
        final:       str | None,
    ) -> None:
        bar = "─" * 62
        print(f"\n{Fore.MAGENTA}{bar}{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}TRACE SUMMARY{Style.RESET_ALL}  run_id={self.run_id}")
        print(f"  Wall time : {total_ms:.0f} ms")
        print(f"  Model calls : {len(model_spans)}   Tool calls : {len(tool_spans)}")
        print(f"  Tokens  — prompt: {prompt_tok}  completion: {comp_tok}  "
              f"total: {prompt_tok + comp_tok}")
        if final:
            print(f"  Answer  : {final[:80]}{'…' if len(final) > 80 else ''}")
        print(f"{Fore.MAGENTA}{bar}{Style.RESET_ALL}")

    # ── persistence ─────────────────────────────────────────────────────────

    def _save(self, trace: dict) -> None:
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, f"trace_{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(trace, f, indent=2)
        print(f"  {Fore.YELLOW}Trace saved → {path}{Style.RESET_ALL}\n")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

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


ALL_TOOL_FUNCTIONS: dict[str, Any] = {
    "get_market_data": get_market_data,
    "get_user_balance": get_user_balance,
    "calculate": calculate,
}

ALL_TOOL_SCHEMAS = [
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


# ── 3. TRACED AGENT ───────────────────────────────────────────────────────────
# This is the key pattern: the agent loop is identical to Lesson 6, except
# every model call and tool execution is bracketed by tracer calls.
#
# Notice: the AGENT LOGIC is NOT changed. Tracing is additive instrumentation.

class TracedAgent:
    """
    An agent that instruments every model call and tool call via a Tracer.

    The constructor accepts a system prompt and a Tracer instance.
    Everything else is the same as the SubAgent in Lesson 6.
    """

    def __init__(self, system_prompt: str, tracer: Tracer):
        self.system_prompt = system_prompt
        self.tracer        = tracer

    def run(self, task: str, max_iterations: int = 6) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": task},
        ]

        for iteration in range(1, max_iterations + 1):
            # ── INSTRUMENT: start timing the model call ───────────────────
            t_model = self.tracer.start_model_call(iteration)

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=ALL_TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=1.0,
                extra_body={"chat-template-kwargs": {"enable_thinking": True}},
            )

            # ── INSTRUMENT: record model span ─────────────────────────────
            self.tracer.finish_model_call(t_model, response, iteration)

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)

                for tc in choice.message.tool_calls:
                    func_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    fn = ALL_TOOL_FUNCTIONS.get(func_name)
                    if not fn:
                        result = json.dumps({"error": f"Unknown tool: {func_name}"})
                    else:
                        # ── INSTRUMENT: start timing the tool call ────────
                        t_tool = self.tracer.start_tool_call()
                        result = fn(**args)
                        # ── INSTRUMENT: record tool span ──────────────────
                        self.tracer.finish_tool_call(t_tool, func_name, args, result, iteration)

                    messages.append({
                        "role":        "tool",
                        "tool_call_id": tc.id,
                        "content":     result,
                    })

            elif choice.finish_reason == "stop":
                content = choice.message.content or ""
                answer  = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return answer

        return "Agent did not complete within max iterations."


# ── 4. RUN IT ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a financial assistant. Answer questions using the available tools. "
    "Be concise. Always include the numbers you looked up."
)


def run_with_trace(question: str) -> None:
    """Run the traced agent on one question, then print and save the trace."""
    print(f"\n{'═'*62}")
    print(f"{Fore.WHITE}QUESTION: {question}{Style.RESET_ALL}")
    print(f"{'═'*62}")

    tracer = Tracer(question)
    agent  = TracedAgent(SYSTEM_PROMPT, tracer)

    answer = agent.run(question)

    print(f"\n{Fore.GREEN}FINAL ANSWER:{Style.RESET_ALL}")
    print(answer)

    tracer.finish_run(answer)


if __name__ == "__main__":
    # Single tool call
    run_with_trace("What is the current price of AAPL?")

    # Two tool calls + calculation
    run_with_trace(
        "mayank wants to buy 10 shares of NVDA. "
        "Get his balance and the market price of NVDA, then calculate the cost."
    )
