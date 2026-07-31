"""
LESSON 10 — Resilience Engineering
=====================================
"Build for failure — tools break, APIs timeout, models return garbage."

Production agents don't live in a clean demo environment. Real-world issues:
  • Network hiccups cause API timeouts
  • External services return 429 / 503 for minutes at a time
  • Your CSV or database file is momentarily locked
  • A tool occasionally takes 45 seconds instead of 1

Without resilience patterns, any one of these kills the whole agent run.
This lesson shows four patterns that compose into a "resilient tool runner".

  PATTERN 1 — RetryPolicy
    Wait → retry → wait longer → retry → give up
    Uses exponential backoff + random jitter to avoid thundering-herd.

  PATTERN 2 — FallbackRegistry
    Primary tool fails? Silently route to a backup implementation.
    e.g.  CSV reader fails → use hardcoded mock DB

  PATTERN 3 — CircuitBreaker
    After N consecutive failures, OPEN the circuit.
    Subsequent calls fail fast (no waiting) until the tool recovers.
    State machine: CLOSED → OPEN → HALF-OPEN → CLOSED

  PATTERN 4 — ToolTimeout
    Enforce a maximum wall-clock time per tool call.
    Tool exceeds the limit? Raise TimeoutError, let retry/fallback handle it.

All four compose into ResilientToolRunner which wraps your tool registry.
The agent calls tools exactly as before — resilience is transparent.

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  AGENT requests: get_market_data("AAPL")                                 │
  │       │                                                                    │
  │  ┌────▼───────────────────────────────────────────────────────────────┐  │
  │  │  ResilientToolRunner                                                │  │
  │  │   ┌────────────────────────────────────────────────────────────┐   │  │
  │  │   │  CircuitBreaker.call(fn)                                    │   │  │
  │  │   │   └─ if OPEN → fail fast (no actual call)                  │   │  │
  │  │   │   └─ if CLOSED/HALF-OPEN:                                  │   │  │
  │  │   │         ┌──────────────────────────────────────────────┐   │   │  │
  │  │   │         │  RetryPolicy.execute(fn, max_retries=3)      │   │   │  │
  │  │   │         │   └─ attempt 1 → ToolTimeout wraps fn        │   │   │  │
  │  │   │         │   └─ attempt 2 → (if 1 failed)               │   │   │  │
  │  │   │         │   └─ attempt 3 → FallbackRegistry on fail    │   │   │  │
  │  │   │         └──────────────────────────────────────────────┘   │   │  │
  │  │   └────────────────────────────────────────────────────────────┘   │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────────────┘

Run:  python3 lessons/10_resilience.py
"""

import csv
import json
import os
import random
import re
import sqlite3
import sys
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

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


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 1 — RetryPolicy
# ══════════════════════════════════════════════════════════════════════════════

class RetryExhausted(Exception):
    """Raised when all retry attempts have failed."""

    def __init__(self, tool_name: str, last_error: Exception):
        self.tool_name  = tool_name
        self.last_error = last_error
        super().__init__(f"All retries failed for '{tool_name}': {last_error}")


class RetryPolicy:
    """
    Exponential backoff retry with full jitter.

    Backoff formula:
      sleep = random.uniform(0, base_delay * 2 ** attempt)
      capped at max_delay

    Full jitter (vs fixed backoff) is preferred in distributed systems
    because it prevents multiple clients from retrying in lockstep
    (the "thundering herd" problem).
    """

    def __init__(
        self,
        max_retries: int   = 3,
        base_delay:  float = 0.5,    # seconds
        max_delay:   float = 8.0,    # seconds
        retryable:   tuple = (Exception,),  # which exception types to retry
    ):
        self.max_retries = max_retries
        self.base_delay  = base_delay
        self.max_delay   = max_delay
        self.retryable   = retryable

    def execute(self, tool_name: str, fn: Callable, *args, **kwargs) -> Any:
        """
        Call fn(*args, **kwargs) up to max_retries times.
        Raises RetryExhausted if all attempts fail.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):   # attempt 0 = first try
            try:
                return fn(*args, **kwargs)
            except self.retryable as e:
                last_error = e
                if attempt == self.max_retries:
                    break

                # Calculate jittered sleep duration
                cap   = min(self.max_delay, self.base_delay * (2 ** attempt))
                sleep = random.uniform(0, cap)
                print(
                    f"  {Fore.YELLOW}[RETRY]{Style.RESET_ALL}  "
                    f"{tool_name}  attempt {attempt+1}/{self.max_retries}  "
                    f"error='{e}'  sleeping {sleep:.2f}s"
                )
                time.sleep(sleep)

        raise RetryExhausted(tool_name, last_error)


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 2 — FallbackRegistry
# ══════════════════════════════════════════════════════════════════════════════

class FallbackRegistry:
    """
    Associates primary tools with fallback alternatives.

    When the primary fails (after retries), the registry routes the call
    to the next fallback. Fallbacks are tried in order.

    Example:
        registry.register("get_market_data", [get_market_data_csv, get_market_data_mock])
        # CSV fails → try mock
    """

    def __init__(self):
        self._fallbacks: dict[str, list[Callable]] = {}

    def register(self, tool_name: str, implementations: list[Callable]) -> "FallbackRegistry":
        """
        Register ordered list of implementations (primary first, fallbacks after).
        At least two entries required (primary + at least one fallback).
        """
        if len(implementations) < 2:
            raise ValueError(f"FallbackRegistry: '{tool_name}' needs at least 2 implementations.")
        self._fallbacks[tool_name] = implementations
        return self

    def call(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Try each implementation in order. Returns the first success.
        Raises RuntimeError if all implementations fail.
        """
        impls  = self._fallbacks.get(tool_name)
        if not impls:
            raise KeyError(f"No implementations registered for '{tool_name}'.")

        errors = []
        for i, impl in enumerate(impls):
            label = "primary" if i == 0 else f"fallback-{i}"
            try:
                result = impl(*args, **kwargs)
                if i > 0:
                    print(
                        f"  {Fore.YELLOW}[FALLBACK]{Style.RESET_ALL}  "
                        f"{tool_name}: {label} succeeded after {i} failure(s)"
                    )
                return result
            except Exception as e:
                errors.append(f"{label}: {e}")
                print(
                    f"  {Fore.YELLOW}[FALLBACK]{Style.RESET_ALL}  "
                    f"{tool_name}: {label} failed — {e}"
                )

        raise RuntimeError(
            f"All implementations failed for '{tool_name}': " + " | ".join(errors)
        )


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 3 — CircuitBreaker
# ══════════════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    CLOSED    = "CLOSED"     # Normal operation. Calls go through.
    OPEN      = "OPEN"       # Failing hard. Calls rejected immediately.
    HALF_OPEN = "HALF_OPEN"  # Testing recovery. One probe call allowed.


class CircuitBreakerOpen(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""

    def __init__(self, tool_name: str):
        super().__init__(f"Circuit OPEN for '{tool_name}' — call rejected (fail fast).")


class CircuitBreaker:
    """
    Tracks consecutive failures per tool and trips after a threshold.

    State transitions:
      CLOSED  → OPEN       when consecutive_failures >= failure_threshold
      OPEN    → HALF_OPEN  when recovery_timeout seconds have passed
      HALF_OPEN → CLOSED   when the probe call succeeds
      HALF_OPEN → OPEN     when the probe call fails (reset timer)

    Thread-safe via a per-instance lock.
    """

    def __init__(
        self,
        failure_threshold: int   = 3,
        recovery_timeout:  float = 30.0,   # seconds
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout
        self._state:         dict[str, CircuitState] = {}
        self._failures:      dict[str, int]          = {}
        self._opened_at:     dict[str, float]        = {}
        self._lock = threading.Lock()

    def _get_state(self, name: str) -> CircuitState:
        state = self._state.get(name, CircuitState.CLOSED)
        if state == CircuitState.OPEN:
            # Check if recovery window has elapsed → probe
            opened = self._opened_at.get(name, 0.0)
            if time.time() - opened >= self.recovery_timeout:
                self._state[name] = CircuitState.HALF_OPEN
                return CircuitState.HALF_OPEN
        return state

    def call(self, tool_name: str, fn: Callable, *args, **kwargs) -> Any:
        """
        Route the call through the circuit breaker.
        Raises CircuitBreakerOpen if the circuit is OPEN.
        """
        with self._lock:
            state = self._get_state(tool_name)

        if state == CircuitState.OPEN:
            raise CircuitBreakerOpen(tool_name)

        if state == CircuitState.HALF_OPEN:
            print(
                f"  {Fore.YELLOW}[CIRCUIT]{Style.RESET_ALL}  "
                f"{tool_name}: HALF-OPEN — sending probe call"
            )

        try:
            result = fn(*args, **kwargs)
            # Success — reset
            with self._lock:
                if self._state.get(tool_name) in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                    print(
                        f"  {Fore.GREEN}[CIRCUIT]{Style.RESET_ALL}  "
                        f"{tool_name}: probe succeeded → CLOSED"
                    )
                self._state[tool_name]    = CircuitState.CLOSED
                self._failures[tool_name] = 0
            return result

        except Exception as e:
            with self._lock:
                self._failures[tool_name] = self._failures.get(tool_name, 0) + 1
                failures = self._failures[tool_name]

                if failures >= self.failure_threshold:
                    self._state[tool_name]    = CircuitState.OPEN
                    self._opened_at[tool_name] = time.time()
                    print(
                        f"  {Fore.RED}[CIRCUIT]{Style.RESET_ALL}  "
                        f"{tool_name}: {failures} failures → OPEN "
                        f"(will retry in {self.recovery_timeout}s)"
                    )
                else:
                    print(
                        f"  {Fore.YELLOW}[CIRCUIT]{Style.RESET_ALL}  "
                        f"{tool_name}: failure {failures}/{self.failure_threshold}"
                    )
            raise  # re-raise so retry policy can handle it

    def status(self, tool_name: str) -> str:
        """Return a readable status string for a tool."""
        state    = self._get_state(tool_name)
        failures = self._failures.get(tool_name, 0)
        return f"{tool_name}: state={state.value}  failures={failures}"


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 4 — ToolTimeout
# ══════════════════════════════════════════════════════════════════════════════

class ToolTimeoutError(Exception):
    """Raised when a tool call exceeds its allowed wall-clock time."""

    def __init__(self, tool_name: str, timeout_s: float):
        super().__init__(f"'{tool_name}' exceeded timeout of {timeout_s}s.")


class ToolTimeout:
    """
    Runs a tool function in a thread and kills it after max_seconds.

    Important: Python threads can't be forcibly killed — the tool's
    thread continues running in the background after timeout. This is an
    inherent limitation of CPython threading. For hard timeouts, use
    subprocess or a separate process (not shown here — YAGNI for most cases).

    What this DOES guarantee: your agent loop gets control back after
    max_seconds even if the tool is stuck.
    """

    def __init__(self, default_timeout: float = 10.0):
        self.default_timeout = default_timeout
        self._timeouts: dict[str, float] = {}

    def set_timeout(self, tool_name: str, seconds: float) -> "ToolTimeout":
        """Override the timeout for a specific tool."""
        self._timeouts[tool_name] = seconds
        return self

    def call(self, tool_name: str, fn: Callable, *args, **kwargs) -> Any:
        """Run fn(*args, **kwargs) with a wall-clock timeout."""
        timeout = self._timeouts.get(tool_name, self.default_timeout)

        result_box:    list[Any]       = []
        exception_box: list[Exception] = []

        def _target():
            try:
                result_box.append(fn(*args, **kwargs))
            except Exception as e:
                exception_box.append(e)

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread is still running — timeout
            raise ToolTimeoutError(tool_name, timeout)

        if exception_box:
            raise exception_box[0]

        return result_box[0]


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITION — ResilientToolRunner
# ══════════════════════════════════════════════════════════════════════════════

class ResilientToolRunner:
    """
    Composes all four resilience patterns into a single tool executor.

    Call order:
      CircuitBreaker → ToolTimeout → RetryPolicy → (FallbackRegistry on fail)

    The agent calls .run(tool_name, args) exactly as it would call the raw
    function — resilience is fully transparent.
    """

    def __init__(
        self,
        retry_policy:      RetryPolicy,
        fallback_registry: FallbackRegistry,
        circuit_breaker:   CircuitBreaker,
        tool_timeout:      ToolTimeout,
    ):
        self.retry      = retry_policy
        self.fallback   = fallback_registry
        self.circuit    = circuit_breaker
        self.timeout    = tool_timeout

    def run(self, tool_name: str, args: dict) -> str:
        """
        Execute a tool by name, applying all four resilience patterns.
        Always returns a JSON string (even on total failure).
        """

        def _call_with_timeout():
            """One attempt: timeout-wrapped call via fallback registry."""
            # ToolTimeout wraps the FallbackRegistry call
            return self.timeout.call(
                tool_name,
                self.fallback.call,
                tool_name,
                **args,
            )

        try:
            # CircuitBreaker wraps the RetryPolicy
            return self.circuit.call(
                tool_name,
                self.retry.execute,
                tool_name,
                _call_with_timeout,
            )
        except CircuitBreakerOpen as e:
            return json.dumps({"error": str(e)})
        except RetryExhausted as e:
            return json.dumps({"error": f"Tool '{tool_name}' failed after all retries: {e.last_error}"})
        except Exception as e:
            return json.dumps({"error": f"Unexpected error in '{tool_name}': {e}"})


# ══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Reliable implementations (same as earlier lessons) ────────────────────────

def get_market_data_csv(ticker: str) -> str:
    """Primary: read from CSV file."""
    try:
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                if row["ticker"].upper() == ticker.upper():
                    return json.dumps({
                        "ticker":       row["ticker"].upper(),
                        "price":        float(row["price"]),
                        "daily_change": row["daily_change"],
                        "volume":       int(row["volume"]),
                        "source":       "csv",
                    })
        return json.dumps({"error": f"'{ticker.upper()}' not in CSV."})
    except FileNotFoundError:
        raise RuntimeError("market_data.csv not found — cannot read CSV.")


def get_market_data_mock(ticker: str) -> str:
    """Fallback: hardcoded mock prices when CSV is unavailable."""
    mock = {
        "AAPL": (185.50, "+0.8%"),
        "GOOG": (175.20, "-0.3%"),
        "MSFT": (420.10, "+1.2%"),
        "NVDA": (875.40, "+3.1%"),
        "TSLA": (245.80, "-1.5%"),
    }
    key = ticker.upper()
    if key in mock:
        price, change = mock[key]
        return json.dumps({
            "ticker":       key,
            "price":        price,
            "daily_change": change,
            "volume":       1_000_000,
            "source":       "mock_fallback",
        })
    return json.dumps({"error": f"'{key}' not in mock data either."})


def get_user_balance(username: str) -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT balance, currency FROM users WHERE username = ?", (username,))
        row  = cur.fetchone()
        conn.close()
        if row:
            return json.dumps({"username": username, "balance": row[0], "currency": row[1]})
        return json.dumps({"error": f"User '{username}' not found."})
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})


def get_user_balance_mock(username: str) -> str:
    """Fallback: hardcoded balances."""
    mock = {"mayank": (15000.0, "USD"), "alice": (8500.0, "USD")}
    entry = mock.get(username.lower())
    if entry:
        return json.dumps({"username": username, "balance": entry[0], "currency": entry[1], "source": "mock_fallback"})
    return json.dumps({"error": f"'{username}' not in mock data."})


def calculate(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid characters."})
        result = eval(expression)  # noqa: S307
        return json.dumps({"expression": expression, "result": round(result, 4)})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Flaky implementation for demo ─────────────────────────────────────────────
# This simulates a real tool that fails occasionally and sometimes takes too long.

_flaky_call_count: dict[str, int] = {}

def get_market_data_flaky(ticker: str) -> str:
    """
    Demo tool that:
      - Fails the first 2 calls with a transient error
      - Hangs for 5 seconds on the 3rd call (timeout demo)
      - Succeeds on the 4th+ call
    """
    key   = ticker.upper()
    count = _flaky_call_count.get(key, 0) + 1
    _flaky_call_count[key] = count

    print(
        f"    {Fore.CYAN}[FLAKY]{Style.RESET_ALL}  "
        f"get_market_data_flaky({key}) — attempt #{count}"
    )

    if count <= 2:
        raise ConnectionError(f"Transient network error (attempt {count})")
    if count == 3:
        print(f"    {Fore.CYAN}[FLAKY]{Style.RESET_ALL}  sleeping 5s (simulated hang)...")
        time.sleep(5)
        return json.dumps({"ticker": key, "price": 185.50, "source": "flaky_slow"})
    # count >= 4: success
    return json.dumps({"ticker": key, "price": 185.50, "daily_change": "+0.8%", "source": "flaky_recovered"})


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


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE THE RESILIENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

retry_policy = RetryPolicy(
    max_retries = 4,
    base_delay  = 0.2,   # short delays for demo; use 1.0+ in production
    max_delay   = 4.0,
    retryable   = (ConnectionError, TimeoutError, ToolTimeoutError),
)

fallback_registry = FallbackRegistry()
fallback_registry.register("get_market_data",  [get_market_data_flaky, get_market_data_mock])
fallback_registry.register("get_user_balance", [get_user_balance, get_user_balance_mock])
# calculate has no fallback — it's pure deterministic code

circuit_breaker = CircuitBreaker(
    failure_threshold = 5,     # trip after 5 consecutive failures
    recovery_timeout  = 60.0,  # probe after 60 seconds
)

tool_timeout = ToolTimeout(default_timeout=10.0)
tool_timeout.set_timeout("get_market_data", 3.0)   # flaky demo: 3s max per attempt

runner = ResilientToolRunner(
    retry_policy      = retry_policy,
    fallback_registry = fallback_registry,
    circuit_breaker   = circuit_breaker,
    tool_timeout      = tool_timeout,
)


# ── CALCULATE: no fallback registered — use directly ─────────────────────────
_direct_tools: dict[str, Callable] = {
    "calculate": calculate,
}


# ══════════════════════════════════════════════════════════════════════════════
# RESILIENT AGENT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a financial assistant. Answer questions using the available tools. "
    "Be concise. Always include the numbers you looked up."
)


def run_resilient_agent(question: str) -> str:
    """
    Agent loop that routes every tool call through the ResilientToolRunner.
    """
    print(f"\n{'═'*65}")
    print(f"{Fore.WHITE}QUESTION: {question}{Style.RESET_ALL}")
    print(f"{'═'*65}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

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
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # ── Route through the resilient runner ────────────────────
                if name in _direct_tools:
                    # Tools with no fallback registered run directly
                    try:
                        result = _direct_tools[name](**args)
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                else:
                    # Tools with fallback chains go through the full runner
                    result = runner.run(name, args)

                print(f"  {Fore.GREEN}[TOOL]{Style.RESET_ALL}  {name} → {result[:80]}")
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })

        elif choice.finish_reason == "stop":
            content = choice.message.content or ""
            answer  = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return answer

    return "Agent did not complete within max iterations."


# ── RUN DEMO ──────────────────────────────────────────────────────────────────
#
# What you'll see:
#   • get_market_data is routed through get_market_data_flaky
#   • Attempts 1-2: ConnectionError → retry with backoff
#   • Attempt 3:    hangs 5s → ToolTimeout fires → retry
#   • Attempt 4+:   success (flaky tool "recovers")
#
# This demonstrates all four patterns firing on a single tool call.

if __name__ == "__main__":
    print(f"{Fore.MAGENTA}")
    print("  RESILIENCE DEMO — watch retry, timeout, and fallback in action")
    print(f"{Style.RESET_ALL}")

    answer = run_resilient_agent("What is the current price of AAPL?")

    print(f"\n{Fore.GREEN}FINAL ANSWER:{Style.RESET_ALL}")
    print(answer)

    # Show circuit breaker status after the run
    print(f"\n{Fore.CYAN}Circuit Breaker Status:{Style.RESET_ALL}")
    for tool_name in ["get_market_data", "get_user_balance"]:
        print(f"  {circuit_breaker.status(tool_name)}")

    # Second run — flaky counter has advanced; tool should succeed faster
    print(f"\n{'─'*65}")
    print("Second run (same tool — flaky counter advanced, fewer retries needed):")
    answer2 = run_resilient_agent("What is mayank's balance and what would 2 shares of AAPL cost?")
    print(f"\n{Fore.GREEN}FINAL ANSWER:{Style.RESET_ALL}")
    print(answer2)
