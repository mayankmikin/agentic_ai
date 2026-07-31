"""
LESSON 6 — Multi-Agent Orchestration
======================================
So far, one agent handled everything. For complex tasks, it's better to have
SPECIALISED agents that each do one thing well, coordinated by an ORCHESTRATOR.

ARCHITECTURE:
  ┌──────────────────────────────────────────────────────┐
  │                   ORCHESTRATOR                       │
  │  Understands the task, routes to the right agent     │
  └────────────┬─────────────────┬────────────────┬──────┘
               │                 │                │
               ▼                 ▼                ▼
        ┌──────────┐     ┌──────────────┐  ┌──────────────┐
        │ PORTFOLIO │     │   MARKET     │  │  CALCULATOR  │
        │  AGENT    │     │   ANALYST    │  │    AGENT     │
        │           │     │   AGENT      │  │              │
        │ Knows DB, │     │ Knows CSV,   │  │ Arithmetic   │
        │ balances, │     │ trends,      │  │ only         │
        │ holdings  │     │ comparisons  │  │              │
        └──────────┘     └──────────────┘  └──────────────┘

WHY MULTI-AGENT?
  • Specialisation — each agent has a focused system prompt and minimal tools.
    This reduces confusion and hallucination.
  • Scalability — add new specialists without touching existing ones.
  • Parallelism — for independent subtasks, run agents concurrently (see note at bottom).
  • Separation of concerns — portfolio data vs market data are different domains.

IMPLEMENTATION APPROACH (simple, synchronous):
  1. Orchestrator classifies the question → picks one or more agents.
  2. Each sub-agent runs with its own tools and system prompt.
  3. Orchestrator aggregates the results into a final response.

Run:  python3 lessons/06_multi_agent.py
"""

import csv
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI

MODEL = "docker.io/ai/gemma4:E4B"

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "market_data.csv")
DB_PATH  = os.path.join(PROJECT_ROOT, "portfolio.db")


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS (shared across agents; each agent only gets the ones it needs)
# ═══════════════════════════════════════════════════════════════════════════════

def get_user_balance(username: str) -> str:
    """Portfolio Agent: check a user's cash balance."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT balance, currency FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()
        if row:
            return json.dumps({"username": username, "balance": row[0], "currency": row[1]})
        return json.dumps({"error": f"User '{username}' not found."})
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})


def get_asset_price_db(ticker: str) -> str:
    """Portfolio Agent: get a stock's price from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT price, currency FROM assets WHERE ticker = ?", (ticker.upper(),))
        row = cur.fetchone()
        conn.close()
        if row:
            return json.dumps({"ticker": ticker.upper(), "price": row[0], "currency": row[1]})
        return json.dumps({"error": f"'{ticker.upper()}' not in DB."})
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})


def get_market_data(ticker: str) -> str:
    """Market Agent: price, daily change, volume from CSV."""
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


def get_top_movers() -> str:
    """Market Agent: find the biggest daily gainers and losers."""
    try:
        rows = []
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                change_str = row["daily_change"].replace("%", "").replace("+", "")
                try:
                    rows.append((row["ticker"].upper(), float(change_str)))
                except ValueError:
                    pass
        rows.sort(key=lambda x: x[1], reverse=True)
        return json.dumps({
            "top_gainer": {"ticker": rows[0][0],  "daily_change": f"{rows[0][1]:+.1f}%"},
            "top_loser":  {"ticker": rows[-1][0], "daily_change": f"{rows[-1][1]:+.1f}%"},
        })
    except FileNotFoundError:
        return json.dumps({"error": "market_data.csv not found."})


def calculate(expression: str) -> str:
    """Calculator Agent: evaluate an arithmetic expression."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid characters."})
        result = eval(expression)  # noqa: S307
        return json.dumps({"expression": expression, "result": round(result, 4)})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALISED AGENTS
# Each agent is defined by: system_prompt + tools it's allowed to use
# ═══════════════════════════════════════════════════════════════════════════════

class SubAgent:
    """A specialised agent with its own identity, tools, and tool schemas."""

    def __init__(self, name: str, system_prompt: str, tool_schemas: list, tool_functions: dict):
        self.name = name
        self.system_prompt = system_prompt
        self.tool_schemas = tool_schemas
        self.tool_functions = tool_functions

    def _execute_tool(self, tool_call) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": "Bad JSON"})
        func = self.tool_functions.get(name)
        if not func:
            return json.dumps({"error": f"Unknown tool: {name}"})
        print(f"    [{self.name}] ▶ {name}({args})")
        result = func(**args)
        print(f"    [{self.name}] ◀ {result}")
        return result

    def run(self, task: str) -> str:
        """Run this agent on a specific task. Returns its textual conclusion."""
        print(f"\n  ── {self.name} working on: {task[:80]}...")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": task},
        ]
        for _ in range(6):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto",
                temperature=1.0,
                extra_body={"chat-template-kwargs": {"enable_thinking": True}},
            )
            choice = response.choices[0]
            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    result = self._execute_tool(tc)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            elif choice.finish_reason == "stop":
                content = choice.message.content or ""
                return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return "Sub-agent did not complete in time."


# ── Portfolio Agent ───────────────────────────────────────────────────────────
portfolio_agent = SubAgent(
    name="PortfolioAgent",
    system_prompt=(
        "You are a portfolio data specialist. You answer questions about user balances "
        "and stock prices stored in the portfolio database. "
        "Be concise and always include the numeric values in your answer."
    ),
    tool_schemas=[
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
                "name": "get_asset_price_db",
                "description": "Get a stock's price from the portfolio database.",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
        },
    ],
    tool_functions={
        "get_user_balance": get_user_balance,
        "get_asset_price_db": get_asset_price_db,
    },
)

# ── Market Analysis Agent ─────────────────────────────────────────────────────
market_agent = SubAgent(
    name="MarketAnalyst",
    system_prompt=(
        "You are a market analysis specialist. You analyse real-time market data: "
        "prices, daily changes, trading volumes, and trends from the live market feed. "
        "Provide data-driven insights and highlight notable movements."
    ),
    tool_schemas=[
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
                "name": "get_top_movers",
                "description": "Find today's biggest gainer and biggest loser.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ],
    tool_functions={
        "get_market_data": get_market_data,
        "get_top_movers": get_top_movers,
    },
)

# ── Calculator Agent ──────────────────────────────────────────────────────────
calculator_agent = SubAgent(
    name="Calculator",
    system_prompt=(
        "You are a precise calculation assistant. "
        "Given numbers and an arithmetic task, compute the answer exactly using the calculate tool. "
        "Show your work clearly."
    ),
    tool_schemas=[
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
        }
    ],
    tool_functions={"calculate": calculate},
)


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# Classifies the request, delegates to sub-agents, aggregates results
# ═══════════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_AGENTS = {
    "portfolio":   portfolio_agent,
    "market":      market_agent,
    "calculation": calculator_agent,
}

ORCHESTRATOR_SYSTEM = """You are an orchestrator that routes financial questions to specialised agents.

Available agents:
- "portfolio"   — user balances, database stock prices, holdings queries
- "market"      — live market data, daily changes, volume, top movers/losers
- "calculation" — arithmetic computations, portfolio value calculations

Given a user question, respond with a JSON object listing which agents to call and what task to give each.
Format EXACTLY:
{
  "agents": [
    {"agent": "portfolio",   "task": "specific task description"},
    {"agent": "market",      "task": "specific task description"},
    {"agent": "calculation", "task": "calculation task with the numbers if known, or 'compute X * Y'"}
  ]
}

Only include agents that are actually needed. Order matters — portfolio/market before calculation.
"""


def orchestrate(user_question: str):
    print(f"\n{'='*65}")
    print(f"QUESTION: {user_question}")
    print(f"{'='*65}")

    # ── Step 1: Classify and decompose ─────────────────────────────────────
    print("\n🧭 ORCHESTRATOR: routing...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_SYSTEM},
            {"role": "user",   "content": user_question},
        ],
        temperature=1.0,
        extra_body={"chat-template-kwargs": {"enable_thinking": True}},
    )
    raw = response.choices[0].message.content or ""
    visible = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Parse the routing JSON
    try:
        json_match = re.search(r"\{.*\}", visible, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in orchestrator response.")
        routing = json.loads(json_match.group())
        agent_tasks = routing.get("agents", [])
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[ERROR] Orchestrator routing failed: {e}\nRaw: {visible}")
        return

    print(f"\nRouting plan ({len(agent_tasks)} agent(s)):")
    for at in agent_tasks:
        print(f"  • {at['agent']}: {at['task'][:80]}")

    # ── Step 2: Execute each sub-agent ─────────────────────────────────────
    print("\n🤖 EXECUTING SUB-AGENTS")
    print("-" * 50)
    sub_results = {}
    for at in agent_tasks:
        agent_name = at.get("agent", "")
        task       = at.get("task", "")
        agent      = ORCHESTRATOR_AGENTS.get(agent_name)

        if not agent:
            print(f"  [SKIP] Unknown agent: '{agent_name}'")
            continue

        result = agent.run(task)
        sub_results[agent_name] = result
        print(f"  ✅ {agent_name} done.")

    # ── Step 3: Synthesise final answer ────────────────────────────────────
    print("\n✍️  SYNTHESISING FINAL ANSWER")
    print("-" * 50)

    sub_results_text = "\n\n".join(
        f"[{name}] {result}" for name, result in sub_results.items()
    )

    synthesis_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a senior financial advisor synthesising findings from multiple specialist agents. "
                "Write a clear, complete, well-structured answer to the user's question. "
                "Integrate all the data from the agent reports. "
                "Do not repeat agent names — write as if it's your own analysis."
            )},
            {"role": "user", "content": (
                f"User question: {user_question}\n\n"
                f"Agent findings:\n{sub_results_text}\n\n"
                f"Write the final answer."
            )},
        ],
        temperature=1.0,
        extra_body={"chat-template-kwargs": {"enable_thinking": True}},
    )
    final_raw = synthesis_response.choices[0].message.content or ""
    final = re.sub(r"<think>.*?</think>", "", final_raw, flags=re.DOTALL).strip()

    print(f"\n{'='*65}")
    print("FINAL ANSWER:")
    print(final)
    print(f"{'='*65}")
    return final


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Routes to market agent only
    orchestrate("Who is today's top gaining stock in the market?")

    # Routes to portfolio + market + calculation
    orchestrate(
        "mayank wants to buy 20 shares of AAPL. "
        "Does he have enough cash? Use his balance from the database "
        "and the live price from market data."
    )

    # Routes to market agent (two tickers + comparison)
    orchestrate(
        "Compare the current market performance of TSLA and NVDA. "
        "Which has better daily momentum today?"
    )

# ── NOTE: PARALLEL EXECUTION ──────────────────────────────────────────────────
# In this lesson, sub-agents run sequentially (one after another).
# For production, you'd run independent agents in parallel using threads or asyncio:
#
#   import concurrent.futures
#   with concurrent.futures.ThreadPoolExecutor() as pool:
#       futures = {pool.submit(agent.run, task): name for name, task in assignments}
#       results = {name: f.result() for name, f in futures.items()}
#
# This is important when agents are independent and you want low latency.
