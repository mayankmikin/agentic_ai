"""
LESSON 5 — Plan-and-Execute Agent
==================================
The agents in lessons 1–4 are "reactive" — they figure out each step on the fly.
For complex multi-step tasks, this leads to mistakes and unnecessary tool calls.

Plan-and-Execute (also called "task decomposition") adds a Planning Phase:

  PHASE 1: PLAN
    Give the model the goal + available tools_local.
    Ask it to output a numbered list of steps — NOT to execute anything yet.
    Review the plan (you could add a human approval step here).

  PHASE 2: EXECUTE
    Walk through each plan step one at a time.
    Run the appropriate tools_local for each step.
    Collect intermediate results.

  PHASE 3: SYNTHESISE
    Feed all intermediate results to the model.
    Ask for a final, coherent answer.

WHY IS THIS BETTER?
  • Forces the model to think about the whole task before acting.
  • Reduces wasted tool calls (no "oops, I should have done X first").
  • Each step is traceable — you can see exactly what the agent planned.
  • You can insert a human review between Plan and Execute.

Run:  python3 lessons/05_planning.py
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


# ── TOOLS (same as Lesson 4, condensed) ──────────────────────────────────────

def get_market_data(ticker: str) -> str:
    try:
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                if row["ticker"].upper() == ticker.upper():
                    return json.dumps({
                        "ticker": row["ticker"].upper(),
                        "price": float(row["price"]),
                        "daily_change": row["daily_change"],
                        "volume": int(row["volume"]),
                    })
        return json.dumps({"error": f"'{ticker.upper()}' not found in CSV."})
    except FileNotFoundError:
        return json.dumps({"error": f"CSV not found at {CSV_PATH}"})


def query_portfolio(query_type: str, parameter: str) -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        if query_type == "balance":
            cur.execute("SELECT balance, currency FROM users WHERE username = ?", (parameter,))
            row = cur.fetchone()
            if row:
                return json.dumps({"username": parameter, "balance": row[0], "currency": row[1]})
            return json.dumps({"error": f"User '{parameter}' not found."})
        elif query_type == "price":
            cur.execute("SELECT price, currency FROM assets WHERE ticker = ?", (parameter.upper(),))
            row = cur.fetchone()
            if row:
                return json.dumps({"ticker": parameter.upper(), "price": row[0], "currency": row[1]})
            return json.dumps({"error": f"'{parameter.upper()}' not in database."})
        return json.dumps({"error": "Unknown query_type."})
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})
    finally:
        conn.close()


def calculate(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid characters."})
        result = eval(expression)  # noqa: S307
        return json.dumps({"expression": expression, "result": round(result, 4)})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_data",
            "description": "Get price, daily_change, volume from the CSV market data file.",
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
            "name": "query_portfolio",
            "description": "Query SQLite DB: query_type='balance' (parameter=username) or 'price' (parameter=ticker).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {"type": "string", "enum": ["balance", "price"]},
                    "parameter":  {"type": "string"},
                },
                "required": ["query_type", "parameter"],
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

TOOL_FUNCTIONS = {
    "get_market_data": get_market_data,
    "query_portfolio": query_portfolio,
    "calculate":       calculate,
}


# ── HELPER: run one tool call ─────────────────────────────────────────────────

def execute_tool(tool_call) -> str:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": "Bad JSON"})
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {name}"})
    print(f"    ▶ {name}({args})")
    result = func(**args)
    print(f"    ◀ {result}")
    return result


# ── HELPER: one-shot model call (no tools_local, just text) ─────────────────────────

def ask_model(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=1.0,
        extra_body={"chat-template-kwargs": {"enable_thinking": True}},
    )
    content = response.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


# ── PHASE 1: PLAN ─────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a planning assistant. Given a task and a list of available tools_local,
output a numbered step-by-step plan to complete the task.

Available tools_local:
- get_market_data(ticker) — price, daily change, volume from CSV
- query_portfolio(query_type, parameter) — balance or stock price from SQLite
- calculate(expression) — arithmetic

Rules:
- Output ONLY the numbered steps. No prose, no tool call syntax.
- Be specific: "Step 2: Call get_market_data('AAPL') to get its current price."
- Include a final synthesis step: "Step N: Combine all gathered data into a final answer."
- Maximum 7 steps.
"""


def create_plan(task: str) -> list[str]:
    """Ask the model to decompose the task into numbered steps."""
    print("\n📋 PLANNING PHASE")
    print("-" * 50)

    plan_text = ask_model(PLANNER_SYSTEM, f"Task: {task}")
    print(plan_text)

    # Parse numbered lines: "1. ...", "Step 1: ..."
    steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if re.match(r"^(step\s*)?\d+[\.\):]", line, re.IGNORECASE) and len(line) > 5:
            # Strip the "1." or "Step 1:" prefix
            step = re.sub(r"^(step\s*)?\d+[\.\):]\s*", "", line, flags=re.IGNORECASE).strip()
            if step:
                steps.append(step)

    return steps


# ── PHASE 2: EXECUTE ─────────────────────────────────────────────────────────

def execute_step(step: str, context: list[dict]) -> str:
    """
    Execute a single plan step.
    We give the model the step description + gathered context so far.
    The model calls tools_local if needed, then returns its intermediate result.
    """
    context_text = "\n".join(
        f"Step {i+1} result: {c['result']}"
        for i, c in enumerate(context)
    ) if context else "No previous results yet."

    messages = [
        {"role": "system", "content": (
            "You are executing one step of a financial analysis plan. "
            "Use the provided tools_local to complete this specific step. "
            "After completing the step, summarise what you found in 1-2 sentences. "
            "Do NOT do more than what this step asks.\n\n"
            f"Context from previous steps:\n{context_text}"
        )},
        {"role": "user", "content": f"Execute this step: {step}"},
    ]

    for _ in range(4):
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
                result = execute_tool(tc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        elif choice.finish_reason == "stop":
            content = choice.message.content or ""
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    return "Step execution timed out."


# ── PHASE 3: SYNTHESISE ───────────────────────────────────────────────────────

def synthesise(task: str, plan: list[str], results: list[dict]) -> str:
    """Combine all step results into a final coherent answer."""
    step_summary = "\n".join(
        f"Step {i+1} ({r['step']}): {r['result']}"
        for i, r in enumerate(results)
    )

    return ask_model(
        "You are a financial analyst writing a clear, accurate summary report.",
        f"Original task: {task}\n\n"
        f"Steps executed and their results:\n{step_summary}\n\n"
        f"Write a complete, well-structured final answer to the original task. "
        f"Include all relevant numbers and conclusions.",
    )


# ── ORCHESTRATOR: ties all three phases together ──────────────────────────────

def run_planning_agent(task: str):
    print(f"\n{'='*65}")
    print(f"TASK: {task}")
    print(f"{'='*65}")

    # ── Phase 1: Create the plan ───────────────────────────────────────────
    plan = create_plan(task)

    if not plan:
        print("[ERROR] Planner produced no steps. Falling back to direct execution.")
        plan = [task]

    print(f"\n✅ Plan has {len(plan)} steps.")

    # Optional: human-in-the-loop approval
    # approval = input("\nApprove this plan? (y/n): ")
    # if approval.lower() != 'y':
    #     print("Plan rejected. Exiting.")
    #     return

    # ── Phase 2: Execute each step ─────────────────────────────────────────
    print("\n⚙️  EXECUTION PHASE")
    print("-" * 50)
    context = []
    for i, step in enumerate(plan):
        print(f"\n  Step {i+1}: {step}")
        result = execute_step(step, context)
        context.append({"step": step, "result": result})
        print(f"  → {result[:200]}{'...' if len(result) > 200 else ''}")

    # ── Phase 3: Synthesise final answer ───────────────────────────────────
    print("\n📝 SYNTHESIS PHASE")
    print("-" * 50)
    final_answer = synthesise(task, plan, context)

    print(f"\n{'='*65}")
    print("FINAL REPORT:")
    print(final_answer)
    print(f"{'='*65}")
    return final_answer


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # A multi-step analysis that benefits from planning
    run_planning_agent(
        "Analyse mayank's investment situation: check his available cash balance, "
        "look up the current prices and daily changes of AAPL and NVDA from the market data, "
        "then determine how many shares of each he could buy with half his balance, "
        "and give a brief recommendation on which looks more promising today based on daily change."
    )
