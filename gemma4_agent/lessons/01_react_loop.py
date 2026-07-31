"""
LESSON 1 — The ReAct Loop
=========================
ReAct = Reason + Act (coined in a 2022 Google DeepMind paper)

The agent alternates between:
  Thought  — the model's internal reasoning (the <think> block)
  Action   — a tool call expressed as JSON in plain text
  Observe  — the tool result injected back into the conversation

Pattern (one iteration):
  User:      "What is the price of AAPL?"
  Assistant: <think>I need to call get_stock_price</think>
             Action: {"tool": "get_stock_price", "args": {"ticker": "AAPL"}}
  User:      Observation: The current price of AAPL is 185.50 USD.
  Assistant: Final Answer: Apple (AAPL) is currently trading at $185.50.

WHY TEXT PARSING?
  Simple. Works with any model. No special API features needed.
  The downside is fragility — the model can mis-format the JSON.
  We fix that with a robust parser and retry on parse errors.

Run:  python3 lessons/01_react_loop.py
"""

import json
import re
import sys
import os

# ── path setup so we can import from the project root ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI

MODEL = "docker.io/ai/gemma4:E4B"

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed",
)


# ── 1. TOOLS ───────────────────────────────────────────────────────────────────
# Each tool is just a plain Python function.
# The agent cannot call these directly — it outputs a JSON command, and WE
# execute the function and hand the result back.

def get_stock_price(ticker: str) -> str:
    """Return the latest price for a stock ticker (mock data)."""
    prices = {
        "AAPL": "185.50 USD",
        "GOOG": "175.20 USD",
        "MSFT": "420.10 USD",
        "NVDA": "875.40 USD",
        "TSLA": "245.80 USD",
    }
    result = prices.get(ticker.upper())
    if result:
        return f"The current price of {ticker.upper()} is {result}."
    return f"Ticker '{ticker.upper()}' not found."


def get_weather(city: str) -> str:
    """Return mock weather for a city."""
    weather = {
        "oslo": "12°C, partly cloudy",
        "london": "8°C, rainy",
        "new york": "22°C, sunny",
    }
    return weather.get(city.lower(), f"No weather data for '{city}'.")


# Registry maps tool names → Python functions
TOOLS = {
    "get_stock_price": get_stock_price,
    "get_weather": get_weather,
}


# ── 2. SYSTEM PROMPT ───────────────────────────────────────────────────────────
# The system prompt is your contract with the model.
# It defines: what the agent IS, what tools exist, and HOW to format actions.

SYSTEM_PROMPT = """You are a financial assistant AI agent. You operate in a loop:
Reason → Act → Observe → Repeat until done.

Available tools:
- get_stock_price(ticker: str) — returns the latest price for a stock symbol.
- get_weather(city: str) — returns current weather for a city.

To call a tool, output EXACTLY this format on its own line:
Action: {"tool": "tool_name", "args": {"param1": "value1"}}

When you have enough information to answer the user, output:
Final Answer: <your complete answer here>

Rules:
- Always think before acting.
- Only call one tool per iteration.
- Never guess a ticker price — always use get_stock_price.
- After receiving an Observation, reason about it before the next step.
"""


# ── 3. HELPER: extract <think> block ──────────────────────────────────────────
def split_thinking(text: str) -> tuple[str, str]:
    """
    Gemma 4 with enable_thinking=True wraps internal reasoning in <think>...</think>.
    We separate it so we can display it distinctly and keep it out of history.

    Returns: (thinking_text, visible_text)
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        visible = text[match.end():].strip()
        return thinking, visible
    return "", text.strip()


# ── 4. HELPER: parse Action JSON ──────────────────────────────────────────────
def parse_action(text: str) -> dict | None:
    """
    Find the first 'Action: {...}' line and parse the JSON.
    Returns None if nothing is found.
    """
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("action:"):
            json_str = line[len("action:"):].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common issues: smart quotes, trailing commas
                json_str = json_str.replace(""", '"').replace(""", '"')
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
    return None


# ── 5. THE AGENT LOOP ─────────────────────────────────────────────────────────
def run_agent(user_request: str, max_iterations: int = 6):
    """
    Run the ReAct agent loop.

    The messages list IS the agent's working memory.
    Each iteration: model reasons → we parse its output → execute tool → observe.
    """
    print(f"\n{'='*60}")
    print(f"USER: {user_request}")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_request},
    ]

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        # ── REASON: ask the model what to do next ──────────────────────────
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=1.0,
            extra_body={"chat-template-kwargs": {"enable_thinking": True}},
        )
        raw_output = response.choices[0].message.content

        # Split the <think> block from the actual response
        thinking, visible = split_thinking(raw_output)

        if thinking:
            print(f"\n[THINKING]\n{thinking}\n")
        print(f"[MODEL OUTPUT]\n{visible}")

        # Store only the visible part in history (saves tokens)
        messages.append({"role": "assistant", "content": visible})

        # ── STOP: did the model reach a conclusion? ─────────────────────────
        if "Final Answer:" in visible:
            final = visible.split("Final Answer:", 1)[1].strip()
            print(f"\n{'='*60}")
            print(f"FINAL ANSWER: {final}")
            print(f"{'='*60}")
            return final

        # ── ACT: did the model request a tool? ──────────────────────────────
        action = parse_action(visible)
        if action:
            tool_name = action.get("tool", "")
            tool_args = action.get("args", {})

            if tool_name in TOOLS:
                print(f"\n[EXECUTING] {tool_name}({tool_args})")
                try:
                    observation = TOOLS[tool_name](**tool_args)
                except TypeError as e:
                    observation = f"Error: Wrong arguments for {tool_name}: {e}"
            else:
                observation = f"Error: Unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"

            print(f"[OBSERVATION] {observation}")

            # Inject the observation back — this is what drives the next iteration
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        else:
            # Model produced neither an Action nor a Final Answer
            # Nudge it to try again rather than looping silently
            print("[SYSTEM] No action or final answer detected. Nudging model...")
            messages.append({
                "role": "user",
                "content": "Please continue. Either call a tool using Action: {...} or provide your Final Answer:",
            })

    print("\n[AGENT] Max iterations reached without a final answer.")
    return None


# ── 6. RUN IT ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: single tool call
    run_agent("What is the current price of Apple stock (AAPL)?")

    # Test 2: requires 2 tool calls
    run_agent("What is the price of MSFT, and what is the weather in Oslo right now?")
