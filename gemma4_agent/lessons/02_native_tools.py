"""
LESSON 2 — Native Tool Calling (OpenAI `tools_local` API)
====================================================
Instead of parsing "Action: {...}" from plain text, we use the OpenAI
`tools_local` parameter to give the model a JSON schema for each tool.

The model returns a structured `tool_calls` object — no text parsing needed.

HOW IT WORKS:
  1. You define tools_local as JSON schemas (what the function does, its parameters).
  2. You pass them to the API: client.chat.completions.create(tools_local=[...])
  3. The model returns finish_reason="tool_calls" instead of "stop".
  4. You read response.choices[0].message.tool_calls to get the call details.
  5. Execute your Python function, then inject the result as a "tool" role message.

COMPARISON WITH LESSON 1:
  Lesson 1 (ReAct/text):   fragile text parsing, works with any model
  Lesson 2 (native tools_local):  structured JSON, reliable, industry standard

IMPORTANT — Message format for tool calls:
  assistant message must include tool_calls (not just content)
  tool result goes in a message with role="tool" and a tool_call_id

Run:  python3 lessons/02_native_tools.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI

MODEL = "docker.io/ai/gemma4:E4B"

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed",
)


# ── 1. TOOL FUNCTIONS ──────────────────────────────────────────────────────────

def get_stock_price(ticker: str) -> str:
    prices = {
        "AAPL": 185.50, "GOOG": 175.20, "MSFT": 420.10,
        "NVDA": 875.40, "TSLA": 245.80, "AMZN": 180.05,
    }
    price = prices.get(ticker.upper())
    if price:
        return json.dumps({"ticker": ticker.upper(), "price": price, "currency": "USD"})
    return json.dumps({"error": f"Ticker '{ticker.upper()}' not found."})


def calculate(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression."""
    try:
        # Only allow digits, operators, spaces, dots, parentheses
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid characters in expression."})
        result = eval(expression)  # noqa: S307 (safe because we filtered)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_company_info(ticker: str) -> str:
    companies = {
        "AAPL":  {"name": "Apple Inc.",       "sector": "Technology", "employees": 164000},
        "GOOG":  {"name": "Alphabet Inc.",    "sector": "Technology", "employees": 182000},
        "MSFT":  {"name": "Microsoft Corp.",  "sector": "Technology", "employees": 221000},
        "NVDA":  {"name": "NVIDIA Corp.",     "sector": "Semiconductors", "employees": 29600},
        "TSLA":  {"name": "Tesla Inc.",       "sector": "Automotive", "employees": 127855},
        "AMZN":  {"name": "Amazon.com Inc.",  "sector": "E-Commerce", "employees": 1541000},
    }
    info = companies.get(ticker.upper())
    if info:
        return json.dumps({"ticker": ticker.upper(), **info})
    return json.dumps({"error": f"No company info for '{ticker.upper()}'."})


# ── 2. TOOL SCHEMAS ────────────────────────────────────────────────────────────
# This is what you pass to the API. Each schema has:
#   type: always "function"
#   function.name: matches your Python function name
#   function.description: tells the model WHEN to use this tool
#   function.parameters: JSON Schema describing the arguments

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price of a stock by its ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. 'AAPL', 'GOOG'.",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression. Use this for any math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '185.50 * 100' or '(420.10 - 185.50) / 2'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "Get company details (name, sector, employee count) for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol.",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
]

# Map schema names → Python functions
TOOL_FUNCTIONS = {
    "get_stock_price": get_stock_price,
    "calculate": calculate,
    "get_company_info": get_company_info,
}


# ── 3. EXECUTE A TOOL CALL ─────────────────────────────────────────────────────
def execute_tool(tool_call) -> str:
    """
    Given an OpenAI ToolCall object, find and run the corresponding function.
    Returns the result as a string (must be a string for the tool message).
    """
    name = tool_call.function.name
    # The arguments come back as a JSON string — parse them
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": "Could not parse tool arguments."})

    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return json.dumps({"error": f"Unknown tool: '{name}'"})

    print(f"  [TOOL CALL] {name}({args})")
    try:
        result = func(**args)
        print(f"  [TOOL RESULT] {result}")
        return result
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── 4. THE AGENT LOOP ─────────────────────────────────────────────────────────
def run_agent(user_request: str, max_iterations: int = 6):
    """
    Native tool-calling agent loop.

    Key difference from Lesson 1:
      - We pass `tools_local=TOOL_SCHEMAS` to every API call.
      - When finish_reason == "tool_calls", we execute tools_local and add them as
        role="tool" messages — not as fake "user" messages.
      - The message history follows the OpenAI multi-turn tool protocol exactly.
    """
    print(f"\n{'='*60}")
    print(f"USER: {user_request}")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": (
            "You are a helpful financial analyst. "
            "Use the available tools_local to answer questions accurately. "
            "After all needed data is gathered, give a clear, concise final answer."
        )},
        {"role": "user", "content": user_request},
    ]

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",   # "auto" = model decides when to call tools_local
            temperature=1.0,
            extra_body={"chat-template-kwargs": {"enable_thinking": True}},
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        message = choice.message

        print(f"[finish_reason: {finish_reason}]")

        # ── CASE A: Model wants to call one or more tools_local ───────────────────
        if finish_reason == "tool_calls":
            # The assistant message must be appended AS-IS (with tool_calls field)
            # so the model knows it already requested these calls.
            messages.append(message)  # the full message object, not just content

            for tool_call in message.tool_calls:
                result = execute_tool(tool_call)

                # Each tool result is a separate "tool" role message.
                # tool_call_id links it back to the specific call.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # ── CASE B: Model is done reasoning ─────────────────────────────────
        elif finish_reason == "stop":
            content = message.content or ""
            # Strip <think> blocks for display
            import re
            visible = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            print(f"\n[FINAL ANSWER]\n{visible}")
            print(f"{'='*60}")
            return visible

        else:
            print(f"[AGENT] Unexpected finish_reason: {finish_reason}")
            break

    print("[AGENT] Max iterations reached.")
    return None


# ── 5. RUN IT ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: single tool
    run_agent("What is the current price of NVDA stock?")

    # Test 2: tool + calculation (two different tools_local chained)
    run_agent(
        "If I own 50 shares of AAPL and 30 shares of MSFT, "
        "what is the total current value of my holdings?"
    )

    # Test 3: company info + price together
    run_agent("Tell me about TSLA — what sector is it in and what is its current stock price?")
