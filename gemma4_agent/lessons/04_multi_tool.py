"""
LESSON 4 — Multi-Tool Agent: Chaining Tool Calls
=================================================
Real tasks often require chaining multiple tools — the output of one tool
feeds into the decision to call the next.

EXAMPLE FLOW:
  User: "Is my AAPL + MSFT portfolio worth more than $50,000?"
  
  Iteration 1: get_stock_price("AAPL")  → $185.50
  Iteration 2: query_portfolio_db("ticker", "MSFT") wait we have MSFT price
               get_stock_price("MSFT")  → $420.10
  Iteration 3: query_portfolio_db("balance", "mayank") → 50 AAPL + 30 MSFT
  Iteration 4: calculate("50*185.50 + 30*420.10") → $18,278
  Final Answer: No, current value is $18,278. That's below $50,000.

The agent decides what to call and when — you just define the tools.

This lesson connects to your REAL tools: csv_tool and sqlite_tool.
We also add a calculator tool so the agent can do math.

Run:  python3 lessons/04_multi_tool.py
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

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "market_data.csv")
DB_PATH  = os.path.join(PROJECT_ROOT, "portfolio.db")


# ── 1. TOOL FUNCTIONS ─────────────────────────────────────────────────────────

def get_market_data(ticker: str) -> str:
    """Read extended market data (price, daily change, volume) from CSV."""
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
        return json.dumps({"error": f"Ticker '{ticker.upper()}' not in market_data.csv"})
    except FileNotFoundError:
        return json.dumps({"error": f"market_data.csv not found at {CSV_PATH}"})


def query_portfolio(query_type: str, parameter: str) -> str:
    """
    Query the SQLite portfolio database.
    query_type: 'balance' (user's cash balance) | 'price' (asset price) | 'holdings' (user's stocks)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()

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
            return json.dumps({"error": f"Ticker '{parameter.upper()}' not in database."})

        else:
            return json.dumps({"error": f"Unknown query_type '{query_type}'. Use 'balance' or 'price'."})

    except sqlite3.Error as e:
        return json.dumps({"error": f"DB error: {e}"})
    finally:
        conn.close()


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression. Use this for any math calculation."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid characters in expression."})
        result = eval(expression)  # noqa: S307
        return json.dumps({"expression": expression, "result": round(result, 4)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def compare_stocks(ticker_a: str, ticker_b: str) -> str:
    """
    Compare two stocks by price. Returns which is more expensive and by how much.
    Demonstrates a tool that internally uses other data (calls get_market_data logic).
    """
    def _get_price(t: str) -> float | None:
        try:
            with open(CSV_PATH, newline="") as f:
                for row in csv.DictReader(f):
                    if row["ticker"].upper() == t.upper():
                        return float(row["price"])
        except FileNotFoundError:
            pass
        return None

    price_a = _get_price(ticker_a)
    price_b = _get_price(ticker_b)

    if price_a is None:
        return json.dumps({"error": f"'{ticker_a.upper()}' not found."})
    if price_b is None:
        return json.dumps({"error": f"'{ticker_b.upper()}' not found."})

    diff = abs(price_a - price_b)
    pct  = diff / min(price_a, price_b) * 100
    more_expensive = ticker_a.upper() if price_a > price_b else ticker_b.upper()
    cheaper        = ticker_b.upper() if price_a > price_b else ticker_a.upper()

    return json.dumps({
        ticker_a.upper(): price_a,
        ticker_b.upper(): price_b,
        "more_expensive": more_expensive,
        "cheaper": cheaper,
        "price_difference": round(diff, 2),
        "percentage_difference": round(pct, 2),
    })


# ── 2. TOOL SCHEMAS ──────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_data",
            "description": (
                "Get real-time market data for a stock from the local CSV file. "
                "Returns price, daily change percentage, and trading volume."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker, e.g. 'AAPL'"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_portfolio",
            "description": (
                "Query the portfolio database. "
                "query_type='balance' returns a user's cash balance (parameter=username). "
                "query_type='price' returns a stock's price from the database (parameter=ticker)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["balance", "price"],
                        "description": "What to query: 'balance' for user cash, 'price' for stock price.",
                    },
                    "parameter": {
                        "type": "string",
                        "description": "Username (for balance) or ticker symbol (for price).",
                    },
                },
                "required": ["query_type", "parameter"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression. Always use this for arithmetic, never guess numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python arithmetic expression, e.g. '50 * 185.50 + 30 * 420.10'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_stocks",
            "description": "Compare two stocks side-by-side: price, difference, and percentage gap.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_a": {"type": "string", "description": "First ticker symbol."},
                    "ticker_b": {"type": "string", "description": "Second ticker symbol."},
                },
                "required": ["ticker_a", "ticker_b"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_market_data":  get_market_data,
    "query_portfolio":  query_portfolio,
    "calculate":        calculate,
    "compare_stocks":   compare_stocks,
}


# ── 3. AGENT RUNNER ──────────────────────────────────────────────────────────

def execute_tool(tool_call) -> str:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": "Bad argument JSON"})
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {name}"})
    print(f"  ▶ {name}({args})")
    result = func(**args)
    print(f"  ◀ {result}")
    return result


def run_agent(question: str):
    print(f"\n{'='*65}")
    print(f"QUESTION: {question}")
    print(f"{'='*65}")

    messages = [
        {"role": "system", "content": (
            "You are a financial analyst assistant. "
            "You have tools to look up stock prices (CSV), query a portfolio database (SQLite), "
            "perform calculations, and compare stocks. "
            "Chain as many tool calls as needed to fully answer the question. "
            "When you have all the data you need, give a clear and complete final answer. "
            "Always show your calculations so the user can verify."
        )},
        {"role": "user", "content": question},
    ]

    tool_call_log = []  # track all tools used for the summary

    for iteration in range(1, 10):
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
                tool_call_log.append(f"{tc.function.name}({tc.function.arguments})")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        elif choice.finish_reason == "stop":
            content = choice.message.content or ""
            answer = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            print(f"\n📊 Tools used ({len(tool_call_log)}):")
            for i, call in enumerate(tool_call_log, 1):
                print(f"  {i}. {call}")
            print(f"\n✅ ANSWER:\n{answer}")
            print(f"{'='*65}")
            return answer
        else:
            break

    return None


# ── 4. TEST CASES ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Single tool
    run_agent("What is the current trading volume of TSLA?")

    # Two tools chained: CSV + calculate
    run_agent(
        "If I buy 100 shares of NVDA and 50 shares of AMZN, "
        "how much would that cost me in total?"
    )

    # Two tools chained: DB + CSV comparison
    run_agent(
        "What is mayank's cash balance, and how does AAPL's price "
        "compare to GOOG's price?"
    )

    # Three tools chained: DB balance + CSV price + calculate
    run_agent(
        "mayank has a balance in the database. If he invested all of it "
        "in MSFT stock at the current CSV price, how many full shares could he buy?"
    )
