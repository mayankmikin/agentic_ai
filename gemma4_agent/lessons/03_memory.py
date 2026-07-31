"""
LESSON 3 — Memory: Making Agents Remember
==========================================
Agents have two kinds of memory. Understanding both is critical.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SHORT-TERM MEMORY (in-context / conversation history)
   • The `messages` list passed to every API call.
   • The model "remembers" everything in this list.
   • Dies when the Python process ends — not persisted.
   • Limited by the model's context window.
   • Use: multi-turn conversation, tool results, reasoning chains.

2. LONG-TERM MEMORY (external / persistent)
   • Data saved to a file, database, or vector store.
   • Survives across sessions.
   • You retrieve only what's relevant (to save context tokens).
   • Use: user profile, facts learned in past sessions, summaries.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This lesson builds:
  - A MemoryStore class that saves/loads facts to a JSON file.
  - Two tools: remember_fact (save) and recall_facts (load).
  - A conversational agent that accumulates knowledge across turns.

Run:  python3 lessons/03_memory.py
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI

MODEL = "docker.io/ai/gemma4:E4B"

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed",
)

# Long-term memory lives in this file (next to the lessons dir)
MEMORY_FILE = Path(__file__).parent.parent / "agent_memory.json"


# ── 1. LONG-TERM MEMORY STORE ─────────────────────────────────────────────────

class MemoryStore:
    """
    Simple key-value store persisted as JSON.

    Each memory has:
      key:       short label (e.g. "user_name", "preferred_stocks")
      value:     the fact to remember (any string)
      timestamp: when it was written
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.filepath.exists():
            with open(self.filepath) as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self._data, f, indent=2)

    def store(self, key: str, value: str) -> str:
        """Save a fact. Overwrites if key already exists."""
        self._data[key] = {
            "value": value,
            "updated_at": datetime.now().isoformat(),
        }
        self._save()
        return f"Stored: '{key}' = '{value}'"

    def retrieve(self, query: str) -> str:
        """
        Return all memory entries whose key or value contains the query.
        Simple substring search — good enough for small memory stores.
        For semantic search over large memory banks, use a vector DB (e.g., ChromaDB).
        """
        if not self._data:
            return "Memory is empty."

        query_lower = query.lower()
        matches = {
            k: v for k, v in self._data.items()
            if query_lower in k.lower() or query_lower in str(v["value"]).lower()
        }

        if not matches:
            # If no keyword match, return everything (for small stores)
            matches = self._data

        lines = [f"- {k}: {v['value']}  (saved {v['updated_at'][:10]})"
                 for k, v in matches.items()]
        return "\n".join(lines)

    def all_facts(self) -> str:
        if not self._data:
            return "No memories stored yet."
        return "\n".join(f"- {k}: {v['value']}" for k, v in self._data.items())


# Shared memory store instance
memory = MemoryStore(MEMORY_FILE)


# ── 2. TOOL FUNCTIONS ─────────────────────────────────────────────────────────

def remember_fact(key: str, value: str) -> str:
    """Persist a fact to long-term memory."""
    result = memory.store(key, value)
    return json.dumps({"status": "saved", "key": key, "value": value})


def recall_facts(query: str) -> str:
    """Search long-term memory for facts matching a keyword."""
    facts = memory.retrieve(query)
    return json.dumps({"query": query, "results": facts})


def get_stock_price(ticker: str) -> str:
    prices = {
        "AAPL": 185.50, "GOOG": 175.20, "MSFT": 420.10,
        "NVDA": 875.40, "TSLA": 245.80, "AMZN": 180.05,
    }
    price = prices.get(ticker.upper())
    if price:
        return json.dumps({"ticker": ticker.upper(), "price": price, "currency": "USD"})
    return json.dumps({"error": f"Ticker '{ticker.upper()}' not found."})


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": (
                "Save an important fact or piece of user information to long-term memory. "
                "Use this when the user tells you something to remember, like their name, "
                "preferences, portfolio details, or any other important fact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "A short label for this fact, e.g. 'user_name', 'favorite_stock'.",
                    },
                    "value": {
                        "type": "string",
                        "description": "The fact or value to remember.",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_facts",
            "description": (
                "Look up facts from long-term memory. "
                "Use this at the start of a conversation to recall what you know about the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A keyword to search memory with, e.g. 'portfolio', 'user', 'stocks'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price of a stock by ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "e.g. 'AAPL', 'MSFT'"}
                },
                "required": ["ticker"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "remember_fact": remember_fact,
    "recall_facts": recall_facts,
    "get_stock_price": get_stock_price,
}


# ── 3. AGENT LOOP (multi-turn chat) ───────────────────────────────────────────

def execute_tool(tool_call) -> str:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": "Bad arguments JSON"})
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {name}"})
    print(f"  [TOOL] {name}({args})")
    result = func(**args)
    print(f"  [RESULT] {result}")
    return result


def chat_with_memory(conversation_history: list, user_message: str) -> str:
    """
    Single turn: append user message, call the model, handle tool calls, return reply.
    The conversation_history list is mutated in-place — it IS the short-term memory.
    """
    conversation_history.append({"role": "user", "content": user_message})

    for _ in range(5):  # tool-call loop within one turn
        response = client.chat.completions.create(
            model=MODEL,
            messages=conversation_history,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=1.0,
            extra_body={"chat-template-kwargs": {"enable_thinking": True}},
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            conversation_history.append(choice.message)
            for tc in choice.message.tool_calls:
                result = execute_tool(tc)
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        elif choice.finish_reason == "stop":
            content = choice.message.content or ""
            visible = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            conversation_history.append({"role": "assistant", "content": visible})
            return visible

    return "I couldn't complete that request."


# ── 4. INTERACTIVE DEMO ───────────────────────────────────────────────────────

def main():
    print("Memory Agent — type 'quit' to exit, 'memories' to inspect stored facts\n")
    print(f"Long-term memory file: {MEMORY_FILE}\n")

    # Short-term memory: the conversation history for this session
    conversation_history = [
        {"role": "system", "content": (
            "You are a personal financial assistant. "
            "You have tools to remember and recall facts about the user across sessions. "
            "At the start, recall what you know about the user. "
            "When users tell you important things (their name, portfolio, preferences), remember them. "
            "Use remembered context to give personalised answers."
        )},
    ]

    # Kick off by recalling existing memories
    print("Recalling memories from previous sessions...")
    print()
    existing = memory.all_facts()
    if existing != "No memories stored yet.":
        print(f"[LONG-TERM MEMORY]\n{existing}\n")
        # Inject existing memories into the system context
        conversation_history[0]["content"] += f"\n\nWhat you already know about this user:\n{existing}"

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "memories":
            print(f"\n[STORED MEMORIES]\n{memory.all_facts()}\n")
            continue

        reply = chat_with_memory(conversation_history, user_input)
        print(f"\nAgent: {reply}\n")

        # Optional: summarise conversation every 10 turns to compress short-term memory
        # This is an advanced technique — see the GUIDE.md
        user_turns = sum(
            1 for m in conversation_history
            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "user"
        )
        if user_turns % 10 == 0 and user_turns > 0:
            print("[SYSTEM] Compressing conversation history...")
            # Keep system message + last 4 exchanges
            conversation_history[:] = conversation_history[:1] + conversation_history[-8:]


if __name__ == "__main__":
    main()
