import json
from openai import OpenAI
# Import your custom tools_local from the separate files
from tools import lookup_ticker_from_csv, query_portfolio_db
from client_model import client
# Update the registry
AVAILABLE_TOOLS = {
    "query_portfolio_db": query_portfolio_db,
    "lookup_ticker_from_csv": lookup_ticker_from_csv
}

# Update the System Prompt to guide Gemma 4's reasoning
SYSTEM_PROMPT = """
You are an AI Agent operating in a loop: Reason, Act, Observe.
You have access to the following data tools_local:

1. query_portfolio_db(query_type: str, parameter: str):
   - Use this to check cash balances or asset prices in the database.
   - Arguments: query_type ('balance' or 'ticker'), parameter (the username string or ticker string).

2. lookup_ticker_from_csv(ticker: str):
   - Use this to pull extended market metrics like volume and daily change.
   - Arguments: ticker (e.g., 'AAPL', 'GOOG').

To use a tool, you MUST use the exact JSON format below:
Action: {"tool": "tool_name", "args": {"param": "value"}}

When you have the final answer after observing tool outputs, reply with:
Final Answer: [Your definitive response here]

Always output your internal thought process first.
"""
# "docker.io/gemma4:E4B",
def run_agent(user_request: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]

    for i in range(5):
        print(f"\n--- Iteration {i+1} ---")

        # --- FIX: SANITIZE TRAILING ASSISTANT MESSAGES ---
        # If the last two messages are BOTH assistant turns, merge them into one.
        if len(messages) >= 2 and messages[-1]["role"] == "assistant" and messages[-2]["role"] == "assistant":
            print("[System Guard]: Merging consecutive assistant turns to protect llama-server context...")
            merged_content = messages[-2]["content"] + "\n" + messages[-1]["content"]
            messages.pop() # Remove the last one
            messages[-1]["content"] = merged_content # Update the new last one

        # --- GUARD B: Handle cases where a tool call step failed to produce a user observation ---
        if messages[-1]["role"] == "assistant":
            # If the last message is an assistant turn, the engine expects the next turn to be a 'user' or 'tool' turn
            messages.append({"role": "user", "content": "Continue your reasoning and provide either the next Action or the Final Answer."})
        # --------------------------------------------------

        try:
            response = client.chat.completions.create(
                model="docker.io/gemma4:E4B",
                messages=messages,
                temperature=1.0,
                extra_body={
                    "chat-template-kwargs": {"enable_thinking": True},
                    "skip_special_tokens": False # Keeps Gemma's structural parsing stable
                }
            )
        except Exception as api_err:
            print(f"[API Crash Dump] Inspecting state: {messages}")
            raise api_err

        response_text = response.choices[0].message.content
        print(response_text)

        messages.append({"role": "assistant", "content": response_text})

        if "Final Answer:" in response_text:
            break

        if "Action:" in response_text:
            try:
                action_line = [line for line in response_text.split('\n') if "Action:" in line][0]
                action_json = json.loads(action_line.replace("Action:", "").strip())

                tool_name = action_json["tool"]
                tool_args = action_json["args"]

                if tool_name in AVAILABLE_TOOLS:
                    print(f"[System Execution]: Running {tool_name}({tool_args})...")
                    observation = AVAILABLE_TOOLS[tool_name](**tool_args)
                else:
                    observation = f"Error: Tool '{tool_name}' is not recognized."

                print(f"[System Observation]: {observation}")
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            except Exception as e:
                error_msg = f"Error parsing/executing action: {str(e)}. Correct your formatting."
                messages.append({"role": "user", "content": error_msg})

# Test Drive
run_agent("Can you look up the current value of my Apple (AAPL) holdings?")