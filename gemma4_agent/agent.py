import json
import httpx
from openai import OpenAI

# Point to Docker Model Runner's OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed-locally"
)

# 1. Define the Tools available to the agent
def get_stock_price(ticker: str) -> str:
    """Mock database/API look up for real-time portfolio data."""
    mock_db = {"AAPL": "185.50 USD", "GOOG": "175.20 USD", "MSFT": "420.10 USD"}
    return f"The current price of {ticker} is {mock_db.get(ticker.upper(), 'Unknown')}."

AVAILABLE_TOOLS = {
    "get_stock_price": get_stock_price
}

# 2. Instruct Gemma 4 on how to think and act
SYSTEM_PROMPT = """
You are a smart AI Agent operating in a loop: Reason, Act, Observe.
You have access to the following tools:
- get_stock_price(ticker: str): Returns the latest price of a stock asset.

To use a tool, you MUST use the exact JSON format below:
Action: {"tool": "tool_name", "args": {"param": "value"}}

When you have the final answer after observing tool outputs, reply with:
Final Answer: [Your definitive response here]

Always output your internal thought process first.
"""


def run_agent(user_request: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]

    # Limit execution loop to prevent infinite runaways
    for i in range(5):
        print(f"\n--- Iteration {i+1} ---")

        # Trigger Gemma 4's reasoning mode natively
        response = client.chat.completions.create(
            model="gemma4:E4B",
            messages=messages,
            temperature=1.0, # Recommended sampling for Gemma 4
            extra_body={"chat-template-kwargs": {"enable_thinking": True}}
        )

        response_text = response.choices[0].message.content
        print(response_text)

        # Append the assistant's turn to memory
        messages.append({"role": "assistant", "content": response_text})

        # Scenario A: Agent reached a conclusion
        if "Final Answer:" in response_text:
            break

        # Scenario B: Agent is requesting a Tool execution
        if "Action:" in response_text:
            try:
                # Parse the Action JSON block
                action_line = [line for line in response_text.split('\n') if "Action:" in line][0]
                action_json = json.loads(action_line.replace("Action:", "").strip())

                tool_name = action_json["tool"]
                tool_args = action_json["args"]

                # Execute the bound python function
                if tool_name in AVAILABLE_TOOLS:
                    print(f"[System Execution]: Running {tool_name}({tool_args})...")
                    observation = AVAILABLE_TOOLS[tool_name](**tool_args)
                else:
                    observation = f"Error: Tool '{tool_name}' is not recognized."

                print(f"[System Observation]: {observation}")

                # Feed the feedback back into the loop as a new user interaction
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            except Exception as e:
                error_msg = f"Error parsing/executing action: {str(e)}. Correct your formatting."
                messages.append({"role": "user", "content": error_msg})

# Test Drive
run_agent("Can you look up the current value of my Apple (AAPL) holdings?")