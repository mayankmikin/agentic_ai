import json
import urllib

from openai import OpenAI
import sys
from tools import AVAILABLE_TOOLS,get_current_weather,list_directory_contents,execute_python_code

# Point to Docker Model Runner's OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" # required but not used
)

# A dictionary to easily access the functions by name
TOOL_FUNCTIONS = {
    "get_current_weather": get_current_weather,
    "execute_python_code": execute_python_code,
    "list_directory_contents": list_directory_contents,
}


# 2. Instruct Gemma 4 on how to think and act
SYSTEM_PROMPT = """
You are a smart AI Agent operating in a loop: Reason, Act, Observe.
You have access to the following tools from python dictionary: 
TOOL_FUNCTIONS

To use a tool, you MUST use the exact JSON format below:
Action: {"tool": "tool_name", "args": {"param": "value"}}

When you have the final answer after observing tool outputs, reply with:
Final Answer: [Your definitive response here]

Always output your internal thought process first.
"""

def call_ollama(payload):
    """Helper function to call the local Ollama API."""
    url = "http://localhost:11434/api/chat"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def run_agent(user_request: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]
    payload = {
        "model": "gemma4:e4b",  # The model you specified
        "messages": messages,
        "tools": AVAILABLE_TOOLS,
        "stream": False
    }

    # get the first response from ollama

    try:
        response_data = call_ollama(payload)
    except Exception as e:
        print(f"    └─ [ERROR] Error calling Ollama API: {e}")
        print("    └─ Make sure Ollama is running and the gemma4:e2b model is pulled.")
        return

    message = response_data.get("message", {})
    # Add the model's tool calls to the chat history
    messages.append(message)

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



def main():
    print("\n[SYSTEM]")

    for tool in TOOL_FUNCTIONS.keys():
        print(f"  ○ {('Tool: ' + tool).ljust(45, '.')} [LOADED]")
    print()

    print("[EXECUTION]")
    print("  ● Querying model...\n")

    # Allow user prompt to be sent via command line as an arg
    if len(sys.argv) > 1:
        user_query = "".join(sys.argv[1:])
    else:
        print("Please try again and provide a prompt to use.\n")
        sys.exit(1)

    run_agent(user_query)

if __name__ == "__main__":
    main()