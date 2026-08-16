import os
from langchain.agents import create_agent
from langchain.tools import tool
# Import the OpenAI integration package
from langchain_openai import ChatOpenAI
from client_type import  ollama_client
# 1. Initialize the Docker Model Runner local instance
# Replace 'localhost' with 'host.docker.internal' if running inside another container
llm = ollama_client
# 2. Define custom tools_local with clear docstrings so the agent knows when to use them
@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two integers together."""
    return a + b

# Tool 2: Calculator
@tool
def calculator(expression: str) -> str:
    """Evaluates a basic math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

# Tool 3: Simulated knowledge base lookup
@tool
def knowledge_base(query: str) -> str:
    """Looks up information from a local knowledge base."""
    kb = {
        "python": "Python is a beginner-friendly programming language widely used in AI and data science.",
        "ai agent": "An AI agent is a program that uses a language model to reason and take actions.",
        "ollama": "Ollama is a tool for running language models locally on your computer.",
    }
    for key in kb:
        if key in query.lower():
            return kb[key]
    return "No information found for that query."

defined_tools = [add_numbers, calculator, knowledge_base]
# 3. Create the agent harness using the llm instance
agent = create_agent(
    model=llm,                         # Pass the configured ChatOpenAI object here
    tools=defined_tools,               # Provide a list of available tools_local
    system_prompt="You are a helpful mathematical assistant.",
)

# 4. Invoke the agent with a message list
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 245 multiplied by 18, and then divided by 5?"},
                 {"role": "user", "content": "Also, can you tell me about Python?"},
                 {"role": "user", "content": "Finally, calculate (100 + 200) * 3."}]
})

# 5. Access the final AI response
print(result["messages"][-1].content)