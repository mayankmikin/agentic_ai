import os
from langchain.agents import create_agent
from langchain.tools import tool
# Import the OpenAI integration package
from langchain_openai import ChatOpenAI
from client_type import  ollama_client
# 1. Initialize the Docker Model Runner local instance
# Replace 'localhost' with 'host.docker.internal' if running inside another container
llm = ollama_client

# 2. Define custom tools with clear docstrings so the agent knows when to use them
@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two integers together."""
    return a + b

# 3. Create the agent harness using the llm instance
agent = create_agent(
    model=llm,                         # Pass the configured ChatOpenAI object here
    tools=[add_numbers],               # Provide a list of available tools
    system_prompt="You are a helpful mathematical assistant.",
)

# 4. Invoke the agent with a message list
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 245 multiplied by 18, and then divided by 5?"}]
})

# 5. Access the final AI response
print(result["messages"][-1].content)