from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from client_type import ollama_client
from tools_local import available_tools

tools = available_tools

system_prompt = (
    "You are an AI Fisheries Officer for Scandinavian marine management. "
    "Use your tools_local to inspect official quota records before answering."
)

# New API
agent = create_agent(
    model=ollama_client,
    tools=tools,
    system_prompt=system_prompt
)

# response = agent.invoke({
#     "messages": [HumanMessage(content="What is the current quota status for Atlantic Cod?")]
# })
#
# print(response["messages"][-1].content)


# 4. Execute a Multi-Step Reasoning Task
query = (
    "A vessel fleet in Tromsø wants to land 30,000 tonnes of Atlantic Cod from the Barents Sea. "
    "Check the current stock status and confirm if this catch is legally compliant under current quotas."
)

response = agent.invoke({"messages": [HumanMessage(content=query)]})

# Print the agent's final decision
print(response["messages"][-1].content)