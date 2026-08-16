from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from client_type import ollama_client


# 1. State definition
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Define tools and bind to model
@tool
def get_fish_stock_data(species_key: str) -> str:
    """Fetch current TAC quotas and catch statistics."""
    return "Atlantic Cod TAC: 453,000 tonnes | Reported: 380,000 tonnes"

tools = [get_fish_stock_data]
model_with_tools = ollama_client.bind_tools(tools)

# 3. Define the agent node
def call_model(state: AgentState):
    system_message = SystemMessage(
        content="You are an AI Fisheries Officer for Scandinavian marine management."
    )
    messages = [system_message] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# 4. Construct the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

agent = workflow.compile()

# 5. Run the graph
# response = agent.invoke({
#     "messages": [HumanMessage(content="Check the remaining quota for Atlantic Cod.")]
# })

# 4. Execute a Multi-Step Reasoning Task
query = (
    "A vessel fleet in Tromsø wants to land 30,000 tonnes of Atlantic Cod from the Barents Sea. "
    "Check the current stock status and confirm if this catch is legally compliant under current quotas."
)

response = agent.invoke({"messages": [HumanMessage(content=query)]})

print(response["messages"][-1].content)