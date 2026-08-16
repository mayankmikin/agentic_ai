import json
import uuid
from typing import Annotated
from typing_extensions import TypedDict

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from client_type import ollama_client
from tools_local import FISHERIES_DATABASE


# ==========================================
# 1. Fisheries Database & Quota State Global / Cached Database Manager
# ==========================================
@st.cache_resource
def get_database():
    """Persistent singleton database across runs and tool executions."""
    return FISHERIES_DATABASE


# ==========================================
# 2. Tools (Safely accessing the cached DB)
# ==========================================
@tool
def get_fish_stock_data(species_key: str) -> str:
    """Fetch current TAC quotas and catch statistics for a Scandinavian species.
    Allowed keys: 'atlantic_cod', 'north_sea_herring', 'atlantic_salmon'
    """
    db = get_database()
    key = species_key.lower().replace(" ", "_")
    data = db.get(key)
    if not data:
        return f"Error: '{species_key}' not found. Available: {list(db.keys())}"
    return json.dumps(data, indent=2)


@tool
def log_commercial_catch(species_key: str, requested_tonnes: float) -> str:
    """Evaluate and log commercial landing against remaining TAC quotas."""
    db = get_database()
    key = species_key.lower().replace(" ", "_")
    data = db.get(key)
    if not data:
        return f"Error: Unknown species '{species_key}'."

    tac = data["tac_tonnes"]
    reported = data["reported_catch_tonnes"]
    remaining = tac - reported

    if requested_tonnes <= remaining:
        # Update cached database
        db[key]["reported_catch_tonnes"] += requested_tonnes
        new_remaining = tac - db[key]["reported_catch_tonnes"]
        return (
            f"APPROVED & LOGGED: {requested_tonnes:,.1f} tonnes landed for {data['common_name']}. "
            f"New quota remaining: {new_remaining:,.1f} tonnes."
        )
    else:
        return (
            f"REJECTED: Requested catch ({requested_tonnes:,.1f} tonnes) exceeds "
            f"safe remaining quota ({remaining:,.1f} tonnes)."
        )


# ==========================================
# 3. LangGraph Setup
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@st.cache_resource
def get_agent_graph():
    tools = [get_fish_stock_data, log_commercial_catch]

    client = ollama_client
    model_with_tools = client.bind_tools(tools)

    def call_model(state: AgentState):
        system_prompt = SystemMessage(
            content=(
                "You are an AI Fisheries Compliance Officer for the Nordic Directorate of Marine Resources. "
                "Always check stock status via tools before approving or logging catches. "
                "Provide direct, factual summaries of TAC usage."
            )
        )
        messages = [system_prompt] + [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="Nordic Fisheries Quota Agent", page_icon="🐟", layout="wide")

db = get_database()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": "Welcome to the Scandinavian Fisheries Quota Portal. How can I assist with your vessel landings and stock verification today?"}
    ]

# Sidebar
with st.sidebar:
    st.header("📊 Active Stock Quotas")
    for key, info in db.items():
        used = info["reported_catch_tonnes"]
        total = info["tac_tonnes"]
        pct = min(1.0, used / total)

        st.subheader(info["common_name"])
        st.caption(f"Area: {info['region']}")
        st.progress(pct)
        st.write(f"**{used:,.0f}** / **{total:,.0f}** tonnes ({pct * 100:.1f}%)")
        st.divider()

    if st.button("Reset Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Session reset. Please enter your query."}
        ]
        st.rerun()

# Chat Area
st.title("🐟 Nordic Fisheries Compliance & Quota Agent")
st.caption("Powered by LangGraph + Local Gemma via Ollama")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("E.g., What is the remaining quota for Atlantic Cod?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    graph = get_agent_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Evaluating compliance & querying database..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
            response_content = result["messages"][-1].content
            st.markdown(response_content)

    st.session_state.chat_history.append({"role": "assistant", "content": response_content})
    st.rerun()