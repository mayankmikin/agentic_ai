import uuid
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
import requests
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from client_type import ollama_client


# =======================================================
# 1. Real-Time API Tools
# =======================================================

@tool
def fetch_ices_stock_assessment(stock_code: str) -> str:
    """Fetch official scientific stock assessment data, landings history,
    and advice from the ICES Stock Assessment Graph (SAG) REST API.
    Common Scandinavian stock codes:
      - 'cod.27.1-2' (North East Arctic / Barents Sea Cod)
      - 'her.27.3a47d' (North Sea Autumn Spawning Herring)
      - 'had.27.1-2' (Barents Sea Haddock)
    """
    url = f"https://sag.ices.dk/ws/Services/StockAssessmentGraphs.asmx/getListStocks?year=2023"

    try:
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=8)
        if response.status_code == 200:
            data = response.json()
            # Filter for target stock
            matches = [s for s in data if stock_code.lower() in str(s.get("FishStock", "")).lower()]
            if matches:
                latest = matches[0]
                return (
                    f"Stock: {latest.get('FishStock')} | Assessment Year: {latest.get('AssessmentYear')}\n"
                    f"Species Common Name: {latest.get('SpeciesCommonName')}\n"
                    f"Scientific Name: {latest.get('SpeciesName')}\n"
                    f"ICES Advice Status: {latest.get('CustomStatus', 'Assessed under MSY framework')}\n"
                    f"ICES Ecoregion: {latest.get('EcoRegion', 'Barents / Norwegian / North Sea')}"
                )
            return f"Stock code '{stock_code}' not found in active ICES registry. Available examples: 'cod.27.1-2', 'her.27.3a47d'."
        return f"ICES API returned status code {response.status_code}"
    except Exception as e:
        return f"Failed to reach ICES REST service: {str(e)}"


@tool
def check_sea_conditions(latitude: float, longitude: float) -> str:
    """Fetch real-time oceanic conditions (wave height, sea surface temperature, currents)
    using the Open-Meteo Marine API. Useful for evaluating landing safety.
    Example coordinates:
      - Tromsø / Barents Coast: lat=69.65, lon=18.96
      - Bergen / North Sea: lat=60.39, lon=5.32
      - Lofoten: lat=68.16, lon=13.75
    """
    url = (
        f"https://marine-api.open-meteo.com/v1/marine?"
        f"latitude={latitude}&longitude={longitude}"
        f"&current=wave_height,wave_period,ocean_current_velocity"
    )
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            res = resp.json()
            curr = res.get("current", {})
            return (
                f"Location ({latitude:.2f}°N, {longitude:.2f}°E):\n"
                f"- Wave Height: {curr.get('wave_height', 'N/A')} m\n"
                f"- Wave Period: {curr.get('wave_period', 'N/A')} s\n"
                f"- Ocean Current: {curr.get('ocean_current_velocity', 'N/A')} m/s"
            )
        return f"Marine API error: HTTP {resp.status_code}"
    except Exception as e:
        return f"Failed to retrieve oceanic conditions: {str(e)}"


@tool
def calculate_catch_feasibility(requested_tonnes: float, max_safe_tonnes: float) -> str:
    """Calculate harvest quotas, remaining tonnage margins, and compliance ratio."""
    if requested_tonnes <= max_safe_tonnes:
        margin = max_safe_tonnes - requested_tonnes
        return (
            f"STATUS: COMPLIANT. Landing of {requested_tonnes:,.1f} tonnes approved. "
            f"Remaining sustainable quota margin: {margin:,.1f} tonnes."
        )
    diff = requested_tonnes - max_safe_tonnes
    return (
        f"STATUS: VIOLATION. Catch request of {requested_tonnes:,.1f} tonnes exceeds "
        f"the ceiling of {max_safe_tonnes:,.1f} tonnes by {diff:,.1f} tonnes."
    )


# =======================================================
# 2. LangGraph Agent Pipeline
# =======================================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@st.cache_resource
def get_agent_graph():
    tools = [fetch_ices_stock_assessment, check_sea_conditions, calculate_catch_feasibility]

    client = ollama_client
    model_with_tools = client.bind_tools(tools)

    def call_model(state: AgentState):
        system_prompt = SystemMessage(
            content=(
                "You are an AI Fisheries Officer for Scandinavian marine management. "
                "Use the ICES REST API tool to inspect real stock status and the Marine API tool "
                "to evaluate maritime safety before issuing landing clearances. "
                "Provide direct, concise assessments."
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


# =======================================================
# 3. Streamlit Interface
# =======================================================

st.set_page_config(page_title="Nordic Fisheries Live Agent", page_icon="🌊", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": "Nordic Fisheries AI Officer online. Connected to ICES Stock Assessment & Open-Meteo Marine APIs. Ask about stock status, quota calculations, or offshore conditions."}
    ]

with st.sidebar:
    st.header("🌐 Connected Real-Time Endpoints")
    st.markdown("""
    * **ICES SAG REST API**  
      `sag.ices.dk` (Stock health & advisory data)
    * **Open-Meteo Marine API**  
      `marine-api.open-meteo.com` (Waves & currents)
    * **Local Model**  
      Ollama Gemma
    """)
    st.divider()
    if st.button("Clear Session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Session reset."}
        ]
        st.rerun()

st.title("🌊 Nordic Fisheries Real-Time Operations Agent")
st.caption("Live stock advice & marine conditions via LangGraph Tool Calling")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(
        "E.g., Check Barents Sea Cod (cod.27.1-2) status and sea conditions near Tromsø (69.65, 18.96)."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    graph = get_agent_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Calling live REST endpoints..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
            response_content = result["messages"][-1].content
            st.markdown(response_content)

    st.session_state.chat_history.append({"role": "assistant", "content": response_content})