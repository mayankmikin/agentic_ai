import sys
import streamlit as st
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Gemma AI Chatbot", page_icon="🤖")
st.title("🤖 Gemma Chatbot (via Docker Model Runner)")

CONFIG={
"model":"gemma4:e4b",
"base_url":"http://localhost:11434/v1",
"api_key":"ollama", # Ollama ignores this, but LangChain requires a non-empty string
"temperature":"0.7"
}

# 1. Initialize LangChain model pointing to Docker Model Runner container
# Default Docker Model Runner API port is usually 12434 or 8000 depending on your runner configuration
@st.cache_resource
def load_llm():
    return  ChatOpenAI(
    model=CONFIG["model"],
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
    temperature=0.7
)

llm = load_llm()

# 2. Maintain Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# 3. Handle User Input
if user_input := st.chat_input("Type your message here..."):
    # Append human message and show immediately
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Pass the full chat message history payload to LangChain
                response = llm.invoke(st.session_state.messages)
                st.write(response.content)
                # Store assistant response
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"Error communicating with Model Runner: {e}")