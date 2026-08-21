from langchain.agents import create_agent
from langchain.tools import tool
# Import the OpenAI integration package
from langchain_openai import ChatOpenAI

dmr_client = ChatOpenAI(
    model="docker.io/gemma4:E4B",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker"  # DMR ignores this, but LangChain requires a non-empty string
)