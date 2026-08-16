from openai import OpenAI
from langchain_openai import ChatOpenAI

CONFIG={
"model":"gemma4:e4b",
"base_url":"http://localhost:11434/v1",
"api_key":"ollama", # Ollama ignores this, but LangChain requires a non-empty string
"temperature":"0.7"
}

ollama_open_ai_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" # required but not used
)

ollama_client = ChatOpenAI(
    model=CONFIG["model"],
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
    temperature=0.7
)