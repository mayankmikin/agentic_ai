import json
import httpx
from openai import OpenAI

# Point to Docker Model Runner's OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" # required but not used
)

messages = [
    {"role": "system", "content": "what is the capital of norway and what is the temperature there right now " }]

# Trigger Gemma 4's reasoning mode natively
response = client.chat.completions.create(
    model="gemma4:E4B",
    messages=messages,
    temperature=1.0, # Recommended sampling for Gemma 4
    extra_body={"chat-template-kwargs": {"enable_thinking": True}}
)

response_text = response.choices[0].message.content
print(response_text)