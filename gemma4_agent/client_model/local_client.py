import json
import httpx
from openai import OpenAI

# Point to Docker Model Runner's OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed-locally"
)