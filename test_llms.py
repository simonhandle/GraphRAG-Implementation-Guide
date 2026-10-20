import requests
import json
import os
from dotenv import load_dotenv
import openai

load_dotenv()

API_KEY = os.getenv("GRAPHRAG_API_KEY")
MODEL = "liquid/lfm-2.5-2.6b:free"

LOCAL_URL = "http://127.0.0.1:8080"
LOCAL_MODEL = "models/Qwen3-4B-Q4_K_M.gguf"
LOCAL_API = os.getenv("LOCAL_KEY")

print("Starting Model Testing ...")

response_liquid = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    },
    data=json.dumps({
        "model":MODEL,
        "messages": [
        {
          "role": "user",
          "content": "Hi there!"
        }
        ],
        "reasoning": {"enabled": True}
    })
)
response_liquid = response_liquid.json()

local_client = openai.OpenAI(base_url=LOCAL_URL, api_key=LOCAL_API)
response_qwen = local_client.chat.completions.create(
    model = LOCAL_MODEL,
    messages=[
        {
          "role": "user",
          "content": "Hi there!"
        }
        ]
)
response_qwen = response_qwen.model_dump()

print(f"Liquid Answer: {response_liquid['choices'][0]['message']['content']}\n"
      f"Qwen (Local) Anwer: {response_qwen['choices'][0]['message']['content']}")



