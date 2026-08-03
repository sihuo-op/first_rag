import os
from dotenv import load_dotenv
import httpx

load_dotenv()

api_key = os.getenv("CHAT_API_KEY")
api_base = os.getenv("CHAT_API_BASE")

print(f"API Key: {api_key[:20]}..." if api_key else "No API Key")
print(f"API Base: {api_base}")

# Test API connection
try:
    client = httpx.Client(timeout=30)
    response = client.get(f"{api_base}/models")
    print(f"API Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Connection error: {e}")
