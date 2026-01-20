import requests
import json

# সরাসরি Ollama-র ঠিকানায় রিকোয়েস্ট পাঠানো
url = "http://127.0.0.1:11434/api/generate"

data = {
    "model": "llama3",
    "prompt": "Hello, are you working?",
    "stream": False
}

print("⏳ Testing Raw Connection to Ollama...")

try:
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print("✅ Success! Ollama responded:")
        print(response.json()['response'])
    else:
        print(f"⚠️ Connected but Error Code: {response.status_code}")
        print(response.text)

except Exception as e:
    print("❌ Connection Failed completely!")
    print(f"Error: {e}")