#!/usr/bin/env python3
"""
Digital Ocean AI Agent + FastAPI /api/chat Endpoint Test Script
"""

import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment_setup():
    print("🔧 Environment Setup Test")
    print("=" * 30)

    vars_to_check = [
        ("agent_endpoint", "GENAI_ENDPOINT"),
        ("agent_access_key", "GENAI_ACCESS_KEY")
    ]

    all_good = True

    for do_name, our_name in vars_to_check:
        do_val = os.getenv(do_name)
        our_val = os.getenv(our_name)

        if do_val:
            print(f"✅ {do_name}: Set (DO format)")
        elif our_val:
            print(f"✅ {our_name}: Set (our format)")
        else:
            print(f"❌ Missing: {do_name} or {our_name}")
            all_good = False

    if not all_good:
        print("\n💡 Set environment variables:")
        print("Option 1 (DO format):")
        print("  export agent_endpoint='https://your-agent.agents.do-ai.run'")
        print("  export agent_access_key='your-access-key'")
        print("\nOption 2 (Our format):")
        print("  export GENAI_ENDPOINT='https://your-agent.agents.do-ai.run'")
        print("  export GENAI_ACCESS_KEY='your-access-key'")

    return all_good

def test_do_ai_agent():
    print("\n🤖 Digital Ocean GenAI Agent Test")
    print("=" * 40)

    endpoint = os.getenv("agent_endpoint") or os.getenv("GENAI_ENDPOINT")
    access_key = os.getenv("agent_access_key") or os.getenv("GENAI_ACCESS_KEY")

    if not endpoint or not access_key:
        print("❌ Missing GENAI endpoint or access key")
        return False

    try:
        client = OpenAI(base_url=endpoint, api_key=access_key)
        print("✅ OpenAI client created")

        response = client.chat.completions.create(
            model="n/a",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=50,
            temperature=0.2
        )

        if response.choices:
            print("✅ GenAI response received")
            print("🔹", response.choices[0].message.content)
            return True
        else:
            print("❌ No response from GenAI")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_fastapi_chat_endpoint():
    print("\n🧪 Testing FastAPI /api/chat route")
    print("=" * 40)

    url = os.getenv("LOCAL_CHAT_URL", "http://localhost:8000/api/chat")
    payload = {
        "message": "Test message from DO console",
        "history": [],
        "filters": {},
        "analytics": True,
        "metadata_focus": [],
        "programs": []
    }

    try:
        print(f"📡 POST {url}")
        response = requests.post(url, json=payload, timeout=10)
        print(f"🔁 Status: {response.status_code}")

        if response.ok:
            data = response.json()
            print("✅ Reply:", data.get("reply", "(missing)"))
            return True
        else:
            print("❌ Error response:")
            print(response.text)
            return False

    except Exception as e:
        print(f"❌ Exception during test: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Full AI System Test Suite")
    print("=" * 50)

    env_ok = test_environment_setup()
    if not env_ok:
        print("\n❌ Environment misconfigured. Aborting tests.")
        exit(1)

    success_genai = test_do_ai_agent()
    if not success_genai:
        print("\n⚠️ GenAI Agent Test FAILED")

    success_chat = test_fastapi_chat_endpoint()
    if not success_chat:
        print("\n⚠️ FastAPI /api/chat Test FAILED")

    if success_genai and success_chat:
        print("\n🎉 All tests passed! System is fully operational.")
    print("=" * 50)
