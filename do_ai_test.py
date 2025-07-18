#!/usr/bin/env python3
"""
Standalone DigitalOcean AI Agent Test
Just run: python test_do_ai.py

No setup required - will prompt for credentials or use environment variables
"""

import os
import requests
import json
import time
from datetime import datetime

# =============================================================================
# CONFIGURATION - You can hardcode your values here if you want
# =============================================================================

# Option 1: Hardcode your values here (uncomment and fill in):
# GENAI_ENDPOINT = "https://tcxn3difq23zdxlph2heigba.agents.do-ai.run"
# GENAI_ACCESS_KEY = "your_key_here"

# Option 2: Leave blank to be prompted or use environment variables
GENAI_ENDPOINT = ""
GENAI_ACCESS_KEY = ""

# =============================================================================

def get_credentials():
    """Get credentials from environment, hardcoded values, or user input"""
    
    # Try hardcoded values first
    endpoint = GENAI_ENDPOINT
    api_key = GENAI_ACCESS_KEY
    
    # If not hardcoded, try environment variables
    if not endpoint:
        endpoint = os.getenv("GENAI_ENDPOINT", "")
    if not api_key:
        api_key = os.getenv("GENAI_ACCESS_KEY", "")
    
    # If still not found, prompt user
    if not endpoint:
        print("🔑 DigitalOcean AI Endpoint not found in environment")
        endpoint = input("Enter your GENAI_ENDPOINT: ").strip()
    
    if not api_key:
        print("🔑 DigitalOcean AI Access Key not found in environment")
        api_key = input("Enter your GENAI_ACCESS_KEY: ").strip()
    
    return endpoint, api_key

def test_digitalocean_ai():
    """Test DigitalOcean AI agent with various scenarios"""
    
    print("🚀 DigitalOcean AI Agent Standalone Test")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get credentials
    endpoint, api_key = get_credentials()
    
    if not endpoint or not api_key:
        print("❌ Missing credentials. Cannot proceed.")
        return False
    
    print(f"📍 Endpoint: {endpoint}")
    print(f"🔑 API Key: {'●' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '●' * len(api_key)} ({len(api_key)} chars)")
    print()
    
    # Test scenarios
    test_cases = [
        {
            "name": "🧪 Test 1: Minimal Request",
            "description": "Simplest possible test",
            "payload": {
                "messages": [{"role": "user", "content": "Say 'hello'"}],
                "temperature": 0.7,
                "max_tokens": 10
            },
            "timeout": 30
        },
        {
            "name": "🧪 Test 2: Basic Conversation", 
            "description": "Simple system + user message",
            "payload": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"}
                ],
                "temperature": 0.7,
                "max_tokens": 50
            },
            "timeout": 30
        },
        {
            "name": "🧪 Test 3: Call Center Context",
            "description": "Similar to your app's usage",
            "payload": {
                "messages": [
                    {"role": "system", "content": "You are an AI assistant for call center evaluation data analysis. Analyze customer service interactions and provide insights."},
                    {"role": "user", "content": "Analyze this call: Customer called about billing issue, agent resolved it quickly."}
                ],
                "temperature": 0.7,
                "max_tokens": 200
            },
            "timeout": 60
        },
        {
            "name": "🧪 Test 4: Large Context",
            "description": "Test with larger context like your app",
            "payload": {
                "messages": [
                    {"role": "system", "content": "You are an AI assistant for call center evaluation data analysis. " + "Context data: " + "Sample evaluation data. " * 100},
                    {"role": "user", "content": "Provide a summary of the call center performance."}
                ],
                "temperature": 0.7,
                "max_tokens": 300
            },
            "timeout": 120
        }
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    url = f"{endpoint.rstrip('/')}/api/v1/chat/completions"
    print(f"🎯 Testing URL: {url}")
    print()
    
    results = []
    
    for test_case in test_cases:
        print(test_case['name'])
        print(f"   📝 {test_case['description']}")
        print(f"   📦 Payload size: {len(json.dumps(test_case['payload']))} chars")
        print(f"   ⏱️  Timeout: {test_case['timeout']}s")
        
        start_time = time.time()
        
        try:
            print("   📤 Sending request...", end=" ", flush=True)
            
            response = requests.post(
                url,
                headers=headers,
                json=test_case['payload'],
                timeout=test_case['timeout']
            )
            
            elapsed = time.time() - start_time
            print(f"Done! ({elapsed:.2f}s)")
            
            result = {
                "name": test_case['name'],
                "success": False,
                "elapsed": elapsed,
                "status_code": response.status_code,
                "error": None,
                "response_preview": None
            }
            
            if response.ok:
                try:
                    json_result = response.json()
                    if "choices" in json_result and json_result["choices"]:
                        reply = json_result["choices"][0]["message"]["content"]
                        result["success"] = True
                        result["response_preview"] = reply[:100] + ("..." if len(reply) > 100 else "")
                        print(f"   ✅ SUCCESS: '{result['response_preview']}'")
                    else:
                        result["error"] = f"Unexpected response format: {json.dumps(json_result)[:100]}"
                        print(f"   ⚠️  Warning: {result['error']}")
                except json.JSONDecodeError as e:
                    result["error"] = f"Invalid JSON: {response.text[:100]}"
                    print(f"   ❌ JSON Error: {result['error']}")
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text[:100]}"
                print(f"   ❌ HTTP Error: {result['error']}")
                
        except requests.exceptions.ConnectTimeout:
            elapsed = time.time() - start_time
            result = {
                "name": test_case['name'],
                "success": False,
                "elapsed": elapsed,
                "error": f"Connection timeout to {url}"
            }
            print(f"Timeout! ({elapsed:.2f}s)")
            print(f"   ❌ CONNECTION TIMEOUT: Cannot connect to the endpoint")
            
        except requests.exceptions.ReadTimeout:
            elapsed = time.time() - start_time
            result = {
                "name": test_case['name'],
                "success": False,
                "elapsed": elapsed,
                "error": f"Read timeout after {test_case['timeout']}s"
            }
            print(f"Timeout! ({elapsed:.2f}s)")
            print(f"   ❌ READ TIMEOUT: No response within {test_case['timeout']}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "name": test_case['name'],
                "success": False,
                "elapsed": elapsed,
                "error": str(e)
            }
            print(f"Error! ({elapsed:.2f}s)")
            print(f"   ❌ ERROR: {str(e)}")
        
        results.append(result)
        print()
    
    # Summary Report
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    print()
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("   Your DigitalOcean AI agent is working correctly.")
        print("   The issue is likely in your application code or routing.")
    elif passed == 0:
        print("💥 ALL TESTS FAILED!")
        print("   Possible issues:")
        print("   • DigitalOcean AI service is down")
        print("   • Wrong API endpoint URL")
        print("   • Invalid API key")
        print("   • Network/firewall blocking connection")
    else:
        print("⚠️  PARTIAL SUCCESS!")
        print("   Some tests passed, some failed.")
        print("   This suggests the service works but has limits.")
    
    print("\n📋 Detailed Results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"   {status} {result['name']} ({result['elapsed']:.2f}s)")
        if result.get('error'):
            print(f"      Error: {result['error']}")
    
    print(f"\n🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed == total

def main():
    """Main function"""
    try:
        success = test_digitalocean_ai()
        exit_code = 0 if success else 1
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        exit(1)

if __name__ == "__main__":
    main()