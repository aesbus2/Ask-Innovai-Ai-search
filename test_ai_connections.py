#!/usr/bin/env python3
"""
Digital Ocean AI Agent Test Script - Correct Format
Based on DO documentation using OpenAI client library
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_do_ai_agent():
    """Test Digital Ocean AI Agent using correct format"""
    print("🤖 Digital Ocean AI Agent Test")
    print("=" * 40)
    
    # Get environment variables (check both naming conventions)
    agent_endpoint = (
        os.getenv("agent_endpoint") or 
        os.getenv("GENAI_ENDPOINT") or 
        ""
    )
    
    agent_access_key = (
        os.getenv("agent_access_key") or 
        os.getenv("GENAI_ACCESS_KEY") or 
        ""
    )
    
    print(f"Endpoint: {agent_endpoint}")
    print(f"Access Key: {agent_access_key[:8]}...{agent_access_key[-4:] if len(agent_access_key) > 12 else '***'}")
    
    if not agent_endpoint or not agent_access_key:
        print("❌ Missing environment variables:")
        print("   Need: agent_endpoint and agent_access_key")
        print("   Or: GENAI_ENDPOINT and GENAI_ACCESS_KEY")
        return False
    
    # Ensure endpoint has the correct format
    if not agent_endpoint.endswith("/api/v1/"):
        if agent_endpoint.endswith("/"):
            agent_endpoint = agent_endpoint + "api/v1/"
        else:
            agent_endpoint = agent_endpoint + "/api/v1/"
    
    print(f"Full endpoint: {agent_endpoint}")
    print()
    
    try:
        # Create OpenAI client with DO endpoint
        print("🔄 Creating OpenAI client...")
        client = OpenAI(
            base_url=agent_endpoint,
            api_key=agent_access_key,
        )
        print("✅ Client created successfully")
        
        # Test basic completion
        print("\n🔄 Testing basic completion...")

        response = client.completions.create(
            model="n/a",
            prompt="Say 'Hello from Digital Ocean AI' and nothing else.",
            max_tokens=50,
            temperature=0.1
        )
        
        if response.choices:    
            print("✅ Basic completion successful!")
            print(f"Response: {response.choices[0].message.content}")
            
            # Test with retrieval info (DO specific feature)
            print("\n🔄 Testing with retrieval info...")
            response_with_retrieval = client.chat.completions.create(
                model="n/a",
                messages=[{
                    "role": "user", 
                    "content": "What is Metro by T-Mobile?"
                }],
                extra_body={"include_retrieval_info": True}
            )
            
            print("✅ Retrieval completion successful!")
            for choice in response_with_retrieval.choices:
                print(f"Content: {choice.message.content}")
            
            # Print retrieval info if available
            response_dict = response_with_retrieval.to_dict()
            if "retrieval" in response_dict:
                print("\n📚 Retrieval Information:")
                print(json.dumps(response_dict["retrieval"], indent=2))
            
            return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Common error diagnostics
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n🔍 Authentication Error Diagnosis:")
            print("- Check that agent_access_key is correct")
            print("- Verify the key hasn't expired")
            print("- Make sure the agent is active in DO dashboard")
        
        elif "404" in str(e) or "Not Found" in str(e):
            print("\n🔍 Endpoint Error Diagnosis:")
            print("- Check that agent_endpoint URL is correct")
            print("- Verify the endpoint format: https://xxx.agents.do-ai.run")
            print("- Make sure '/api/v1/' is appended")
        
        elif "timeout" in str(e).lower():
            print("\n🔍 Timeout Error Diagnosis:")
            print("- The agent might be slow to respond")
            print("- Try increasing timeout or try again later")
        
        return False

def test_environment_setup():
    """Test environment variable setup"""
    print("🔧 Environment Setup Test")
    print("=" * 30)
    
    # Check for both naming conventions
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

def create_sample_env_file():
    """Create a sample .env file"""
    env_content = """# Digital Ocean AI Agent Configuration
# Use either the DO format or our format (not both)

# DO Documentation Format:
agent_endpoint=https://your-agent-id.agents.do-ai.run
agent_access_key=your-access-key-here

# Our Format (alternative):
# GENAI_ENDPOINT=https://your-agent-id.agents.do-ai.run
# GENAI_ACCESS_KEY=your-access-key-here
# GENAI_MODEL=n/a
"""
    
    try:
        with open('.env.sample', 'w') as f:
            f.write(env_content)
        print("📄 Created .env.sample file with correct format")
    except Exception as e:
        print(f"Could not create .env.sample: {e}")

if __name__ == "__main__":
    print("🚀 Digital Ocean AI Agent Test Suite")
    print("=" * 50)
    
    # Test environment setup
    env_ok = test_environment_setup()
    
    if not env_ok:
        create_sample_env_file()
        print("\n❌ Environment not properly configured")
        exit(1)
    
    # Test the actual AI agent
    success = test_do_ai_agent()
    
    if success:
        print("\n🎉 SUCCESS! Your Digital Ocean AI Agent is working correctly.")
        print("\n📋 Key findings:")
        print("- Use OpenAI client library (not direct HTTP)")
        print("- Model should be 'n/a'")
        print("- Endpoint needs '/api/v1/' suffix")
        print("- Can use extra_body for retrieval info")
    else:
        print("\n🔧 FAILED! Check the error messages above.")
        print("\n📋 Common fixes:")
        print("1. Verify your agent endpoint URL")
        print("2. Check your access key")
        print("3. Make sure the agent is active in DO dashboard")
        print("4. Install required library: pip install openai")
    
    print("\n" + "="*50)