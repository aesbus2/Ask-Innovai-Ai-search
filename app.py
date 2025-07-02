# Endpoint-Specific Debug App.py
# Version: 2.8.0 - Debug specifically for innovai-6abj.onrender.com/api/content
# Helps identify why we're getting HTML instead of JSON from the correct endpoint

import os
import logging
import requests
import asyncio
import json
import sys
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from uuid import uuid4

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ask-innovai")

# Load environment variables
load_dotenv()

# Import your modules with error handling
try:
    from sentence_splitter import split_into_chunks
    logger.info("✅ sentence_splitter imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import sentence_splitter: {e}")
    sys.exit(1)

try:
    from opensearch_client import search_opensearch, index_document
    logger.info("✅ opensearch_client imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import opensearch_client: {e}")
    sys.exit(1)

# Import embedder with fallback
EMBEDDER_AVAILABLE = False
try:
    from embedder import embed_text, get_embedding_stats, preload_embedding_model
    EMBEDDER_AVAILABLE = True
    logger.info("✅ embedder imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ embedder import failed: {e} - will run without embeddings")

# Create FastAPI app
app = FastAPI(
    title="Ask InnovAI Endpoint Debug",
    description="Debug specifically for innovai-6abj.onrender.com/api/content",
    version="2.8.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with error handling
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Static files mounted")
except Exception as e:
    logger.warning(f"⚠️ Failed to mount static files: {e}")

# Configuration
GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT", "https://tcxn3difq23zdxlph2heigba.agents.do-ai.run")
GENAI_ACCESS_KEY = os.getenv("GENAI_ACCESS_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://innovai-6abj.onrender.com/api/content")  # Default to your endpoint
API_AUTH_KEY = os.getenv("API_AUTH_KEY", "Authorization")
API_AUTH_VALUE = os.getenv("API_AUTH_VALUE")

logger.info(f"🔧 Endpoint Debug Configuration:")
logger.info(f"   API_BASE_URL: {API_BASE_URL}")
logger.info(f"   API_AUTH_KEY: {API_AUTH_KEY}")
logger.info(f"   API_AUTH_VALUE: {'✅ Set (' + str(len(API_AUTH_VALUE)) + ' chars)' if API_AUTH_VALUE else '❌ Missing'}")

# Create a session for consistent testing
session = requests.Session()

# Simple import status tracking
import_status = {
    "status": "idle",
    "start_time": None,
    "end_time": None,
    "current_step": None,
    "results": {},
    "error": None
}

# Enhanced logging for debugging
debug_logs = []

def log_debug(message: str):
    """Add message to debug logs"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    debug_logs.append(log_entry)
    logger.info(message)
    if len(debug_logs) > 300:
        debug_logs.pop(0)

def update_import_status(status: str, step: str = None, results: dict = None, error: str = None):
    """Update import status"""
    import_status["status"] = status
    import_status["current_step"] = step
    if results:
        import_status["results"] = results
    if error:
        import_status["error"] = error
    if status == "running" and not import_status["start_time"]:
        import_status["start_time"] = datetime.now().isoformat()
    elif status in ["completed", "failed"]:
        import_status["end_time"] = datetime.now().isoformat()

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    history: list
    programs: list = []

class ImportRequest(BaseModel):
    collection: str = "all"
    max_docs: Optional[int] = None
    import_type: str = "full"

# ============================================================================
# COMPREHENSIVE ENDPOINT DEBUGGING FUNCTIONS
# ============================================================================

def test_endpoint_with_method(url: str, method: str, headers: dict, params: dict = None, timeout: int = 30) -> dict:
    """Test endpoint with specific method and comprehensive logging"""
    
    log_debug(f"🧪 TESTING: {method} {url}")
    log_debug(f"   Headers: {json.dumps(headers, indent=2)}")
    log_debug(f"   Params: {params}")
    log_debug(f"   Timeout: {timeout}s")
    
    try:
        # Make the request
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=timeout, allow_redirects=False)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, params=params, json={}, timeout=timeout, allow_redirects=False)
        else:
            return {"success": False, "error": f"Unsupported method: {method}"}
        
        # Log response details
        log_debug(f"📊 RESPONSE:")
        log_debug(f"   Status: {response.status_code} {response.reason}")
        log_debug(f"   Headers: {dict(response.headers)}")
        log_debug(f"   Content-Length: {len(response.content)} bytes")
        log_debug(f"   Content-Type: {response.headers.get('content-type', 'Not set')}")
        log_debug(f"   Location: {response.headers.get('location', 'Not set')}")
        
        # Handle redirects manually to see what's happening
        if response.status_code in [301, 302, 303, 307, 308]:
            redirect_url = response.headers.get('location', 'No location header')
            log_debug(f"🔄 REDIRECT DETECTED: {response.status_code} -> {redirect_url}")
            
            # Follow redirect manually to see what we get
            if redirect_url and redirect_url.startswith('http'):
                log_debug(f"🔄 Following redirect to: {redirect_url}")
                redirect_response = requests.get(redirect_url, headers=headers, timeout=timeout)
                log_debug(f"🔄 Redirect response: {redirect_response.status_code}")
                log_debug(f"🔄 Redirect content preview: {redirect_response.text[:200]}")
                
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "redirect_url": redirect_url,
                    "redirect_status": redirect_response.status_code,
                    "redirect_content_preview": redirect_response.text[:500],
                    "error": f"Request redirected to {redirect_url}"
                }
        
        # Analyze response content
        response_text = response.text
        
        # Check if empty
        if not response_text.strip():
            log_debug(f"❌ EMPTY RESPONSE")
            return {
                "success": False,
                "status_code": response.status_code,
                "error": "Empty response",
                "headers": dict(response.headers)
            }
        
        # Log response preview
        log_debug(f"📄 RESPONSE PREVIEW (first 300 chars):")
        log_debug(f"   {response_text[:300]}")
        
        # Detect response type
        if response_text.strip().startswith("<!DOCTYPE") or response_text.strip().startswith("<html"):
            log_debug(f"🌐 DETECTED: HTML Response")
            
            # Extract useful info from HTML
            soup = BeautifulSoup(response_text, "html.parser")
            title = soup.find("title")
            title_text = title.get_text() if title else "No title"
            
            # Look for error messages
            error_msgs = []
            for tag in soup.find_all(text=True):
                text = tag.strip().lower()
                if any(keyword in text for keyword in ['error', 'unauthorized', 'forbidden', 'not found', 'invalid']):
                    error_msgs.append(tag.strip()[:100])
            
            # Look for login forms
            forms = soup.find_all("form")
            login_indicators = soup.find_all(text=lambda text: text and 'login' in text.lower())
            
            return {
                "success": False,
                "status_code": response.status_code,
                "response_type": "html",
                "html_title": title_text,
                "error_messages": error_msgs[:3],  # First 3 error messages
                "has_login_form": len(forms) > 0,
                "login_indicators": len(login_indicators) > 0,
                "content_preview": response_text[:800],
                "error": f"Received HTML page '{title_text}' instead of JSON"
            }
        
        elif response_text.strip().startswith("{") or response_text.strip().startswith("["):
            log_debug(f"✅ DETECTED: JSON Response")
            
            try:
                json_data = json.loads(response_text)
                
                # Analyze JSON structure
                analysis = {"json_valid": True}
                
                if isinstance(json_data, dict):
                    analysis["type"] = "object"
                    analysis["keys"] = list(json_data.keys())
                    
                    if "evaluations" in json_data:
                        evaluations = json_data["evaluations"]
                        analysis["evaluations_count"] = len(evaluations) if isinstance(evaluations, list) else "Not a list"
                        
                        if isinstance(evaluations, list) and len(evaluations) > 0:
                            sample = evaluations[0]
                            analysis["sample_evaluation"] = {
                                "has_internalId": "internalId" in sample,
                                "has_evaluationId": "evaluationId" in sample,
                                "has_evaluation": "evaluation" in sample,
                                "has_transcript": "transcript" in sample,
                                "keys": list(sample.keys()) if isinstance(sample, dict) else "Not an object"
                            }
                
                elif isinstance(json_data, list):
                    analysis["type"] = "array"
                    analysis["length"] = len(json_data)
                    
                    if len(json_data) > 0:
                        sample = json_data[0]
                        analysis["sample_item"] = {
                            "type": type(sample).__name__,
                            "keys": list(sample.keys()) if isinstance(sample, dict) else "Not an object"
                        }
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "response_type": "json",
                    "json_analysis": analysis,
                    "json_data": json_data
                }
                
            except json.JSONDecodeError as e:
                log_debug(f"❌ JSON PARSE ERROR: {e}")
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "response_type": "invalid_json",
                    "json_error": str(e),
                    "content_preview": response_text[:500],
                    "error": f"Invalid JSON: {e}"
                }
        
        else:
            log_debug(f"❓ UNKNOWN RESPONSE TYPE")
            return {
                "success": False,
                "status_code": response.status_code,
                "response_type": "unknown",
                "content_preview": response_text[:500],
                "error": "Unknown response format"
            }
            
    except requests.exceptions.ConnectionError as e:
        log_debug(f"❌ CONNECTION ERROR: {e}")
        return {"success": False, "error": f"Connection failed: {e}"}
    
    except requests.exceptions.Timeout as e:
        log_debug(f"⏰ TIMEOUT ERROR: {e}")
        return {"success": False, "error": f"Request timeout: {e}"}
    
    except Exception as e:
        log_debug(f"❌ UNEXPECTED ERROR: {e}")
        return {"success": False, "error": f"Unexpected error: {e}"}

# ============================================================================
# SPECIALIZED DEBUG ENDPOINTS
# ============================================================================

@app.get("/debug/test-your-endpoint")
async def test_your_endpoint():
    """Test your specific endpoint with multiple authentication methods"""
    
    log_debug("🎯 TESTING YOUR SPECIFIC ENDPOINT")
    log_debug(f"   URL: {API_BASE_URL}")
    
    if not API_AUTH_VALUE:
        return {
            "status": "error",
            "error": "API_AUTH_VALUE not configured",
            "your_endpoint": API_BASE_URL
        }
    
    results = {
        "endpoint": API_BASE_URL,
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test different authentication methods with your endpoint
    auth_methods = [
        ("current", {API_AUTH_KEY: API_AUTH_VALUE}),
        ("bearer", {"Authorization": f"Bearer {API_AUTH_VALUE}"}),
        ("api_key", {"X-API-Key": API_AUTH_VALUE}),
        ("basic", {"Authorization": f"Basic {API_AUTH_VALUE}"}),
        ("token", {"X-Auth-Token": API_AUTH_VALUE}),
        ("no_auth", {})
    ]
    
    successful_method = None
    
    for method_name, auth_headers in auth_methods:
        log_debug(f"🧪 Testing {method_name} authentication...")
        
        # Base headers
        headers = {
            "Accept": "application/json",
            "User-Agent": "Ask-InnovAI/2.8.0",
            "Cache-Control": "no-cache"
        }
        headers.update(auth_headers)
        
        # Test with GET
        result = test_endpoint_with_method(API_BASE_URL, "GET", headers)
        results["tests"][f"{method_name}_get"] = result
        
        if result.get("success"):
            successful_method = (method_name, headers, "GET")
            log_debug(f"✅ SUCCESS: {method_name} GET worked!")
            break
        
        # If GET failed, try POST
        result_post = test_endpoint_with_method(API_BASE_URL, "POST", headers)
        results["tests"][f"{method_name}_post"] = result_post
        
        if result_post.get("success"):
            successful_method = (method_name, headers, "POST")
            log_debug(f"✅ SUCCESS: {method_name} POST worked!")
            break
    
    # Summary
    if successful_method:
        method_name, headers, http_method = successful_method
        results["recommendation"] = {
            "use_method": method_name,
            "use_headers": headers,
            "use_http_method": http_method,
            "message": f"Use {method_name} authentication with {http_method} method"
        }
    else:
        results["recommendation"] = {
            "message": "No authentication method worked",
            "possible_issues": [
                "API endpoint URL is incorrect",
                "API requires different authentication",
                "API is blocking requests from this server",
                "API expects different headers or parameters"
            ]
        }
    
    return results

@app.get("/debug/compare-browsers")
async def compare_with_browser():
    """Compare our request with what a browser would send"""
    
    if not API_AUTH_VALUE:
        return {"error": "API_AUTH_VALUE not configured"}
    
    log_debug("🌐 COMPARING WITH BROWSER REQUEST")
    
    # Test 1: Our current method
    current_headers = {
        API_AUTH_KEY: API_AUTH_VALUE,
        "Accept": "application/json",
        "User-Agent": "Ask-InnovAI/2.8.0"
    }
    
    current_result = test_endpoint_with_method(API_BASE_URL, "GET", current_headers)
    
    # Test 2: Full browser headers
    browser_headers = {
        API_AUTH_KEY: API_AUTH_VALUE,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Cache-Control": "max-age=0"
    }
    
    browser_result = test_endpoint_with_method(API_BASE_URL, "GET", browser_headers)
    
    # Test 3: Minimal headers
    minimal_headers = {
        API_AUTH_KEY: API_AUTH_VALUE
    }
    
    minimal_result = test_endpoint_with_method(API_BASE_URL, "GET", minimal_headers)
    
    return {
        "endpoint": API_BASE_URL,
        "comparison": {
            "current_method": {
                "headers": current_headers,
                "result": current_result
            },
            "browser_style": {
                "headers": browser_headers,
                "result": browser_result
            },
            "minimal": {
                "headers": minimal_headers,
                "result": minimal_result
            }
        },
        "analysis": {
            "current_works": current_result.get("success", False),
            "browser_works": browser_result.get("success", False),
            "minimal_works": minimal_result.get("success", False)
        }
    }

@app.get("/debug/test-url-variations")
async def test_url_variations():
    """Test variations of your endpoint URL"""
    
    if not API_AUTH_VALUE:
        return {"error": "API_AUTH_VALUE not configured"}
    
    base_url = "https://innovai-6abj.onrender.com"
    
    # Different endpoint variations to test
    url_variations = [
        f"{base_url}/api/content",  # Your current URL
        f"{base_url}/api/evaluations",
        f"{base_url}/api/v1/content",
        f"{base_url}/api/v1/evaluations",
        f"{base_url}/content",
        f"{base_url}/evaluations",
        f"{base_url}/api/content/",  # With trailing slash
        f"{base_url}/api/content?format=json",  # With format parameter
    ]
    
    headers = {
        API_AUTH_KEY: API_AUTH_VALUE,
        "Accept": "application/json",
        "User-Agent": "Ask-InnovAI/2.8.0"
    }
    
    results = {}
    
    for url in url_variations:
        log_debug(f"🧪 Testing URL: {url}")
        result = test_endpoint_with_method(url, "GET", headers)
        results[url] = {
            "success": result.get("success", False),
            "status_code": result.get("status_code"),
            "response_type": result.get("response_type"),
            "error": result.get("error")
        }
        
        if result.get("success"):
            results[url]["evaluations_found"] = "Yes" if result.get("json_analysis", {}).get("evaluations_count", 0) else "No"
    
    return {
        "base_url": base_url,
        "your_current_url": API_BASE_URL,
        "url_test_results": results,
        "working_urls": [url for url, result in results.items() if result["success"]]
    }

@app.get("/debug/logs")
async def get_debug_logs():
    """Get detailed debug logs"""
    return {
        "logs": debug_logs,
        "log_count": len(debug_logs),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/debug/clear-logs")
async def clear_debug_logs():
    """Clear debug logs"""
    global debug_logs
    debug_logs = []
    return {"status": "success", "message": "Debug logs cleared"}

# ============================================================================
# BASIC ENDPOINTS
# ============================================================================

@app.get("/ping")
async def ping():
    return {"status": "ok", "timestamp": datetime.now().isoformat(), "version": "2.8.0_endpoint_debug"}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content=f"""
        <html><body>
        <h1>🤖 Ask InnovAI Endpoint Debug</h1>
        <p>Debug interface for your specific endpoint: <code>{API_BASE_URL}</code></p>
        <h2>Debug Endpoints:</h2>
        <ul>
        <li><a href="/debug/test-your-endpoint">/debug/test-your-endpoint</a> - Test your specific endpoint with all auth methods</li>
        <li><a href="/debug/compare-browsers">/debug/compare-browsers</a> - Compare with browser requests</li>
        <li><a href="/debug/test-url-variations">/debug/test-url-variations</a> - Test different URL variations</li>
        <li><a href="/debug/logs">/debug/logs</a> - View detailed debug logs</li>
        </ul>
        <h2>Instructions:</h2>
        <ol>
        <li>Run <strong>/debug/test-your-endpoint</strong> first</li>
        <li>Check results to see which authentication method works</li>
        <li>If none work, try <strong>/debug/test-url-variations</strong></li>
        <li>Check <strong>/debug/logs</strong> for detailed request/response info</li>
        </ol>
        <p><strong>Your current endpoint:</strong> {API_BASE_URL}</p>
        </body></html>
        """)

@app.get("/status")
async def get_import_status():
    return import_status

# Simplified import for testing
async def run_import_debug(collection: str = "all", max_docs: int = None):
    """Debug import using the working authentication method"""
    try:
        update_import_status("running", "Starting endpoint-specific debug import")
        log_debug("🚀 STARTING ENDPOINT DEBUG IMPORT")
        
        # First, find working authentication method
        log_debug("🔍 Finding working authentication method...")
        
        auth_methods = [
            ("current", {API_AUTH_KEY: API_AUTH_VALUE}),
            ("bearer", {"Authorization": f"Bearer {API_AUTH_VALUE}"}),
            ("api_key", {"X-API-Key": API_AUTH_VALUE})
        ]
        
        working_method = None
        
        for method_name, auth_headers in auth_methods:
            headers = {
                "Accept": "application/json",
                "User-Agent": "Ask-InnovAI/2.8.0"
            }
            headers.update(auth_headers)
            
            result = test_endpoint_with_method(API_BASE_URL, "GET", headers, {"limit": 1} if max_docs else None)
            
            if result.get("success") and result.get("json_data"):
                working_method = (method_name, headers)
                log_debug(f"✅ Found working method: {method_name}")
                break
        
        if not working_method:
            error_msg = "No authentication method worked for your endpoint"
            log_debug(f"❌ {error_msg}")
            update_import_status("failed", error=error_msg)
            return
        
        method_name, headers = working_method
        
        # Use working method to fetch data
        log_debug(f"📡 Fetching data using {method_name} method...")
        params = {"limit": max_docs} if max_docs else {}
        
        result = test_endpoint_with_method(API_BASE_URL, "GET", headers, params)
        
        if not result.get("success"):
            error_msg = f"Data fetch failed: {result.get('error', 'Unknown error')}"
            log_debug(f"❌ {error_msg}")
            update_import_status("failed", error=error_msg)
            return
        
        json_data = result.get("json_data", {})
        evaluations = json_data.get("evaluations", [])
        
        if isinstance(json_data, list):
            evaluations = json_data
        
        log_debug(f"📊 Successfully retrieved {len(evaluations)} evaluations")
        
        # Process a few for testing
        test_count = min(2, len(evaluations))
        log_debug(f"🧪 Processing {test_count} evaluations for testing")
        
        for i, evaluation in enumerate(evaluations[:test_count]):
            log_debug(f"📄 Evaluation {i+1}:")
            log_debug(f"   Internal ID: {evaluation.get('internalId', 'missing')}")
            log_debug(f"   Evaluation ID: {evaluation.get('evaluationId', 'missing')}")
            log_debug(f"   Agent: {evaluation.get('agentName', 'missing')}")
            log_debug(f"   Has evaluation text: {bool(evaluation.get('evaluation'))}")
            log_debug(f"   Has transcript: {bool(evaluation.get('transcript'))}")
        
        results = {
            "total_available": len(evaluations),
            "authentication_method": method_name,
            "endpoint_working": True,
            "sample_data": evaluations[0] if evaluations else None,
            "debug_mode": True,
            "completed_at": datetime.now().isoformat()
        }
        
        log_debug(f"🎉 Endpoint debug completed successfully!")
        log_debug(f"   Working auth method: {method_name}")
        log_debug(f"   Evaluations available: {len(evaluations)}")
        
        update_import_status("completed", results=results)
        
    except Exception as e:
        error_msg = f"Endpoint debug failed: {str(e)}"
        log_debug(f"❌ {error_msg}")
        update_import_status("failed", error=error_msg)

@app.post("/import")
async def start_import(request: ImportRequest, background_tasks: BackgroundTasks):
    """Start endpoint debug import"""
    if import_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Import already running")
    
    # Reset status
    import_status.update({
        "status": "idle",
        "start_time": None,
        "end_time": None,
        "current_step": None,
        "results": {},
        "error": None
    })
    
    # Start debug import
    background_tasks.add_task(run_import_debug, request.collection, request.max_docs)
    return {"status": "success", "message": "Endpoint debug import started"}

# For Digital Ocean App Platform
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    logger.info(f"🚀 Starting Ask InnovAI Endpoint Debug on port {port}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )