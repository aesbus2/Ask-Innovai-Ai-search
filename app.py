# Robust app.py for Digital Ocean App Platform
# Version: 2.1.0 - Enhanced startup and error handling

import os
import logging
import requests
import asyncio
import json
import sys
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
    from embedder import embed_text, get_embedding_stats
    EMBEDDER_AVAILABLE = True
    logger.info("✅ embedder imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ embedder import failed: {e} - will run without embeddings")

# Create FastAPI app
app = FastAPI(
    title="Ask InnovAI",
    description="AI-Powered Knowledge Assistant",
    version="2.1.0"
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

# Configuration with validation
GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT", "https://api.digitalocean.com/v1/ai/chat")
GENAI_ACCESS_KEY = os.getenv("GENAI_ACCESS_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "")
API_AUTH_KEY = os.getenv("API_AUTH_KEY", "Authorization")
API_AUTH_VALUE = os.getenv("API_AUTH_VALUE", "")

logger.info(f"🔧 Configuration loaded:")
logger.info(f"   GENAI_ENDPOINT: {GENAI_ENDPOINT}")
logger.info(f"   GENAI_ACCESS_KEY: {'✅ Set' if GENAI_ACCESS_KEY else '❌ Missing'}")
logger.info(f"   API_BASE_URL: {'✅ Set' if API_BASE_URL else '❌ Missing'}")
logger.info(f"   API_AUTH_VALUE: {'✅ Set' if API_AUTH_VALUE else '❌ Missing'}")

# Simple import status tracking
import_status = {
    "status": "idle",
    "start_time": None,
    "end_time": None,
    "current_step": None,
    "results": {},
    "error": None
}

# In-memory logs
import_logs = []

def log_import(message: str):
    """Add message to import logs"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    import_logs.append(log_entry)
    logger.info(message)
    if len(import_logs) > 500:
        import_logs.pop(0)

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

# Essential health check endpoint (must be first)
@app.get("/ping")
async def ping():
    """Critical health check endpoint"""
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "service": "ask-innovai",
        "version": "2.1.0"
    }

# Basic endpoints with error handling
@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
        <h1>🤖 Ask InnovAI Admin</h1>
        <p>Admin interface file not found. Service is running.</p>
        <p>Available endpoints:</p>
        <ul>
        <li><a href="/ping">/ping</a> - Health check</li>
        <li><a href="/health">/health</a> - System health</li>
        <li><a href="/chat">/chat</a> - Chat interface</li>
        </ul>
        </body></html>
        """)

@app.get("/chat", response_class=HTMLResponse)
async def get_chat():
    try:
        with open("static/chat.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
        <h1>🤖 Ask InnovAI Chat</h1>
        <p>Chat interface file not found, but service is running.</p>
        <p><a href="/">← Back to Admin</a></p>
        </body></html>
        """)

@app.get("/health")
async def health():
    """Comprehensive health check"""
    components = {}
    
    # GenAI status
    components["genai"] = {
        "status": "connected" if GENAI_ACCESS_KEY else "not configured",
        "endpoint": GENAI_ENDPOINT
    }
    
    # OpenSearch status
    opensearch_host = os.getenv("OPENSEARCH_HOST")
    components["opensearch"] = {
        "status": "configured" if opensearch_host else "not configured",
        "host": opensearch_host or "not set"
    }
    
    # API Source status
    components["api_source"] = {
        "status": "configured" if (API_BASE_URL and API_AUTH_VALUE) else "not configured",
        "base_url": API_BASE_URL or "not set"
    }
    
    # Embedder status
    if EMBEDDER_AVAILABLE:
        try:
            # Import the module-level function
            from embedder import get_embedding_stats, EMBEDDING_MODEL
            
            stats = get_embedding_stats()
            components["embeddings"] = {
                "status": "healthy", 
                "model": stats.get("model_name", EMBEDDING_MODEL),  # Fallback to default
                "provider": stats.get("provider", "local"),
                "dimension": stats.get("embedding_dimension", 384),
                "model_loaded": stats.get("model_loaded", False)
            }
        except ImportError as e:
            components["embeddings"] = {
                "status": "warning", 
                "error": f"Import failed: {str(e)}",
                "model": "sentence-transformers/all-MiniLM-L6-v2"  # Default fallback
            }
        except Exception as e:
            # Fallback - try to get the default model name
            try:
                from embedder import EMBEDDING_MODEL
                model_name = EMBEDDING_MODEL
            except:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                
            components["embeddings"] = {
                "status": "warning", 
                "error": str(e),
                "model": model_name,
                "note": "Service available but not initialized"
            }
    else:
        components["embeddings"] = {
            "status": "not available",
            "note": "Will run without embeddings",
            "model": "none"
        }
@app.get("/debug/api-raw-response")
async def debug_api_raw_response():
    """Debug what the API actually returns"""
    if not API_BASE_URL or not API_AUTH_VALUE:
        return {"error": "API not configured"}
    
    try:
        headers = {API_AUTH_KEY: API_AUTH_VALUE, "Content-Type": "application/json"}
        params = {"limit": 1}
        
        logger.info(f"🔍 DEBUG: Calling API at {API_BASE_URL}")
        logger.info(f"🔍 DEBUG: Headers: {API_AUTH_KEY}: [HIDDEN]")
        logger.info(f"🔍 DEBUG: Params: {params}")
        
        response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=15)
        
        debug_info = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content_length": len(response.content),
            "content_type": response.headers.get("content-type", "unknown"),
            "raw_content_preview": response.text[:500],  # First 500 characters
            "is_json": False,
            "parsed_data": None,
            "error": None
        }
        
        # Try to parse as JSON
        try:
            if response.text.strip():  # Check if not empty
                data = response.json()
                debug_info["is_json"] = True
                debug_info["parsed_data"] = str(data)[:300]  # Preview of parsed data
            else:
                debug_info["error"] = "Response body is empty"
        except json.JSONDecodeError as e:
            debug_info["error"] = f"JSON decode failed: {str(e)}"
            debug_info["is_json"] = False
        
        return debug_info
        
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

@app.get("/status")
async def get_import_status():
    """Get import status"""
    return import_status

@app.get("/logs")
async def get_logs():
    """Get recent logs"""
    return {"logs": import_logs[-50:]}

# Chat endpoint with better error handling
@app.post("/chat")
async def chat_handler(request: ChatRequest):
    """Enhanced chat functionality"""
    try:
        if not GENAI_ACCESS_KEY:
            return {"reply": "Chat service not configured. Please set GENAI_ACCESS_KEY in environment variables."}
        
        # Simple search for context
        context = ""
        try:
            search_results = search_opensearch(request.message)
            if search_results:
                first_result = search_results[0].get('_source', {})
                context = first_result.get('text', '')[:500]
        except Exception as search_error:
            logger.warning(f"Search failed: {search_error}")
        
        # Build messages
        system_msg = "You are MetroAI, an intelligent assistant for Metro by T-Mobile customer service operations."
        if context:
            system_msg += f" Here's relevant context from the knowledge base: {context}"
        
        messages = [{"role": "system", "content": system_msg}]
        messages.extend([{"role": m["role"], "content": m["content"]} for m in request.history])
        messages.append({"role": "user", "content": request.message})
        
        # Call Digital Ocean AI
        payload = {"model": GENAI_ACCESS_KEY, "messages": messages}
        
        response = requests.post(
            GENAI_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GENAI_ACCESS_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "reply": reply.strip() if reply else "Sorry, I couldn't generate a response.",
                "context_used": bool(context)
            }
        else:
            logger.error(f"AI service error: {response.status_code} - {response.text}")
            return {"reply": f"AI service temporarily unavailable (status: {response.status_code})"}
            
    except requests.exceptions.Timeout:
        return {"reply": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"reply": "Unable to connect to AI service. Please check configuration."}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"reply": "Sorry, there was an unexpected error. Please try again."}

# Search endpoint
@app.get("/search")
async def search(q: str):
    """Basic search with error handling"""
    try:
        results = search_opensearch(q)
        formatted_results = []
        
        for hit in results:
            source = hit.get('_source', {})
            metadata = source.get('metadata', {})
            
            formatted_results.append({
                "id": hit.get('_id'),
                "title": f"{metadata.get('agent', 'Agent')} - {metadata.get('disposition', 'Call')}",
                "text": source.get('text', ''),
                "score": hit.get('_score', 0),
                "collection": metadata.get('template', 'unknown'),
                "metadata": metadata
            })
        
        return {
            "status": "success", 
            "results": formatted_results,
            "total_hits": len(formatted_results)
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "status": "error", 
            "error": str(e),
            "results": []
        }

# Simple import process
async def process_evaluation(evaluation: Dict) -> Dict:
    """Process single evaluation with error handling"""
    try:
        transcript = evaluation.get("transcript", "")
        if not transcript:
            return {"status": "skipped", "reason": "no_transcript"}
        
        # Clean transcript
        soup = BeautifulSoup(transcript, "html.parser")
        raw_text = soup.get_text(" ", strip=True)
        
        if len(raw_text.strip()) < 10:
            return {"status": "skipped", "reason": "too_short"}
        
        # Split into chunks
        chunks = split_into_chunks(raw_text)
        
        # Metadata from Endpoint API
        meta = {
            "evaluation_id": evaluation.get("evaluationId"),
            "template": evaluation.get("template_name"),
            "program": evaluation.get("partner"),
            "agent": evaluation.get("agentName"),
            "disposition": evaluation.get("disposition"),
            "language": evaluation.get("language"),
            "call_date": evaluation.get("call_date")
        }
        
        doc_id = str(evaluation.get("internalId", uuid4()))
        collection = evaluation.get("template_name", "default")
        
        # Index chunks
        indexed_chunks = 0
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding if available
                embedding = None
                if EMBEDDER_AVAILABLE:
                    try:
                        embedding = embed_text(chunk["text"])
                    except Exception as embed_error:
                        logger.warning(f"Embedding failed for chunk {i}: {embed_error}")
                
                doc_body = {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "text": chunk["text"],
                    "metadata": meta,
                    "source": "evaluation_api",
                    "indexed_at": datetime.now().isoformat()
                }
                
                if embedding:
                    doc_body["embedding"] = embedding
                
                index_document(f"{doc_id}-{i}", doc_body, index_override=collection)
                indexed_chunks += 1
                
            except Exception as e:
                logger.warning(f"Failed to index chunk {i} for evaluation {evaluation.get('evaluationId')}: {e}")
                continue
        
        return {
            "status": "success",
            "document_id": doc_id,
            "chunks_indexed": indexed_chunks,
            "collection": collection
        }
        
    except Exception as e:
        logger.error(f"Failed to process evaluation {evaluation.get('evaluationId')}: {e}")
        return {"status": "error", "error": str(e)}

async def run_import(collection: str = "all", max_docs: int = None):
    """Import process with comprehensive error handling"""
    try:
        update_import_status("running", "Starting import")
        log_import("Starting data import")
        
        if not API_BASE_URL or not API_AUTH_VALUE:
            raise Exception("API configuration missing: API_BASE_URL and API_AUTH_VALUE required")
        
        # Fetch data
        update_import_status("running", "Fetching data from API")
        log_import(f"Fetching data from: {API_BASE_URL}")
        
        headers = {API_AUTH_KEY: API_AUTH_VALUE, "Content-Type": "application/json"}
        params = {}
        if max_docs:
            params["limit"] = max_docs
        
        response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        evaluations = data.get("evaluations", [])
        log_import(f"Fetched {len(evaluations)} evaluations from API")
        
        if not evaluations:
            results = {"total_documents_processed": 0, "total_chunks_indexed": 0, "import_type": "full"}
            update_import_status("completed", results=results)
            return
        
        # Process evaluations
        update_import_status("running", f"Processing {len(evaluations)} evaluations")
        total_processed = 0
        total_chunks = 0
        errors = 0
        
        for i, evaluation in enumerate(evaluations):
            if i % 10 == 0:
                update_import_status("running", f"Processing evaluation {i+1}/{len(evaluations)}")
            
            result = await process_evaluation(evaluation)
            if result["status"] == "success":
                total_processed += 1
                total_chunks += result["chunks_indexed"]
            elif result["status"] == "error":
                errors += 1
            
            # Small delay to prevent overwhelming the system
            if i % 20 == 0:
                await asyncio.sleep(0.1)
        
        # Complete
        results = {
            "total_documents_processed": total_processed,
            "total_chunks_indexed": total_chunks,
            "errors": errors,
            "import_type": "full",
            "completed_at": datetime.now().isoformat()
        }
        
        log_import(f"Import completed: {total_processed} documents, {total_chunks} chunks, {errors} errors")
        update_import_status("completed", results=results)
        
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Could not connect to API: {str(e)}"
        log_import(error_msg)
        update_import_status("failed", error=error_msg)
    except requests.exceptions.Timeout as e:
        error_msg = f"API request timed out: {str(e)}"
        log_import(error_msg)
        update_import_status("failed", error=error_msg)
    except Exception as e:
        error_msg = f"Import failed: {str(e)}"
        log_import(error_msg)
        update_import_status("failed", error=error_msg)

@app.post("/import")
async def start_import(request: ImportRequest, background_tasks: BackgroundTasks):
    """Start import process"""
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
    
    # Start import
    background_tasks.add_task(run_import, request.collection, request.max_docs)
    return {"status": "success", "message": "Import started"}

# Additional helpful endpoints
@app.get("/debug")
async def debug_info():
    """Debug information"""
    return {
        "python_version": sys.version,
        "embedder_available": EMBEDDER_AVAILABLE,
        "environment_variables": {
            "PORT": os.getenv("PORT", "not set"),
            "OPENSEARCH_HOST": os.getenv("OPENSEARCH_HOST", "not set"),
            "GENAI_ENDPOINT": GENAI_ENDPOINT,
            "API_BASE_URL": API_BASE_URL[:50] + "..." if API_BASE_URL else "not set"
        },
        "import_logs_count": len(import_logs),
        "current_time": datetime.now().isoformat()
    }

# Add these missing endpoints to your app.py file

@app.get("/health")
async def health():
    """Comprehensive health check with better error handling"""
    components = {}
    
    try:
        # GenAI status
        components["genai"] = {
            "status": "connected" if GENAI_ACCESS_KEY else "not configured",
            "endpoint": GENAI_ENDPOINT
        }
        
        # OpenSearch status
        opensearch_host = os.getenv("OPENSEARCH_HOST")
        components["opensearch"] = {
            "status": "configured" if opensearch_host else "not configured",
            "host": opensearch_host or "not set"
        }
        
        # API Source status
        components["api_source"] = {
            "status": "configured" if (API_BASE_URL and API_AUTH_VALUE) else "not configured",
            "base_url": API_BASE_URL or "not set"
        }
        
        # Embedder status with better error handling
        if EMBEDDER_AVAILABLE:
            try:
                from embedder import get_embedding_stats, EMBEDDING_MODEL
                
                stats = get_embedding_stats()
                components["embeddings"] = {
                    "status": "healthy", 
                    "model": stats.get("model_name", EMBEDDING_MODEL),
                    "provider": stats.get("provider", "local"),
                    "dimension": stats.get("embedding_dimension", 384),
                    "model_loaded": stats.get("model_loaded", False)
                }
            except ImportError as e:
                components["embeddings"] = {
                    "status": "warning", 
                    "error": f"Import failed: {str(e)}",
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            except Exception as e:
                # Fallback - try to get the default model name
                try:
                    from embedder import EMBEDDING_MODEL
                    model_name = EMBEDDING_MODEL
                except:
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    
                components["embeddings"] = {
                    "status": "warning", 
                    "error": str(e),
                    "model": model_name,
                    "note": "Service available but not initialized"
                }
        else:
            components["embeddings"] = {
                "status": "not available",
                "note": "Will run without embeddings",
                "model": "none"
            }
        
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "components": components,
            "import_status": import_status["status"],
            "environment": "digital_ocean"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "components": components,  # Return whatever we managed to collect
            "error": str(e),
            "import_status": import_status["status"],
            "environment": "digital_ocean"
        }

@app.get("/last_import_info")
async def get_last_import_info():
    """Get information about the last import"""
    try:
        # For now, we'll use the import_status to track the last import
        # In a production system, you'd typically store this in a database
        
        if import_status.get("end_time") and import_status.get("status") == "completed":
            return {
                "status": "success",
                "last_import_timestamp": import_status["end_time"],
                "last_import_status": import_status["status"],
                "last_import_results": import_status.get("results", {})
            }
        else:
            return {
                "status": "success",
                "last_import_timestamp": None,
                "message": "No completed import found"
            }
            
    except Exception as e:
        logger.error(f"Failed to get last import info: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/clear_import_timestamp")
async def clear_import_timestamp():
    """Clear the import timestamp (for incremental imports)"""
    try:
        # Reset the import status
        import_status.update({
            "status": "idle",
            "start_time": None,
            "end_time": None,
            "current_step": None,
            "results": {},
            "error": None
        })
        
        log_import("Import timestamp cleared by user")
        
        return {
            "status": "success",
            "message": "Import timestamp cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear import timestamp: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# Add some additional helpful endpoints for debugging

@app.get("/import_statistics")
async def get_import_statistics():
    """Get basic statistics about the knowledge base"""
    try:
        # This is a simple implementation - in production you'd query your database
        return {
            "status": "success",
            "statistics": {
                "total_documents": import_status.get("results", {}).get("total_documents_processed", 0),
                "total_chunks": import_status.get("results", {}).get("total_chunks_indexed", 0),
                "collections": {
                    "evaluations": import_status.get("results", {}).get("total_documents_processed", 0)
                },
                "last_updated": import_status.get("end_time", "Never")
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/count_by_collection_and_program")
async def count_by_collection_and_program():
    """Count documents by collection and program"""
    try:
        # Simple implementation - in production you'd query OpenSearch
        return {
            "status": "success",
            "collection_program_counts": {
                "evaluations": {
                    "iQor": import_status.get("results", {}).get("total_documents_processed", 0)
                }
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Startup event with comprehensive error handling
@app.on_event("startup")
async def startup_event():
    """Startup initialization with better error handling"""
    try:
        logger.info("🚀 Ask InnovAI starting up...")
        logger.info(f"   Python version: {sys.version}")
        logger.info(f"   PORT: {os.getenv('PORT', '8080')}")
        
        # Test basic functionality
        try:
            test_chunks = split_into_chunks("This is a test sentence for startup validation.")
            logger.info(f"✅ Text chunking working: {len(test_chunks)} chunks")
        except Exception as e:
            logger.error(f"❌ Text chunking failed: {e}")
        
        # Try to preload embedder if available (but don't fail if it doesn't work)
        if EMBEDDER_AVAILABLE:
            try:
                from embedder import preload_embedding_model
                logger.info("🔄 Attempting to preload embedding model...")
                preload_embedding_model()
                logger.info("✅ Embedding model preloaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Embedding preload failed (will load on demand): {e}")
        
        logger.info("✅ Ask InnovAI startup complete - ready to serve requests")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        # Don't exit - let the app start anyway

@app.get("/debug/config")
async def debug_config():
    """Debug configuration without exposing sensitive values"""
    return {
        "api_base_url_set": bool(API_BASE_URL),
        "api_base_url_preview": API_BASE_URL[:50] + "..." if API_BASE_URL else "NOT_SET",
        "api_auth_key": API_AUTH_KEY,
        "api_auth_value_set": bool(API_AUTH_VALUE),
        "opensearch_host": os.getenv("OPENSEARCH_HOST", "NOT_SET"),
        "opensearch_user": os.getenv("OPENSEARCH_USER", "admin"),
        "opensearch_pass_set": bool(os.getenv("OPENSEARCH_PASS")),
        "genai_endpoint": GENAI_ENDPOINT,
        "genai_key_set": bool(GENAI_ACCESS_KEY),
        "embedder_available": EMBEDDER_AVAILABLE,
        "environment_vars": {
            "PORT": os.getenv("PORT", "not set"), 
            "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "default"),
            "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "local")
        }
    }

@app.post("/debug/test-connections")
async def test_all_connections():
    """Comprehensive connection testing with detailed logs"""
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "testing",
        "tests": {},
        "logs": [],
        "recommendations": []
    }
    
    def add_log(message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        test_results["logs"].append(f"[{timestamp}] {level}: {message}")
        logger.info(f"TEST: {message}")
    
    add_log("Starting comprehensive connection tests...")
    
    # Test 1: Environment Variables
    add_log("Testing environment variables...")
    env_test = {
        "status": "pass",
        "details": {},
        "issues": []
    }
    
    required_vars = {
        "API_BASE_URL": API_BASE_URL,
        "API_AUTH_VALUE": API_AUTH_VALUE,
        "OPENSEARCH_HOST": os.getenv("OPENSEARCH_HOST"),
        "GENAI_ACCESS_KEY": GENAI_ACCESS_KEY
    }
    
    for var_name, var_value in required_vars.items():
        if not var_value:
            env_test["issues"].append(f"{var_name} is not set")
            env_test["status"] = "fail"
        else:
            preview = var_value[:20] + "..." if len(var_value) > 20 else var_value
            env_test["details"][var_name] = f"Set ({preview})"
    
    test_results["tests"]["environment"] = env_test
    
    if env_test["status"] == "fail":
        test_results["recommendations"].append("Set missing environment variables in Digital Ocean App Platform settings")
        add_log(f"Environment test FAILED: {', '.join(env_test['issues'])}", "ERROR")
    else:
        add_log("Environment variables test PASSED")
    
    # Test 2: API Connection
    add_log("Testing API connection...")
    api_test = {
        "status": "unknown",
        "details": {},
        "response_preview": None,
        "error": None
    }
    
    if not API_BASE_URL or not API_AUTH_VALUE:
        api_test["status"] = "fail"
        api_test["error"] = "API_BASE_URL or API_AUTH_VALUE not configured"
        add_log("API test SKIPPED: Missing configuration", "ERROR")
    else:
        try:
            add_log(f"Attempting to connect to: {API_BASE_URL}")
            headers = {
                API_AUTH_KEY: API_AUTH_VALUE,
                "Content-Type": "application/json",
                "User-Agent": "AskInnovAI-TestClient/1.0"
            }
            add_log(f"Using auth header: {API_AUTH_KEY}")
            
            response = requests.get(
                API_BASE_URL,
                headers=headers,
                params={"limit": 1},
                timeout=15
            )
            
            api_test["details"]["status_code"] = response.status_code
            api_test["details"]["response_size"] = len(response.text)
            api_test["details"]["content_type"] = response.headers.get("content-type", "unknown")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    evaluations = data.get("evaluations", [])
                    api_test["details"]["evaluations_count"] = len(evaluations)
                    api_test["response_preview"] = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                    api_test["status"] = "pass"
                    add_log(f"API test PASSED: {len(evaluations)} evaluations available")
                except json.JSONDecodeError as e:
                    api_test["status"] = "fail"
                    api_test["error"] = f"Invalid JSON response: {e}"
                    api_test["response_preview"] = response.text[:200]
                    add_log(f"API test FAILED: Invalid JSON response", "ERROR")
            else:
                api_test["status"] = "fail"
                api_test["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                add_log(f"API test FAILED: HTTP {response.status_code}", "ERROR")
                
                if response.status_code == 401:
                    test_results["recommendations"].append("Check API_AUTH_VALUE - authentication failed")
                elif response.status_code == 403:
                    test_results["recommendations"].append("Check API permissions - access forbidden")
                elif response.status_code == 404:
                    test_results["recommendations"].append("Check API_BASE_URL - endpoint not found")
                    
        except requests.exceptions.Timeout:
            api_test["status"] = "fail"
            api_test["error"] = "Connection timeout (15s)"
            add_log("API test FAILED: Connection timeout", "ERROR")
            test_results["recommendations"].append("Check if API endpoint is accessible from Digital Ocean")
        except requests.exceptions.ConnectionError as e:
            api_test["status"] = "fail"
            api_test["error"] = f"Connection error: {str(e)}"
            add_log(f"API test FAILED: Connection error", "ERROR")
            test_results["recommendations"].append("Check API_BASE_URL format and network connectivity")
        except Exception as e:
            api_test["status"] = "fail"
            api_test["error"] = f"Unexpected error: {str(e)}"
            add_log(f"API test FAILED: {str(e)}", "ERROR")
    
    test_results["tests"]["api"] = api_test
    
    # Test 3: OpenSearch Connection
    add_log("Testing OpenSearch connection...")
    opensearch_test = {
        "status": "unknown",
        "details": {},
        "error": None
    }
    
    opensearch_host = os.getenv("OPENSEARCH_HOST")
    if not opensearch_host:
        opensearch_test["status"] = "fail"
        opensearch_test["error"] = "OPENSEARCH_HOST not configured"
        add_log("OpenSearch test SKIPPED: Missing configuration", "ERROR")
        test_results["recommendations"].append("Set OPENSEARCH_HOST environment variable")
    else:
        try:
            from opensearch_client import client
            add_log(f"Attempting to connect to OpenSearch: {opensearch_host}")
            
            info = client.info()
            opensearch_test["details"]["version"] = info.get("version", {}).get("number", "unknown")
            opensearch_test["details"]["cluster_name"] = info.get("cluster_name", "unknown")
            opensearch_test["status"] = "pass"
            add_log(f"OpenSearch test PASSED: Version {opensearch_test['details']['version']}")
            
            try:
                indices = client.cat.indices(format="json")
                opensearch_test["details"]["indices_count"] = len(indices) if indices else 0
                add_log(f"Found {opensearch_test['details']['indices_count']} indices")
            except Exception as e:
                add_log(f"Could not list indices: {e}", "WARN")
                
        except ImportError as e:
            opensearch_test["status"] = "fail"
            opensearch_test["error"] = f"opensearch-py not installed: {e}"
            add_log("OpenSearch test FAILED: Library not installed", "ERROR")
        except Exception as e:
            opensearch_test["status"] = "fail"
            opensearch_test["error"] = str(e)
            add_log(f"OpenSearch test FAILED: {str(e)}", "ERROR")
            test_results["recommendations"].append("Check OpenSearch credentials and network connectivity")
    
    test_results["tests"]["opensearch"] = opensearch_test
    
    # Test 4: Embeddings Service
    add_log("Testing embeddings service...")
    embeddings_test = {
        "status": "unknown",
        "details": {},
        "error": None
    }
    
    if not EMBEDDER_AVAILABLE:
        embeddings_test["status"] = "fail"
        embeddings_test["error"] = "Embedder module not available"
        add_log("Embeddings test FAILED: Module not available", "ERROR")
        test_results["recommendations"].append("Check if sentence-transformers is installed correctly")
    else:
        try:
            from embedder import embed_text, get_embedding_stats
            add_log("Testing embedding generation...")
            
            test_text = "This is a test embedding for connection validation."
            embedding = embed_text(test_text)
            
            embeddings_test["details"]["embedding_dimension"] = len(embedding)
            embeddings_test["details"]["sample_values"] = embedding[:3] if len(embedding) >= 3 else embedding
            embeddings_test["status"] = "pass"
            add_log(f"Embeddings test PASSED: {len(embedding)} dimensions")
            
            try:
                stats = get_embedding_stats()
                embeddings_test["details"]["model"] = stats.get("model_name", "unknown")
                embeddings_test["details"]["provider"] = stats.get("provider", "unknown")
                add_log(f"Using model: {embeddings_test['details']['model']}")
            except Exception as e:
                add_log(f"Could not get embedding stats: {e}", "WARN")
                
        except Exception as e:
            embeddings_test["status"] = "fail"
            embeddings_test["error"] = str(e)
            add_log(f"Embeddings test FAILED: {str(e)}", "ERROR")
            
            if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                test_results["recommendations"].append("Consider increasing memory or using a smaller embedding model")
            elif "model" in str(e).lower():
                test_results["recommendations"].append("Check if embedding model can be downloaded/loaded")
    
    test_results["tests"]["embeddings"] = embeddings_test
    
    # Test 5: Digital Ocean AI
    add_log("Testing Digital Ocean AI connection...")
    genai_test = {
        "status": "unknown",
        "details": {},
        "error": None
    }
    
    if not GENAI_ACCESS_KEY:
        genai_test["status"] = "fail"
        genai_test["error"] = "GENAI_ACCESS_KEY not configured"
        add_log("GenAI test SKIPPED: Missing configuration", "ERROR")
        test_results["recommendations"].append("Set GENAI_ACCESS_KEY for Digital Ocean AI")
    else:
        try:
            add_log(f"Testing Digital Ocean AI endpoint: {GENAI_ENDPOINT}")
            
            test_payload = {
                "model": GENAI_ACCESS_KEY,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                GENAI_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {GENAI_ACCESS_KEY}",
                    "Content-Type": "application/json"
                },
                json=test_payload,
                timeout=10
            )
            
            genai_test["details"]["status_code"] = response.status_code
            genai_test["details"]["response_size"] = len(response.text)
            
            if response.status_code == 200:
                genai_test["status"] = "pass"
                add_log("GenAI test PASSED")
            else:
                genai_test["status"] = "fail"
                genai_test["error"] = f"HTTP {response.status_code}: {response.text[:100]}"
                add_log(f"GenAI test FAILED: HTTP {response.status_code}", "ERROR")
                
        except Exception as e:
            genai_test["status"] = "fail"
            genai_test["error"] = str(e)
            add_log(f"GenAI test FAILED: {str(e)}", "ERROR")
    
    test_results["tests"]["genai"] = genai_test
    
    # Overall assessment
    all_tests = [test["status"] for test in test_results["tests"].values()]
    failed_tests = [name for name, test in test_results["tests"].items() if test["status"] == "fail"]
    
    if "fail" in all_tests:
        test_results["overall_status"] = "fail"
        add_log(f"OVERALL: FAILED - {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}", "ERROR")
    else:
        test_results["overall_status"] = "pass"
        add_log("OVERALL: ALL TESTS PASSED", "SUCCESS")
    
    add_log("Test suite completed")
    
    return test_results


# For Digital Ocean App Platform - Enhanced port binding
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8080
    port = int(os.getenv("PORT", 8080))
    
    logger.info(f"🚀 Starting Ask InnovAI on port {port}")
    
    # Start the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )