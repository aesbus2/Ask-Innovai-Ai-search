# Restored API App.py - Adding back all the missing request configurations
# Version: 2.6.0 - Restored full headers and request handling
# Fixes empty response issue by adding back essential request components

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
    title="Ask InnovAI",
    description="AI-Powered Knowledge Assistant with Restored API Handling",
    version="2.6.0"
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
API_BASE_URL = os.getenv("API_BASE_URL")  
API_AUTH_KEY = os.getenv("API_AUTH_KEY", "Authorization")
API_AUTH_VALUE = os.getenv("API_AUTH_VALUE")

logger.info(f"🔧 Restored Configuration:")
logger.info(f"   API_BASE_URL: {API_BASE_URL}")
logger.info(f"   API_AUTH_KEY: {API_AUTH_KEY}")
logger.info(f"   API_AUTH_VALUE: {'✅ Set' if API_AUTH_VALUE else '❌ Missing'}")

# Create a session with restored headers and settings
session = requests.Session()

# RESTORED: Complete headers that were likely removed
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'cross-site',
})

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

# ============================================================================
# RESTORED API PROCESSING FUNCTIONS
# ============================================================================

def extract_qa_pairs(evaluation_text: str) -> List[str]:
    """Extract Question and Answer pairs from evaluation text, keeping them together."""
    if not evaluation_text:
        return []
    
    # Clean HTML
    soup = BeautifulSoup(evaluation_text, "html.parser")
    clean_text = soup.get_text(" ", strip=True)
    
    qa_chunks = []
    
    # Split by sections first (Section: XXXX)
    sections = re.split(r'Section:\s*([^<\n]+?)(?=Section:|$)', clean_text, flags=re.IGNORECASE)
    
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            section_name = sections[i].strip()
            section_content = sections[i + 1].strip()
            
            # Extract Q&A pairs from this section
            qa_pattern = r'Question:\s*([^?]+\??)\s*Answer:\s*([^.]+\.?)'
            matches = re.finditer(qa_pattern, section_content, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                qa_chunk = f"Section: {section_name}\nQuestion: {question}\nAnswer: {answer}"
                qa_chunks.append(qa_chunk)
    
    # Fallback: if no sections, try direct Q&A extraction
    if not qa_chunks:
        qa_pattern = r'Question:\s*([^?]+\??)\s*Answer:\s*([^.]+\.?)'
        matches = re.finditer(qa_pattern, clean_text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            qa_chunk = f"Question: {question}\nAnswer: {answer}"
            qa_chunks.append(qa_chunk)
    
    return qa_chunks

def split_transcript_by_speakers(transcript: str) -> List[str]:
    """Split transcript while preserving speaker boundaries."""
    if not transcript:
        return []
    
    # Clean HTML
    soup = BeautifulSoup(transcript, "html.parser")
    clean_transcript = soup.get_text(" ", strip=True)
    
    # Split by speaker patterns (Speaker A/B with timestamps)
    speaker_pattern = r'(Speaker [AB] \(\d{2}:\d{2}:\d{2}\):)'
    parts = re.split(speaker_pattern, clean_transcript)
    
    if len(parts) < 3:
        # No speaker patterns found, use regular chunking
        chunks = split_into_chunks(clean_transcript, max_chars=1100, overlap=100)
        return [chunk["text"] for chunk in chunks]
    
    # Group speaker turns into chunks
    speaker_turns = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            speaker = parts[i].strip()
            content = parts[i + 1].strip()
            if content:
                speaker_turn = f"{speaker} {content}"
                speaker_turns.append(speaker_turn)
    
    # Combine turns into appropriately sized chunks without breaking speaker boundaries
    chunks = []
    current_chunk = ""
    max_size = 1100
    
    for turn in speaker_turns:
        if current_chunk and len(current_chunk + "\n" + turn) > max_size:
            chunks.append(current_chunk.strip())
            current_chunk = turn
        else:
            current_chunk = current_chunk + "\n" + turn if current_chunk else turn
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

async def process_evaluation(evaluation: Dict) -> Dict:
    """Process evaluation with complete metadata extraction"""
    try:
        evaluation_text = evaluation.get("evaluation", "")
        transcript_text = evaluation.get("transcript", "")
        
        if not evaluation_text and not transcript_text:
            return {"status": "skipped", "reason": "no_content"}
        
        all_chunks = []
        
        # Process evaluation Q&A
        if evaluation_text:
            qa_chunks = extract_qa_pairs(evaluation_text)
            for qa_text in qa_chunks:
                if len(qa_text.strip()) >= 20:
                    all_chunks.append({
                        "text": qa_text,
                        "content_type": "evaluation_qa",
                        "offset": 0,
                        "length": len(qa_text),
                        "chunk_index": len(all_chunks)
                    })
        
        # Process transcript
        if transcript_text:
            transcript_chunks = split_transcript_by_speakers(transcript_text)
            for transcript_chunk in transcript_chunks:
                if len(transcript_chunk.strip()) >= 20:
                    all_chunks.append({
                        "text": transcript_chunk,
                        "content_type": "transcript",
                        "offset": 0,
                        "length": len(transcript_chunk),
                        "chunk_index": len(all_chunks)
                    })
        
        if not all_chunks:
            return {"status": "skipped", "reason": "no_meaningful_content"}
        
        # Complete metadata extraction
        meta = {
            "evaluation_id": evaluation.get("evaluationId"),
            "internal_id": evaluation.get("internalId"),
            "template_id": evaluation.get("template_id"),
            "template": evaluation.get("template_name"),
            "program": evaluation.get("partner"),
            "site": evaluation.get("site"),
            "lob": evaluation.get("lob"),
            "agent": evaluation.get("agentName"),
            "disposition": evaluation.get("disposition"),
            "sub_disposition": evaluation.get("subDisposition"),
            "language": evaluation.get("language"),
            "call_date": evaluation.get("call_date"),
            "call_duration": evaluation.get("call_duration"),
            "created_on": evaluation.get("created_on")
        }
        
        # Use internalId as document ID
        doc_id = str(evaluation.get("internalId", uuid4()))
        collection = evaluation.get("template_name", "evaluations")
        
        # Index chunks with embeddings
        indexed_chunks = 0
        for i, chunk in enumerate(all_chunks):
            try:
                # Generate embedding
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
                    "content_type": chunk["content_type"],
                    "metadata": meta,
                    "source": "evaluation_api",
                    "indexed_at": datetime.now().isoformat(),
                    "chunk_length": chunk["length"]
                }
                
                if embedding:
                    doc_body["embedding"] = embedding
                
                chunk_id = f"{doc_id}-{chunk['content_type']}-{i}"
                index_document(chunk_id, doc_body, index_override=collection)
                indexed_chunks += 1
                
            except Exception as e:
                logger.warning(f"Failed to index chunk {i}: {e}")
                continue
        
        return {
            "status": "success",
            "document_id": doc_id,
            "chunks_indexed": indexed_chunks,
            "collection": collection,
            "evaluation_chunks": len([c for c in all_chunks if c["content_type"] == "evaluation_qa"]),
            "transcript_chunks": len([c for c in all_chunks if c["content_type"] == "transcript"]),
            "total_content_length": sum(chunk["length"] for chunk in all_chunks)
        }
        
    except Exception as e:
        error_msg = f"Failed to process evaluation {evaluation.get('evaluationId')}: {e}"
        logger.error(error_msg)
        return {"status": "error", "error": str(e)}

# ============================================================================
# RESTORED API FETCHING WITH COMPLETE HEADERS
# ============================================================================

async def fetch_evaluations(max_docs: int = None):
    """RESTORED: Fetch evaluations with complete headers and session handling"""
    try:
        if not API_BASE_URL:
            raise Exception("API_BASE_URL is not configured")
        
        if not API_AUTH_VALUE:
            raise Exception("API_AUTH_VALUE is not configured")
        
        log_import(f"📡 RESTORED: Fetching evaluations from: {API_BASE_URL}")
        
        # RESTORED: Complete headers that may have been removed
        headers = {
            API_AUTH_KEY: API_AUTH_VALUE,
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Origin': API_BASE_URL.split('/api')[0] if '/api' in API_BASE_URL else API_BASE_URL,
            'Referer': API_BASE_URL,
            'X-Requested-With': 'XMLHttpRequest'  # Many APIs require this
        }
        
        # RESTORED: Complete request parameters
        params = {}
        if max_docs:
            params["limit"] = max_docs
        
        log_import(f"🔑 RESTORED: Using headers: {list(headers.keys())}")
        log_import(f"📋 RESTORED: Request params: {params}")
        
        # RESTORED: Use session with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                log_import(f"🔄 RESTORED: Attempt {attempt + 1}/{max_retries}")
                
                response = session.get(
                    API_BASE_URL, 
                    headers=headers, 
                    params=params, 
                    timeout=60,
                    verify=True,  # RESTORED: SSL verification
                    allow_redirects=True,  # RESTORED: Follow redirects
                    stream=False  # RESTORED: Don't stream response
                )
                
                # Log comprehensive response details
                log_import(f"📊 RESTORED: Response status: {response.status_code}")
                log_import(f"📄 RESTORED: Response content-type: {response.headers.get('content-type', 'unknown')}")
                log_import(f"📏 RESTORED: Response content length: {len(response.content)}")
                log_import(f"🔗 RESTORED: Final URL after redirects: {response.url}")
                
                if response.status_code == 200:
                    break
                elif response.status_code in [502, 503, 504] and attempt < max_retries - 1:
                    log_import(f"⚠️ RESTORED: Server error {response.status_code}, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise Exception(f"API returned HTTP {response.status_code}: {response.text[:200]}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    log_import(f"⚠️ RESTORED: Request failed, retrying: {e}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise Exception(f"Request failed after {max_retries} attempts: {e}")
        
        if not response.text.strip():
            # RESTORED: Try alternative request methods if empty response
            log_import("⚠️ RESTORED: Empty response, trying alternative methods...")
            
            # Try without some headers that might be causing issues
            minimal_headers = {API_AUTH_KEY: API_AUTH_VALUE}
            response = session.get(API_BASE_URL, headers=minimal_headers, params=params, timeout=60)
            
            if not response.text.strip():
                raise Exception("API returned empty response even with minimal headers")
        
        # RESTORED: Enhanced response validation
        response_text = response.text.strip()
        
        # Check for common error responses
        if response_text.startswith("<!DOCTYPE") or response_text.startswith("<html"):
            log_import("❌ RESTORED: Received HTML instead of JSON")
            # Try to extract useful information from HTML
            soup = BeautifulSoup(response_text, "html.parser")
            title = soup.find("title")
            title_text = title.get_text() if title else "Unknown"
            raise Exception(f"API returned HTML page '{title_text}' instead of JSON. Check API endpoint URL.")
        
        if response_text.startswith("<?xml"):
            raise Exception("API returned XML instead of JSON")
        
        # Parse JSON with enhanced error handling
        try:
            data = response.json()
            log_import("✅ RESTORED: JSON parsing successful")
        except json.JSONDecodeError as e:
            log_import(f"❌ RESTORED: JSON parsing failed: {str(e)}")
            log_import(f"Raw response preview: {response_text[:500]}")
            
            # Try to clean response and parse again
            cleaned_text = response_text.replace('\n', '').replace('\r', '').strip()
            if cleaned_text and (cleaned_text.startswith('{') or cleaned_text.startswith('[')):
                try:
                    data = json.loads(cleaned_text)
                    log_import("✅ RESTORED: JSON parsing successful after cleaning")
                except:
                    raise Exception(f"API returned invalid JSON: {str(e)}")
            else:
                raise Exception(f"API returned invalid JSON: {str(e)}")
        
        # Extract evaluations from response
        evaluations = data.get("evaluations", [])
        log_import(f"📋 RESTORED: Found {len(evaluations)} evaluations")
        
        if not evaluations and isinstance(data, list):
            evaluations = data
            log_import(f"📋 RESTORED: Data is array format: {len(evaluations)} items")
        
        if not evaluations:
            log_import(f"⚠️ RESTORED: No evaluations found. Response keys: {list(data.keys()) if isinstance(data, dict) else 'Array response'}")
        
        return evaluations
        
    except Exception as e:
        logger.error(f"❌ RESTORED: Failed to fetch evaluations: {e}")
        raise Exception(f"API request failed: {str(e)}")

async def run_import(collection: str = "all", max_docs: int = None):
    """RESTORED: Import process with enhanced error handling"""
    try:
        update_import_status("running", "Starting RESTORED import with complete headers")
        log_import("🚀 RESTORED: Starting import with complete API handling")
        
        if not API_AUTH_VALUE:
            raise Exception("API_AUTH_VALUE environment variable is required")
        
        if not API_BASE_URL:
            raise Exception("API_BASE_URL environment variable is required")
        
        # Fetch evaluations with restored handling
        update_import_status("running", "Fetching evaluation data with RESTORED headers")
        
        evaluations = await fetch_evaluations(max_docs)
        
        if not evaluations:
            log_import(f"⚠️ RESTORED: No evaluations found")
            results = {"total_documents_processed": 0, "total_chunks_indexed": 0, "import_type": "full"}
            update_import_status("completed", results=results)
            return
        
        # Process evaluations
        update_import_status("running", f"Processing {len(evaluations)} evaluations")
        log_import(f"🔄 RESTORED: Processing {len(evaluations)} evaluations")
        
        total_processed = 0
        total_chunks = 0
        total_qa_chunks = 0
        total_transcript_chunks = 0
        errors = 0
        
        for i, evaluation in enumerate(evaluations):
            if i % 10 == 0:
                update_import_status("running", f"Processing evaluation {i+1}/{len(evaluations)}")
            
            result = await process_evaluation(evaluation)
            if result["status"] == "success":
                total_processed += 1
                total_chunks += result["chunks_indexed"]
                total_qa_chunks += result.get("evaluation_chunks", 0)
                total_transcript_chunks += result.get("transcript_chunks", 0)
            elif result["status"] == "error":
                errors += 1
            
            # Small delay to prevent overwhelming
            if i % 20 == 0:
                await asyncio.sleep(0.1)
        
        # Complete with results
        results = {
            "total_documents_processed": total_processed,
            "total_chunks_indexed": total_chunks,
            "total_qa_chunks": total_qa_chunks,
            "total_transcript_chunks": total_transcript_chunks,
            "errors": errors,
            "import_type": "full",
            "completed_at": datetime.now().isoformat(),
            "api_endpoint": API_BASE_URL,
            "restored_headers": True,
            "api_rules_applied": [
                "Keep Question and Answers together",
                "Never split between speakers",
                "Complete metadata extraction",
                "Restored complete headers and session handling"
            ]
        }
        
        log_import(f"🎉 RESTORED: Import completed successfully!")
        log_import(f"   📄 Documents processed: {total_processed}")
        log_import(f"   🧩 Total chunks: {total_chunks}")
        log_import(f"   📋 Q&A chunks: {total_qa_chunks}")
        log_import(f"   🎙️ Transcript chunks: {total_transcript_chunks}")
        log_import(f"   ❌ Errors: {errors}")
        
        update_import_status("completed", results=results)
        
    except Exception as e:
        error_msg = f"RESTORED import failed: {str(e)}"
        log_import(f"❌ {error_msg}")
        update_import_status("failed", error=error_msg)

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@app.get("/ping")
async def ping():
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "service": "ask-innovai-restored",
        "version": "2.6.0"
    }

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
        <h1>🤖 Ask InnovAI Admin - RESTORED</h1>
        <p>Restored admin interface with complete API headers and session handling.</p>
        <p>This version includes all the missing headers and request configurations.</p>
        <ul>
        <li><a href="/ping">/ping</a> - Health check</li>
        <li><a href="/health">/health</a> - System health</li>
        <li><a href="/debug/test-restored-api">/debug/test-restored-api</a> - Test restored API handling</li>
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
        <h1>🤖 Ask InnovAI Chat - RESTORED</h1>
        <p>Chat interface with restored API processing.</p>
        <p><a href="/">← Back to Admin</a></p>
        </body></html>
        """)

@app.get("/health")
async def health():
    """Enhanced health check"""
    try:
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
            "endpoint": API_BASE_URL or "not set",
            "restored_headers": True,
            "session_configured": True
        }
        
        # Embedder status
        if EMBEDDER_AVAILABLE:
            try:
                stats = get_embedding_stats()
                components["embeddings"] = {
                    "status": "healthy", 
                    "model": stats.get("model_name", "unknown"),
                    "provider": stats.get("provider", "local"),
                    "dimension": stats.get("embedding_dimension", 384),
                    "model_loaded": stats.get("model_loaded", False)
                }
            except Exception as e:
                components["embeddings"] = {
                    "status": "warning", 
                    "error": str(e)
                }
        else:
            components["embeddings"] = {
                "status": "not available",
                "note": "Will run without embeddings"
            }
        
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "components": components,
            "import_status": import_status["status"],
            "environment": "digital_ocean",
            "version": "2.6.0_restored_headers",
            "restored_features": [
                "Complete HTTP headers",
                "Session-based requests", 
                "Retry logic with exponential backoff",
                "Enhanced response validation",
                "SSL verification",
                "Redirect handling"
            ]
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/debug/test-restored-api")
async def test_restored_api():
    """Test the restored API handling"""
    try:
        if not API_BASE_URL or not API_AUTH_VALUE:
            return {
                "status": "error",
                "error": "API_BASE_URL or API_AUTH_VALUE not configured"
            }
        
        log_import("🧪 Testing RESTORED API handling...")
        
        # Test the restored fetch function
        test_evaluations = await fetch_evaluations(1)  # Get just 1 for testing
        
        return {
            "status": "success",
            "restored_api_test": "passed",
            "evaluations_found": len(test_evaluations) if test_evaluations else 0,
            "sample_evaluation": {
                "internal_id": test_evaluations[0].get("internalId") if test_evaluations else None,
                "evaluation_id": test_evaluations[0].get("evaluationId") if test_evaluations else None,
                "agent": test_evaluations[0].get("agentName") if test_evaluations else None,
                "has_evaluation": bool(test_evaluations[0].get("evaluation")) if test_evaluations else False,
                "has_transcript": bool(test_evaluations[0].get("transcript")) if test_evaluations else False
            } if test_evaluations else None,
            "restored_features": [
                "✅ Complete HTTP headers",
                "✅ Session-based requests",
                "✅ Retry logic",
                "✅ Enhanced validation",
                "✅ SSL verification",
                "✅ Redirect handling"
            ],
            "message": "RESTORED API handling is working correctly!"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "RESTORED API test failed - check logs for details"
        }

# Chat endpoint
@app.post("/chat")
async def chat_handler(request: ChatRequest):
    """Chat functionality"""
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
                "metadata": metadata,
                "content_type": source.get('content_type', 'unknown')
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

# Import management endpoints
@app.get("/status")
async def get_import_status():
    return import_status

@app.get("/logs")
async def get_logs():
    return {"logs": import_logs[-50:]}

@app.post("/import")
async def start_import(request: ImportRequest, background_tasks: BackgroundTasks):
    """Start RESTORED import process"""
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
    
    # Start restored import
    background_tasks.add_task(run_import, request.collection, request.max_docs)
    return {"status": "success", "message": "RESTORED import started with complete headers and session handling"}

# Additional endpoints
@app.get("/last_import_info")
async def get_last_import_info():
    """Get information about the last import"""
    try:
        if import_status.get("end_time") and import_status.get("status") == "completed":
            return {
                "status": "success",
                "last_import_timestamp": import_status["end_time"],
                "last_import_status": import_status["status"],
                "last_import_results": import_status.get("results", {}),
                "restored_headers": True
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
    """Clear the import timestamp"""
    try:
        import_status.update({
            "status": "idle",
            "start_time": None,
            "end_time": None,
            "current_step": None,
            "results": {},
            "error": None
        })
        
        log_import("🔄 Import timestamp cleared by user")
        
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

# Startup event
@app.on_event("startup")
async def startup_event():
    """RESTORED startup initialization"""
    try:
        logger.info("🚀 Ask InnovAI RESTORED starting up...")
        logger.info(f"   Version: 2.6.0 - RESTORED with complete headers")
        logger.info(f"   Python version: {sys.version}")
        logger.info(f"   PORT: {os.getenv('PORT', '8080')}")
        
        # Log restored features
        logger.info("✅ RESTORED features enabled:")
        logger.info("   • Complete HTTP headers (User-Agent, Accept, etc.)")
        logger.info("   • Session-based requests with connection pooling")
        logger.info("   • Retry logic with exponential backoff")
        logger.info("   • Enhanced response validation")
        logger.info("   • SSL verification")
        logger.info("   • Redirect handling")
        
        # Try to preload embedder if available
        if EMBEDDER_AVAILABLE:
            try:
                preload_embedding_model()
                logger.info("✅ Embedding model preloaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Embedding preload failed (will load on demand): {e}")
        
        logger.info("🎉 Ask InnovAI RESTORED startup complete!")
        logger.info("📋 Test endpoint: /debug/test-restored-api")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

# For Digital Ocean App Platform
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    logger.info(f"🚀 Starting Ask InnovAI RESTORED on port {port}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )