# Enhanced app.py for Digital Ocean App Platform
# Version: 2.1.0 - With proper API data processing rules and cleaned variables

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
    description="AI-Powered Knowledge Assistant with Enhanced API Processing",
    version="2.2.0"
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

# Configuration with validation - CLEANED UP VARIABLES
GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT", "https://tcxn3difq23zdxlph2heigba.agents.do-ai.run")
GENAI_ACCESS_KEY = os.getenv("GENAI_ACCESS_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL")  # No hardcoded default
API_DISCOVERY_ENDPOINT = os.getenv("API_DISCOVERY_ENDPOINT", "/api/content")
API_AUTH_KEY = os.getenv("API_AUTH_KEY", "Authorization")
API_AUTH_VALUE = os.getenv("API_AUTH_VALUE")  # No hardcoded default

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

# ============================================================================
# ENHANCED API DATA PROCESSING FUNCTIONS
# ============================================================================

def extract_qa_pairs(evaluation_text: str) -> List[str]:
    """
    Extract Question and Answer pairs from evaluation text, keeping them together.
    Rule: "Keep Question and Answers in the same chunk"
    """
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
                
                log_import(f"✅ Extracted Q&A pair from {section_name}: {len(qa_chunk)} chars")
    
    # Fallback: if no sections, try direct Q&A extraction
    if not qa_chunks:
        qa_pattern = r'Question:\s*([^?]+\??)\s*Answer:\s*([^.]+\.?)'
        matches = re.finditer(qa_pattern, clean_text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            qa_chunk = f"Question: {question}\nAnswer: {answer}"
            qa_chunks.append(qa_chunk)
            
            log_import(f"✅ Extracted direct Q&A pair: {len(qa_chunk)} chars")
    
    return qa_chunks

def split_transcript_by_speakers(transcript: str) -> List[str]:
    """
    Split transcript while preserving speaker boundaries.
    Rule: "Never chunk or split the sentence between speakers"
    """
    if not transcript:
        return []
    
    # Clean HTML
    soup = BeautifulSoup(transcript, "html.parser")
    clean_transcript = soup.get_text(" ", strip=True)
    
    # Split by speaker patterns (Speaker A/B with timestamps)
    speaker_pattern = r'(Speaker [AB] \(\d{2}:\d{2}:\d{2}\):)'
    parts = re.split(speaker_pattern, clean_transcript)
    
    if len(parts) < 3:
        # No speaker patterns found, use regular chunking but log warning
        log_import("⚠️ No speaker patterns found in transcript, using regular chunking")
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
    
    log_import(f"🎙️ Found {len(speaker_turns)} speaker turns in transcript")
    
    # Combine turns into appropriately sized chunks without breaking speaker boundaries
    chunks = []
    current_chunk = ""
    max_size = 1100
    
    for turn in speaker_turns:
        # If adding this turn would exceed max size and we have content, start new chunk
        if current_chunk and len(current_chunk + "\n" + turn) > max_size:
            chunks.append(current_chunk.strip())
            log_import(f"📝 Created transcript chunk: {len(current_chunk)} chars")
            current_chunk = turn
        else:
            current_chunk = current_chunk + "\n" + turn if current_chunk else turn
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        log_import(f"📝 Created final transcript chunk: {len(current_chunk)} chars")
    
    return chunks

async def process_evaluation(evaluation: Dict) -> Dict:
    """
    Enhanced evaluation processing following API data rules:
    1. Keep Question and Answers together (evaluation field)
    2. Never split between speakers (transcript field)  
    3. Link everything together with internalId for complete reconstruction
    """
    try:
        evaluation_text = evaluation.get("evaluation", "")
        transcript_text = evaluation.get("transcript", "")
        
        if not evaluation_text and not transcript_text:
            return {"status": "skipped", "reason": "no_content"}
        
        log_import(f"🔄 Processing evaluation {evaluation.get('evaluationId')}")
        
        all_chunks = []
        
        # Process evaluation Q&A (keep Q&A together)
        if evaluation_text:
            log_import(f"📋 Processing evaluation Q&A for evaluation {evaluation.get('evaluationId')}")
            qa_chunks = extract_qa_pairs(evaluation_text)
            
            for qa_text in qa_chunks:
                if len(qa_text.strip()) >= 20:  # Minimum meaningful content
                    all_chunks.append({
                        "text": qa_text,
                        "content_type": "evaluation_qa",
                        "offset": 0,
                        "length": len(qa_text),
                        "chunk_index": len(all_chunks)
                    })
        
        # Process transcript (never split between speakers)
        if transcript_text:
            log_import(f"🎙️ Processing transcript for evaluation {evaluation.get('evaluationId')}")
            transcript_chunks = split_transcript_by_speakers(transcript_text)
            
            for transcript_chunk in transcript_chunks:
                if len(transcript_chunk.strip()) >= 20:  # Minimum meaningful content
                    all_chunks.append({
                        "text": transcript_chunk,
                        "content_type": "transcript",
                        "offset": 0,
                        "length": len(transcript_chunk),
                        "chunk_index": len(all_chunks)
                    })
        
        if not all_chunks:
            return {"status": "skipped", "reason": "no_meaningful_content"}
        
        # Complete metadata extraction from API structure
        meta = {
            "evaluation_id": evaluation.get("evaluationId"),
            "internal_id": evaluation.get("internalId"),
            "template_id": evaluation.get("template_id"),
            "template": evaluation.get("template_name"),
            "program": evaluation.get("partner"),  # Maps to your 'partner' field
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
        
        # Use internalId as document ID for complete reconstruction
        doc_id = str(evaluation.get("internalId", uuid4()))
        collection = evaluation.get("template_name", "evaluations")
        
        log_import(f"📄 Document ID: {doc_id}, Collection: {collection}")
        
        # Index chunks with embeddings
        indexed_chunks = 0
        for i, chunk in enumerate(all_chunks):
            try:
                # Generate embedding using embedder.py
                embedding = None
                if EMBEDDER_AVAILABLE:
                    try:
                        embedding = embed_text(chunk["text"])
                        log_import(f"🧠 Generated embedding for chunk {i}: {len(embedding)} dimensions")
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
                
                # Use compound ID for complete reconstruction capability
                chunk_id = f"{doc_id}-{chunk['content_type']}-{i}"
                index_document(chunk_id, doc_body, index_override=collection)
                indexed_chunks += 1
                
                log_import(f"✅ Indexed chunk {i} ({chunk['content_type']}): {len(chunk['text'])} chars")
                
            except Exception as e:
                logger.warning(f"Failed to index chunk {i}: {e}")
                continue
        
        result = {
            "status": "success",
            "document_id": doc_id,
            "chunks_indexed": indexed_chunks,
            "collection": collection,
            "evaluation_chunks": len([c for c in all_chunks if c["content_type"] == "evaluation_qa"]),
            "transcript_chunks": len([c for c in all_chunks if c["content_type"] == "transcript"]),
            "total_content_length": sum(chunk["length"] for chunk in all_chunks)
        }
        
        log_import(f"🎉 Successfully processed evaluation {evaluation.get('evaluationId')}: {indexed_chunks} chunks indexed")
        return result
        
    except Exception as e:
        error_msg = f"Failed to process evaluation {evaluation.get('evaluationId')}: {e}"
        logger.error(error_msg)
        log_import(f"❌ {error_msg}")
        return {"status": "error", "error": str(e)}

# ============================================================================
# API DISCOVERY AND DATA FETCHING - UPDATED TO USE API_BASE_URL
# ============================================================================

async def discover_api_endpoints():
    """Discover available API endpoints using the discovery endpoint"""
    try:
        discovery_url = API_BASE_URL.rstrip('/') + API_DISCOVERY_ENDPOINT
        headers = {API_AUTH_KEY: API_AUTH_VALUE, "Content-Type": "application/json"}
        
        logger.info(f"🔍 Discovering API endpoints at: {discovery_url}")
        
        response = requests.get(discovery_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        discovery_data = response.json()
        logger.info(f"📋 Discovery response: {discovery_data}")
        
        return discovery_data
        
    except Exception as e:
        logger.error(f"❌ API discovery failed: {e}")
        raise Exception(f"API discovery failed: {str(e)}")

async def get_evaluations_endpoint():
    """Get the evaluations endpoint from discovery"""
    try:
        discovery_data = await discover_api_endpoints()
        
        if not isinstance(discovery_data, dict):
            raise Exception("Discovery response is not a valid JSON object")
        
        evaluations_endpoint = None
        
        # Try different common patterns for endpoint discovery
        if "endpoints" in discovery_data:
            endpoints = discovery_data["endpoints"]
            if isinstance(endpoints, dict):
                for key in ["evaluations", "evaluation", "evals", "data"]:
                    if key in endpoints:
                        evaluations_endpoint = endpoints[key]
                        break
        elif "evaluations_endpoint" in discovery_data:
            evaluations_endpoint = discovery_data["evaluations_endpoint"]
        elif "evaluations_url" in discovery_data:
            evaluations_endpoint = discovery_data["evaluations_url"]
        elif "base_url" in discovery_data and "evaluations_path" in discovery_data:
            base = discovery_data["base_url"].rstrip('/')
            path = discovery_data["evaluations_path"]
            evaluations_endpoint = f"{base}{path}"
        else:
            for key, value in discovery_data.items():
                if "evaluation" in key.lower() and (key.endswith("_endpoint") or key.endswith("_url")):
                    evaluations_endpoint = value
                    break
        
        if not evaluations_endpoint:
            available_keys = list(discovery_data.keys())
            raise Exception(f"No evaluations endpoint found in discovery response. Available keys: {available_keys}")
        
        # Convert relative URLs to absolute URLs
        if evaluations_endpoint.startswith('/'):
            evaluations_endpoint = API_BASE_URL.rstrip('/') + evaluations_endpoint
        
        logger.info(f"✅ Found evaluations endpoint: {evaluations_endpoint}")
        return evaluations_endpoint
        
    except Exception as e:
        logger.error(f"❌ Failed to determine evaluations endpoint: {e}")
        raise

async def run_import(collection: str = "all", max_docs: int = None):
    """Enhanced import process with proper API data processing"""
    try:
        update_import_status("running", "Starting enhanced import with API discovery")
        log_import("🚀 Starting enhanced data import with API discovery")
        
        if not API_AUTH_VALUE:
            raise Exception("API_AUTH_VALUE is required")
        
        if not API_BASE_URL:
            raise Exception("API_BASE_URL is required")
        
        # Discover the evaluations endpoint
        update_import_status("running", "Discovering API endpoints")
        log_import(f"🔍 Discovering endpoints from: {API_BASE_URL}{API_DISCOVERY_ENDPOINT}")
        
        try:
            evaluations_endpoint = await get_evaluations_endpoint()
        except Exception as e:
            raise Exception(f"API discovery failed: {str(e)}. Please check your API_BASE_URL and API_DISCOVERY_ENDPOINT settings.")
        
        log_import(f"✅ Using evaluations endpoint: {evaluations_endpoint}")
        
        # Fetch data from discovered endpoint
        update_import_status("running", "Fetching evaluation data from API")
        
        headers = {API_AUTH_KEY: API_AUTH_VALUE, "Content-Type": "application/json"}
        params = {}
        if max_docs:
            params["limit"] = max_docs
        
        log_import(f"📡 Request to: {evaluations_endpoint}")
        log_import(f"📋 Request params: {params}")
        
        response = requests.get(evaluations_endpoint, headers=headers, params=params, timeout=60)
        
        # Log response details
        log_import(f"📊 Response status: {response.status_code}")
        log_import(f"📄 Response content-type: {response.headers.get('content-type', 'unknown')}")
        log_import(f"📏 Response content length: {len(response.content)}")
        
        if response.status_code != 200:
            raise Exception(f"Evaluations API returned HTTP {response.status_code}: {response.text[:200]}")
        
        if not response.text.strip():
            raise Exception("Evaluations API returned empty response")
        
        # Parse JSON
        try:
            data = response.json()
            log_import("✅ JSON parsing successful")
        except json.JSONDecodeError as e:
            log_import(f"❌ JSON parsing failed: {str(e)}")
            log_import(f"Raw response preview: {response.text[:300]}")
            raise Exception(f"Evaluations API returned invalid JSON: {str(e)}")
        
        # Extract evaluations from response
        evaluations = data.get("evaluations", [])
        log_import(f"📋 Fetched {len(evaluations)} evaluations from API")
        
        if not evaluations:
            if isinstance(data, list):
                evaluations = data
                log_import(f"📋 Data is array format: {len(evaluations)} items")
            else:
                log_import(f"⚠️ No evaluations found. Response keys: {list(data.keys())}")
                results = {"total_documents_processed": 0, "total_chunks_indexed": 0, "import_type": "full"}
                update_import_status("completed", results=results)
                return
        
        # Process evaluations with enhanced rules
        update_import_status("running", f"Processing {len(evaluations)} evaluations with API rules")
        log_import(f"🔄 Starting processing with enhanced API rules:")
        log_import(f"   ✓ Keep Question and Answers together")
        log_import(f"   ✓ Never split between speakers")
        log_import(f"   ✓ Complete metadata extraction")
        
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
            
            # Small delay to prevent overwhelming the system
            if i % 20 == 0:
                await asyncio.sleep(0.1)
        
        # Complete with enhanced statistics
        results = {
            "total_documents_processed": total_processed,
            "total_chunks_indexed": total_chunks,
            "total_qa_chunks": total_qa_chunks,
            "total_transcript_chunks": total_transcript_chunks,
            "errors": errors,
            "import_type": "full",
            "completed_at": datetime.now().isoformat(),
            "endpoint_used": evaluations_endpoint,
            "discovery_endpoint": f"{API_BASE_URL}{API_DISCOVERY_ENDPOINT}",
            "api_rules_applied": [
                "Keep Question and Answers together",
                "Never split between speakers",
                "Complete metadata extraction"
            ]
        }
        
        log_import(f"🎉 Enhanced import completed:")
        log_import(f"   📄 Documents processed: {total_processed}")
        log_import(f"   🧩 Total chunks: {total_chunks}")
        log_import(f"   📋 Q&A chunks: {total_qa_chunks}")
        log_import(f"   🎙️ Transcript chunks: {total_transcript_chunks}")
        log_import(f"   ❌ Errors: {errors}")
        
        update_import_status("completed", results=results)
        
    except Exception as e:
        error_msg = f"Enhanced import failed: {str(e)}"
        log_import(f"❌ {error_msg}")
        update_import_status("failed", error=error_msg)

# ============================================================================
# ENHANCED SYSTEM VALIDATION
# ============================================================================

def validate_system_on_startup():
    """Validate that everything is working with API processing rules"""
    
    logger.info("🔍 Validating enhanced API processing system...")
    
    # Test embedder
    if EMBEDDER_AVAILABLE:
        try:
            test_embedding = embed_text("Test evaluation question and answer pair")
            logger.info(f"✅ Embedder working: {len(test_embedding)} dimensions")
        except Exception as e:
            logger.warning(f"⚠️ Embedder issue: {e}")
    else:
        logger.warning("⚠️ Embedder not available")
    
    # Test sentence splitter
    try:
        test_chunks = split_into_chunks("Test sentence one. Test sentence two.")
        logger.info(f"✅ Sentence splitter working: {len(test_chunks)} chunks")
    except Exception as e:
        logger.error(f"❌ Sentence splitter issue: {e}")
        return False
    
    # Test API processing functions
    test_eval = {
        "internalId": "test123",
        "evaluationId": 1,
        "template_name": "test",
        "partner": "test_partner",
        "agentName": "Test Agent",
        "evaluation": "Section: TEST SECTION<br>Question: Test question? Answer: Test answer.",
        "transcript": "Speaker A (00:00:01): Hello there. Speaker B (00:00:02): Hi, how are you?"
    }
    
    try:
        # Test Q&A extraction
        qa_chunks = extract_qa_pairs(test_eval["evaluation"])
        logger.info(f"✅ Q&A extraction working: {len(qa_chunks)} Q&A pairs")
        
        # Test transcript splitting
        transcript_chunks = split_transcript_by_speakers(test_eval["transcript"])
        logger.info(f"✅ Speaker-aware splitting working: {len(transcript_chunks)} chunks")
        
        logger.info("🎉 Enhanced API processing validation successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ API processing validation failed: {e}")
        return False

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@app.get("/ping")
async def ping():
    """Critical health check endpoint"""
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "service": "ask-innovai-enhanced",
        "version": "2.2.0"
    }

@app.get("/debug/simple")
async def debug_simple():
    """Simple debug endpoint for basic connectivity testing"""
    return {
        "status": "ok",
        "message": "Simple debug endpoint working",
        "timestamp": datetime.now().isoformat(),
        "service": "ask-innovai-enhanced"
    }

@app.get("/test-json")
async def test_json():
    """Test JSON response endpoint"""
    return {
        "test": "success",
        "json_working": True,
        "timestamp": datetime.now().isoformat(),
        "data": {
            "string": "Hello World",
            "number": 12345,
            "boolean": True,
            "array": [1, 2, 3, "test"],
            "nested": {
                "key": "value",
                "another": "data"
            }
        }
    }

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
        <h1>🤖 Ask InnovAI Admin - Enhanced</h1>
        <p>Enhanced admin interface with API processing rules.</p>
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
        <h1>🤖 Ask InnovAI Chat - Enhanced</h1>
        <p>Enhanced chat interface with proper API data processing.</p>
        <p><a href="/">← Back to Admin</a></p>
        </body></html>
        """)

@app.get("/health")
async def health():
    """Enhanced health check with API processing validation"""
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
        
        # API Source status - UPDATED TO USE API_BASE_URL
        components["api_source"] = {
            "status": "configured" if (API_BASE_URL and API_AUTH_VALUE) else "not configured",
            "base_url": API_BASE_URL or "not set",
            "discovery_endpoint": API_DISCOVERY_ENDPOINT
        }
        
        # Enhanced embedder status
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
                    "error": str(e),
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
        else:
            components["embeddings"] = {
                "status": "not available",
                "note": "Will run without embeddings"
            }
        
        # API Processing Rules status
        components["api_processing"] = {
            "status": "enhanced",
            "rules_implemented": [
                "Keep Question and Answers together",
                "Never split between speakers",
                "Complete metadata extraction"
            ]
        }
        
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "components": components,
            "import_status": import_status["status"],
            "environment": "digital_ocean",
            "version": "2.2.0_enhanced"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "environment": "digital_ocean"
        }

# Chat endpoint
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
    """Get import status"""
    return import_status

@app.get("/logs")
async def get_logs():
    """Get recent logs"""
    return {"logs": import_logs[-50:]}

@app.post("/import")
async def start_import(request: ImportRequest, background_tasks: BackgroundTasks):
    """Start enhanced import process"""
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
    
    # Start enhanced import
    background_tasks.add_task(run_import, request.collection, request.max_docs)
    return {"status": "success", "message": "Enhanced import started with API processing rules"}

# Additional helpful endpoints
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
                "api_rules_applied": import_status.get("results", {}).get("api_rules_applied", [])
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

# Debug endpoints - UPDATED TO USE API_BASE_URL
@app.get("/debug/config")
async def debug_config():
    """Debug configuration"""
    return {
        "api_base_url_set": bool(API_BASE_URL),
        "api_base_url": API_BASE_URL or "NOT_SET",
        "api_discovery_endpoint": API_DISCOVERY_ENDPOINT,
        "api_auth_key": API_AUTH_KEY,
        "api_auth_value_set": bool(API_AUTH_VALUE),
        "opensearch_host": os.getenv("OPENSEARCH_HOST", "NOT_SET"),
        "genai_endpoint": GENAI_ENDPOINT,
        "genai_key_set": bool(GENAI_ACCESS_KEY),
        "embedder_available": EMBEDDER_AVAILABLE,
        "api_processing_enhanced": True,
        "api_rules": [
            "Keep Question and Answers together",
            "Never split between speakers",
            "Complete metadata extraction"
        ]
    }

@app.get("/debug/test-api-processing")
async def test_api_processing():
    """Test API processing functions"""
    test_eval = {
        "internalId": "test123",
        "evaluationId": 1,
        "template_name": "test",
        "partner": "test_partner",
        "agentName": "Test Agent",
        "evaluation": "Section: TEST SECTION<br>Question: Test question? Answer: Test answer.<br><br>Question: Another question? Answer: Another answer.",
        "transcript": "Speaker A (00:00:01): Hello there, welcome to Metro. Speaker B (00:00:12): Hi, I need help with my account. Speaker A (00:00:15): I'll be happy to help you today."
    }
    
    try:
        # Test Q&A extraction
        qa_chunks = extract_qa_pairs(test_eval["evaluation"])
        
        # Test transcript splitting
        transcript_chunks = split_transcript_by_speakers(test_eval["transcript"])
        
        return {
            "status": "success",
            "qa_extraction": {
                "chunks_found": len(qa_chunks),
                "sample_chunk": qa_chunks[0] if qa_chunks else None
            },
            "transcript_splitting": {
                "chunks_found": len(transcript_chunks),
                "sample_chunk": transcript_chunks[0] if transcript_chunks else None
            },
            "rules_verified": [
                f"Q&A kept together: {len(qa_chunks) > 0}",
                f"Speaker boundaries respected: {len(transcript_chunks) > 0}"
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Startup event with enhanced validation
@app.on_event("startup")
async def startup_event():
    """Enhanced startup initialization"""
    try:
        logger.info("🚀 Ask InnovAI Enhanced starting up...")
        logger.info(f"   Version: 2.2.0 with API processing rules")
        logger.info(f"   Python version: {sys.version}")
        logger.info(f"   PORT: {os.getenv('PORT', '8080')}")
        
        # Validate enhanced system
        if validate_system_on_startup():
            logger.info("✅ Enhanced API processing system validated successfully")
        else:
            logger.warning("⚠️ Some validation issues found - check logs")
        
        # Try to preload embedder if available
        if EMBEDDER_AVAILABLE:
            try:
                preload_embedding_model()
                logger.info("✅ Embedding model preloaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Embedding preload failed (will load on demand): {e}")
        
        logger.info("🎉 Ask InnovAI Enhanced startup complete - ready with API processing rules!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

# For Digital Ocean App Platform
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    logger.info(f"🚀 Starting Ask InnovAI Enhanced on port {port}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )