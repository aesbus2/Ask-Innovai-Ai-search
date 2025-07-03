# Production App.py - Clean & Ready for Production
# Version: 3.1.0 - Production Clean

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

# Production logging setup
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

# Import modules
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
    title="Ask InnovAI Production",
    description="AI-Powered Knowledge Assistant - Production Ready",
    version="3.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"⚠️ Failed to mount static files: {e}")

# Production Configuration
GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT", "https://tcxn3difq23zdxlph2heigba.agents.do-ai.run")
GENAI_ACCESS_KEY = os.getenv("GENAI_ACCESS_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://innovai-6abj.onrender.com/api/content")
API_AUTH_KEY = os.getenv("API_AUTH_KEY", "auth")
API_AUTH_VALUE = os.getenv("API_AUTH_VALUE", "")

# Production import status tracking
import_status = {
    "status": "idle",
    "start_time": None,
    "end_time": None,
    "current_step": None,
    "results": {},
    "error": None,
    "import_type": "full"
}

# In-memory logs (last 100 entries)
import_logs = []

def log_import(message: str):
    """Add message to import logs"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    import_logs.append(log_entry)
    logger.info(message)
    if len(import_logs) > 100:
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
# PRODUCTION DATA PROCESSING FUNCTIONS
# ============================================================================

def extract_qa_pairs(evaluation_text: str) -> List[str]:
    """Extract Question and Answer pairs from evaluation text"""
    if not evaluation_text:
        return []
    
    # Clean HTML
    soup = BeautifulSoup(evaluation_text, "html.parser")
    clean_text = soup.get_text(" ", strip=True)
    
    qa_chunks = []
    
    # Split by sections first
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
    """Split transcript while preserving speaker boundaries"""
    if not transcript:
        return []
    
    # Clean HTML
    soup = BeautifulSoup(transcript, "html.parser")
    clean_transcript = soup.get_text(" ", strip=True)
    
    # Split by speaker patterns
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
    
    # Combine turns into appropriately sized chunks
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
    """Process evaluation with production rules"""
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
        
        doc_id = str(evaluation.get("internalId", uuid4()))
        collection = evaluation.get("template_name", "evaluations")
        
        # Index chunks
        indexed_chunks = 0
        for i, chunk in enumerate(all_chunks):
            try:
                # Generate embedding
                embedding = None
                if EMBEDDER_AVAILABLE:
                    try:
                        embedding = embed_text(chunk["text"])
                    except Exception:
                        pass
                
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
                error_msg = f"Failed to index chunk {i}: {str(e)}"
                logger.warning(error_msg)
                
                # Check if it's an OpenSearch connectivity issue
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    raise Exception(f"OpenSearch connection error: {str(e)}")
                
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
        logger.error(f"Failed to process evaluation: {e}")
        return {"status": "error", "error": str(e)}

# ============================================================================
# PRODUCTION API FETCHING
# ============================================================================

async def fetch_evaluations(max_docs: int = None):
    """Fetch evaluations from API"""
    try:
        if not API_BASE_URL or not API_AUTH_VALUE:
            raise Exception("API configuration missing")
        
        headers = {
            API_AUTH_KEY: API_AUTH_VALUE,
            'Accept': 'application/json',
            'User-Agent': 'Ask-InnovAI-Production/3.1.0'
        }
        
        params = {}
        if max_docs:
            params["limit"] = max_docs
        
        # Make request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    API_BASE_URL, 
                    headers=headers, 
                    params=params, 
                    timeout=60
                )
                
                if response.status_code == 200:
                    break
                elif response.status_code in [502, 503, 504] and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise Exception(f"API returned HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise Exception(f"Request failed after {max_retries} attempts: {e}")
        
        # Parse response
        data = response.json()
        evaluations = data.get("evaluations", [])
        
        if not evaluations and isinstance(data, list):
            evaluations = data
        
        return evaluations
        
    except Exception as e:
        logger.error(f"Failed to fetch evaluations: {e}")
        raise

async def run_production_import(collection: str = "all", max_docs: int = None):
    """Production import process with better error handling"""
    try:
        update_import_status("running", "Starting import")
        log_import("🚀 Starting production import")
        
        # Check OpenSearch connectivity first
        update_import_status("running", "Checking OpenSearch connectivity")
        try:
            from opensearch_client import test_connection
            
            # Test OpenSearch connection
            if test_connection():
                log_import("✅ OpenSearch connection verified")
            else:
                error_msg = "OpenSearch connection failed - database may be unavailable"
                log_import(f"❌ {error_msg}")
                update_import_status("failed", error=error_msg)
                return
                
        except Exception as e:
            error_msg = f"OpenSearch connection check failed: {str(e)}"
            log_import(f"❌ {error_msg}")
            update_import_status("failed", error=error_msg)
            return
        
        # Fetch evaluations
        update_import_status("running", "Fetching evaluation data")
        evaluations = await fetch_evaluations(max_docs)
        
        if not evaluations:
            results = {"total_documents_processed": 0, "total_chunks_indexed": 0, "import_type": "full"}
            update_import_status("completed", results=results)
            return
        
        # Process evaluations
        update_import_status("running", f"Processing {len(evaluations)} evaluations")
        
        total_processed = 0
        total_chunks = 0
        errors = 0
        opensearch_errors = 0
        
        for i, evaluation in enumerate(evaluations):
            if i % 10 == 0:
                update_import_status("running", f"Processing evaluation {i+1}/{len(evaluations)}")
            
            result = await process_evaluation(evaluation)
            if result["status"] == "success":
                total_processed += 1
                total_chunks += result["chunks_indexed"]
            elif result["status"] == "error":
                errors += 1
                # Check if it's an OpenSearch error
                if "opensearch" in str(result.get("error", "")).lower() or "timeout" in str(result.get("error", "")).lower():
                    opensearch_errors += 1
            
            # If too many OpenSearch errors, stop the import
            if opensearch_errors > 5:
                error_msg = f"Too many OpenSearch connection errors ({opensearch_errors}). Stopping import."
                log_import(f"❌ {error_msg}")
                update_import_status("failed", error=error_msg)
                return
            
            if i % 20 == 0:
                await asyncio.sleep(0.1)
        
        # Complete
        results = {
            "total_documents_processed": total_processed,
            "total_chunks_indexed": total_chunks,
            "errors": errors,
            "opensearch_errors": opensearch_errors,
            "import_type": "full",
            "completed_at": datetime.now().isoformat()
        }
        
        log_import(f"🎉 Import completed: {total_processed} documents, {total_chunks} chunks, {errors} errors")
        update_import_status("completed", results=results)
        
    except Exception as e:
        error_msg = f"Import failed: {str(e)}"
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
        "service": "ask-innovai-production",
        "version": "3.1.0"
    }

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
        <h1>🤖 Ask InnovAI Production</h1>
        <p><strong>Status:</strong> Production Ready ✅</p>
        <p><strong>Version:</strong> 3.1.0</p>
        <p>Admin interface file not found. Please ensure static/index.html exists.</p>
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
        <p>Chat interface file not found. Please ensure static/chat.html exists.</p>
        <p><a href="/">← Back to Admin</a></p>
        </body></html>
        """)

@app.get("/health")
async def health():
    """Production health check with OpenSearch connectivity"""
    try:
        components = {}
        
        # GenAI status
        components["genai"] = {
            "status": "configured" if GENAI_ACCESS_KEY else "not configured"
        }
        
        # OpenSearch status with actual connectivity test
        try:
            from opensearch_client import get_connection_status, test_connection, get_opensearch_config
            
            config = get_opensearch_config()
            
            # Get current connection status
            conn_status = get_connection_status()
            
            if config["host"] == "not_configured":
                components["opensearch"] = {
                    "status": "not configured",
                    "message": "OPENSEARCH_HOST not set"
                }
            elif conn_status["connected"]:
                components["opensearch"] = {
                    "status": "connected",
                    "host": config["host"],
                    "port": config["port"],
                    "url": config["url"]
                }
            elif conn_status["tested"]:
                components["opensearch"] = {
                    "status": "connection_failed",
                    "host": config["host"],
                    "port": config["port"],
                    "url": config["url"],
                    "error": conn_status["last_error"][:100] if conn_status["last_error"] else "Unknown error"
                }
            else:
                # Try to test connection now
                if test_connection():
                    components["opensearch"] = {
                        "status": "connected",
                        "host": config["host"],
                        "port": config["port"],
                        "url": config["url"]
                    }
                else:
                    components["opensearch"] = {
                        "status": "connection_failed",
                        "host": config["host"],
                        "port": config["port"],
                        "url": config["url"],
                        "error": "Connection test failed"
                    }
                    
        except Exception as e:
            components["opensearch"] = {
                "status": "error",
                "error": str(e)[:100]
            }
        
        # API Source status
        components["api_source"] = {
            "status": "configured" if API_AUTH_VALUE else "not configured"
        }
        
        # Embedder status
        if EMBEDDER_AVAILABLE:
            try:
                stats = get_embedding_stats()
                components["embeddings"] = {
                    "status": "healthy", 
                    "model_loaded": stats.get("model_loaded", False)
                }
            except Exception:
                components["embeddings"] = {"status": "warning"}
        else:
            components["embeddings"] = {"status": "not available"}
        
        # Overall status
        overall_status = "ok"
        if components["opensearch"]["status"] == "connection_failed":
            overall_status = "degraded"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "components": components,
                "import_status": import_status["status"],
                "version": "3.1.0"
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

@app.post("/chat")
async def chat_handler(request: ChatRequest):
    """Production chat functionality"""
    try:
        if not GENAI_ACCESS_KEY:
            return {"reply": "Chat service not configured. Please set GENAI_ACCESS_KEY."}
        
        # Search for context
        context = ""
        try:
            # Search in your main data index
            from opensearch_client import get_connection_status
            conn_status = get_connection_status()
            
            if conn_status.get("connected", False):
                search_results = search_opensearch(request.message, index_override="Ai Corporate SPTR - TEST")
                if search_results:
                    first_result = search_results[0].get('_source', {})
                    context = first_result.get('text', '')[:500]
        except Exception as e:
            logger.warning(f"Search context failed: {e}")
            pass
        
        # Build messages
        system_msg = "You are MetroAI, an intelligent assistant for Metro by T-Mobile customer service operations."
        if context:
            system_msg += f" Context: {context}"
        
        messages = [{"role": "system", "content": system_msg}]
        messages.extend([{"role": m["role"], "content": m["content"]} for m in request.history])
        messages.append({"role": "user", "content": request.message})
        
        # Call AI service
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
            return {"reply": reply.strip() if reply else "Sorry, I couldn't generate a response."}
        else:
            return {"reply": "AI service temporarily unavailable"}
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"reply": "Sorry, there was an error. Please try again."}

@app.get("/search")
async def search(q: str):
    """Production search"""
    try:
        # Search in your main data index
        results = search_opensearch(q, index_override="Ai Corporate SPTR - TEST")
        formatted_results = []
        
        for hit in results:
            source = hit.get('_source', {})
            metadata = source.get('metadata', {})
            
            formatted_results.append({
                "id": hit.get('_id'),
                "title": f"{metadata.get('agent', 'Agent')} - {metadata.get('disposition', 'Call')}",
                "text": source.get('text', ''),
                "score": hit.get('_score', 0),
                "metadata": metadata
            })
        
        return {
            "status": "success", 
            "results": formatted_results,
            "total_hits": len(formatted_results)
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"status": "error", "error": str(e), "results": []}

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
    """Start import process with proper error handling"""
    try:
        if import_status["status"] == "running":
            return JSONResponse(
                status_code=400, 
                content={"status": "error", "message": "Import already running"}
            )
        
        # Reset status
        import_status.update({
            "status": "idle",
            "start_time": None,
            "end_time": None,
            "current_step": None,
            "results": {},
            "error": None,
            "import_type": request.import_type
        })
        
        # Start import
        background_tasks.add_task(run_production_import, request.collection, request.max_docs)
        
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "Import started"}
        )
        
    except Exception as e:
        logger.error(f"Import endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to start import: {str(e)}"}
        )

@app.get("/import_statistics")
async def get_import_statistics():
    """Get import statistics"""
    try:
        # This would typically query your database
        # For now, return basic stats from last import
        if import_status.get("results"):
            return {
                "status": "success",
                "statistics": {
                    "total_documents": import_status["results"].get("total_documents_processed", 0),
                    "total_chunks": import_status["results"].get("total_chunks_indexed", 0),
                    "last_import": import_status.get("end_time")
                }
            }
        else:
            return {
                "status": "success",
                "statistics": {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "last_import": None
                }
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/test_opensearch")
async def test_opensearch_connection():
    """Test OpenSearch connectivity and list indices"""
    try:
        from opensearch_client import test_connection, get_connection_status, get_opensearch_config, client
        
        config = get_opensearch_config()
        
        # Test connection
        connection_ok = test_connection()
        conn_status = get_connection_status()
        
        if not connection_ok:
            return JSONResponse(
                status_code=503,  # Service Unavailable
                content={
                    "status": "error",
                    "message": f"OpenSearch connection failed: {conn_status.get('last_error', 'Unknown error')}",
                    "host": config["host"],
                    "port": config["port"],
                    "url": config["url"],
                    "ssl_enabled": config["ssl_enabled"],
                    "details": {
                        "tested": conn_status.get("tested", False),
                        "last_test": conn_status.get("last_test"),
                        "connection_working": False
                    }
                }
            )
        
        # Get list of indices
        try:
            indices = client.indices.get_alias(index="*")
            index_names = list(indices.keys())
            
            # Filter out system indices (those starting with .)
            user_indices = [name for name in index_names if not name.startswith('.')]
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "OpenSearch connection successful",
                    "host": config["host"],
                    "port": config["port"],
                    "url": config["url"],
                    "ssl_enabled": config["ssl_enabled"],
                    "indices": {
                        "total_count": len(index_names),
                        "user_indices": user_indices,
                        "system_indices_count": len(index_names) - len(user_indices)
                    },
                    "details": {
                        "tested": conn_status.get("tested", False),
                        "last_test": conn_status.get("last_test"),
                        "connection_working": True
                    }
                }
            )
            
        except Exception as e:
            # Connection works but can't list indices
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "OpenSearch connection successful (indices list unavailable)",
                    "host": config["host"],
                    "port": config["port"],
                    "url": config["url"],
                    "ssl_enabled": config["ssl_enabled"],
                    "warning": f"Could not list indices: {str(e)[:100]}",
                    "details": {
                        "tested": conn_status.get("tested", False),
                        "last_test": conn_status.get("last_test"),
                        "connection_working": True
                    }
                }
            )
        
    except Exception as e:
        logger.error(f"OpenSearch test failed: {e}")
        
        # Try to get config even if test fails
        try:
            from opensearch_client import get_opensearch_config
            config = get_opensearch_config()
        except:
            config = {
                "host": "unknown",
                "port": "unknown", 
                "url": "unknown",
                "ssl_enabled": False
            }
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"OpenSearch test failed: {str(e)}",
                "host": config["host"],
                "port": config["port"],
                "url": config["url"],
                "ssl_enabled": config["ssl_enabled"],
                "details": {
                    "tested": False,
                    "connection_working": False,
                    "error": str(e)[:200]
                }
            }
        )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Non-blocking startup - app starts even if OpenSearch is down"""
    try:
        logger.info("🚀 Ask InnovAI Production starting...")
        logger.info(f"   Version: 3.1.0")
        logger.info(f"   Port: {os.getenv('PORT', '8080')}")
        
        # Check configuration without blocking
        api_configured = bool(API_AUTH_VALUE)
        genai_configured = bool(GENAI_ACCESS_KEY)
        opensearch_configured = bool(os.getenv("OPENSEARCH_HOST"))
        
        logger.info(f"   API Source: {'✅ Configured' if api_configured else '❌ Missing'}")
        logger.info(f"   GenAI: {'✅ Configured' if genai_configured else '❌ Missing'}")
        logger.info(f"   OpenSearch: {'✅ Configured' if opensearch_configured else '❌ Missing'}")
        
        # Preload embedder if available (non-blocking)
        if EMBEDDER_AVAILABLE:
            try:
                preload_embedding_model()
                logger.info("✅ Embedding model preloaded")
            except Exception as e:
                logger.warning(f"⚠️ Embedding preload failed: {e}")
        
        # Test OpenSearch in background (non-blocking)
        if opensearch_configured:
            def background_opensearch_test():
                try:
                    import time
                    time.sleep(3)  # Give app time to fully start
                    from opensearch_client import test_connection
                    if test_connection():
                        logger.info("✅ OpenSearch connection verified in background")
                    else:
                        logger.warning("⚠️ OpenSearch connection failed in background test")
                except Exception as e:
                    logger.warning(f"⚠️ Background OpenSearch test failed: {e}")
            
            # Run in background thread
            import threading
            threading.Thread(target=background_opensearch_test, daemon=True).start()
        
        logger.info("🎉 Production startup complete (non-blocking)")
        
        # Log readiness status
        ready_components = sum([api_configured, genai_configured, opensearch_configured])
        logger.info(f"📊 Ready components: {ready_components}/3")
        
        if ready_components == 3:
            logger.info("🟢 All components configured - ready for production")
        elif ready_components >= 2:
            logger.info("🟡 Most components configured - check missing components")
        else:
            logger.warning("🔴 Many components missing - check configuration")
            
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        # Don't re-raise - let app start anyway

# For Digital Ocean App Platform
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    logger.info(f"🚀 Starting Ask InnovAI Production on port {port}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )