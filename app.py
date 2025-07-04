# Production App.py - Clean & Ready for Production with Enhanced Import Process
# Version: 3.3.0 - Production Clean with Collection Cleanup and AgentId Endpoint

import os
import logging
import requests
import asyncio
import json
import sys
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from uuid import uuid4
from chat_handlers import chat_router
from typing import Dict, List, Any, Optional
from fastapi import Query
from collections import defaultdict




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
    from opensearch_client import search_opensearch, index_document, search_vector
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
    version="3.3.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

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

def clean_collection_name(collection_name: str) -> str:
    """
    Clean collection name by removing spaces and special characters
    to create valid OpenSearch index names
    """
    if not collection_name:
        return "default-collection"
    
    # Convert to lowercase and replace spaces with hyphens
    cleaned = collection_name.lower().strip()
    
    # Replace spaces and underscores with hyphens
    cleaned = re.sub(r'[\s_]+', '-', cleaned)
    
    # Remove special characters except hyphens and alphanumeric
    cleaned = re.sub(r'[^a-z0-9\-]', '', cleaned)
    
    # Remove multiple consecutive hyphens
    cleaned = re.sub(r'-+', '-', cleaned)
    
    # Remove leading/trailing hyphens
    cleaned = cleaned.strip('-')
    
    # Ensure it's not empty
    if not cleaned:
        cleaned = "default-collection"
    
    # Ensure it starts with a letter or number (OpenSearch requirement)
    if not cleaned[0].isalnum():
        cleaned = f"collection-{cleaned}"
    
    # Limit length (OpenSearch has limits)
    if len(cleaned) > 50:
        cleaned = cleaned[:50].rstrip('-')
    
    return cleaned

def extract_agent_info(evaluation: Dict) -> Dict[str, Any]:
    """Extract agent information from evaluation"""
    agent_name = evaluation.get("agentName", "")
    
    # Generate agent ID from name if not provided
    agent_id = str(hash(agent_name) % 100000000) if agent_name else "00000000"
    
    return {
        "agentId": agent_id,
        "agentName": agent_name,
        "site": evaluation.get("site", ""),
        "partner": evaluation.get("partner", ""),
        "lob": evaluation.get("lob", ""),
        "disposition": evaluation.get("disposition", ""),
        "subDisposition": evaluation.get("subDisposition", ""),
        "language": evaluation.get("language", ""),
        "call_date": evaluation.get("call_date", ""),
        "call_duration": evaluation.get("call_duration", 0),
        "template_name": evaluation.get("template_name", "")
    }

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    history: list
    programs: list = []

class ImportRequest(BaseModel):
    collection: str = "all"
    max_docs: Optional[int] = None
    import_type: str = "full"
    batch_size: Optional[int] = None

class AgentSearchRequest(BaseModel):
    agentId: Optional[str] = None
    agentName: Optional[str] = None
    site: Optional[str] = None
    partner: Optional[str] = None
    lob: Optional[str] = None
    limit: Optional[int] = 10

# ============================================================================
# MEMORY MANAGEMENT FUNCTIONS
# ============================================================================

async def cleanup_memory_after_batch():
    """Comprehensive memory cleanup after processing a batch"""
    import gc
    
    try:
        # Clear embedding cache if available
        if EMBEDDER_AVAILABLE:
            try:
                from embedder import get_embedding_service
                service = get_embedding_service()
                # Clear LRU cache
                if hasattr(service, '_cached_embed_single'):
                    cache_info = service._cached_embed_single.cache_info()
                    if cache_info.currsize > 100:  # Only clear if cache is getting large
                        service._cached_embed_single.cache_clear()
                        log_import(f"🧹 Cleared embedding LRU cache ({cache_info.currsize} entries)")
            except Exception as e:
                log_import(f"⚠️ Could not clear embedding cache: {str(e)[:50]}")
        
        # Clear OpenSearch connection pool stats (but keep connections)
        try:
            from opensearch_client import get_opensearch_manager
            manager = get_opensearch_manager()
            # Don't reset all stats, just clear any large cached data
            if hasattr(manager, '_performance_stats'):
                # Reset detailed logs but keep important counters
                if manager._performance_stats.get('total_requests', 0) > 1000:
                    log_import("🧹 Resetting OpenSearch performance stats")
                    manager._performance_stats['total_response_time'] = 0.0
                    manager._performance_stats['avg_response_time'] = 0.0
        except Exception as e:
            log_import(f"⚠️ Could not clear OpenSearch cache: {str(e)[:50]}")
        
        # Force garbage collection
        collected = gc.collect()
        if collected > 0:
            log_import(f"🧹 Garbage collected {collected} objects")
        
        # Clear any large local variables that might be lingering
        locals().clear()
        
        # Small delay to let cleanup complete
        await asyncio.sleep(0.1)
        
    except Exception as e:
        log_import(f"⚠️ Memory cleanup error: {str(e)[:100]}")

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
    """Process evaluation with production rules and cleaned collection names"""
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
        
        # Complete metadata extraction with agent info
        agent_info = extract_agent_info(evaluation)
        
        meta = {
            "evaluation_id": evaluation.get("evaluationId"),
            "internal_id": evaluation.get("internalId"),
            "template_id": evaluation.get("template_id"),
            "template": evaluation.get("template_name"),
            "program": evaluation.get("partner"),
            "site": evaluation.get("site"),
            "lob": evaluation.get("lob"),
            "agent": evaluation.get("agentName"),
            "agent_id": agent_info["agentId"],  # Add agent ID
            "disposition": evaluation.get("disposition"),
            "sub_disposition": evaluation.get("subDisposition"),
            "language": evaluation.get("language"),
            "call_date": evaluation.get("call_date"),
            "call_duration": evaluation.get("call_duration"),
            "created_on": evaluation.get("created_on")
        }
        
        doc_id = str(evaluation.get("internalId", uuid4()))
        
        # Clean collection name before indexing
        raw_collection = evaluation.get("template_name", "evaluations")
        collection = clean_collection_name(raw_collection)
        
        log_import(f"📁 Collection: '{raw_collection}' -> '{collection}'")
        
        # Index chunks
        indexed_chunks = 0
        for i, chunk in enumerate(all_chunks):
            try:
                # Generate embedding
                embedding = None
                if EMBEDDER_AVAILABLE:
                    try:
                        embedding = embed_text(chunk["text"])
                    except Exception as e:
                        log_import(f"⚠️ Embedding failed for chunk {i}: {str(e)[:50]}")
                
                doc_body = {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "text": chunk["text"],
                    "content_type": chunk["content_type"],
                    "metadata": meta,
                    "source": "evaluation_api",
                    "indexed_at": datetime.now().isoformat(),
                    "chunk_length": chunk["length"],
                    "collection_raw": raw_collection,  # Store original collection name
                    "collection_cleaned": collection   # Store cleaned collection name
                }
                
                if embedding:
                    doc_body["embedding"] = embedding
                
                chunk_id = f"{doc_id}-{chunk['content_type']}-{i}"
                
                # Try to index with retries
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        index_document(chunk_id, doc_body, index_override=collection)
                        indexed_chunks += 1
                        break  # Success, exit retry loop
                    except Exception as index_error:
                        if retry < max_retries - 1:
                            delay = (retry + 1) * 2  # 2, 4, 6 seconds
                            log_import(f"⚠️ Indexing retry {retry + 1}/{max_retries} for chunk {i} in {delay}s: {str(index_error)[:50]}")
                            time.sleep(delay)  # Use sync sleep since we're not in an async context here
                        else:
                            # Final retry failed
                            raise index_error
                
            except Exception as e:
                error_msg = f"Failed to index chunk {i}: {str(e)}"
                log_import(f"❌ {error_msg}")
                
                # Check if it's an OpenSearch connectivity issue
                if any(keyword in str(e).lower() for keyword in ["timeout", "connection", "unreachable", "opensearch"]):
                    raise Exception(f"OpenSearch connection error: {str(e)}")
                
                # Non-OpenSearch errors are logged but don't stop processing
                continue
        
        return {
            "status": "success",
            "document_id": doc_id,
            "chunks_indexed": indexed_chunks,
            "collection": collection,
            "collection_raw": raw_collection,
            "agent_id": agent_info["agentId"],
            "agent_name": agent_info["agentName"],
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
            'User-Agent': 'Ask-InnovAI-Production/3.3.0'
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

async def run_production_import(collection: str = "all", max_docs: int = None, batch_size: int = None):
    """Production import process with enhanced error handling, backpressure, and memory management"""
    import gc
    import os
    
    # Import psutil with fallback
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
        log_import("⚠️ psutil not available - memory monitoring disabled")
    
    try:
        update_import_status("running", "Starting import")
        log_import("🚀 Starting production import with enhanced error handling and memory management")
        
        # Memory management settings
        BATCH_SIZE = batch_size or int(os.getenv("IMPORT_BATCH_SIZE", "5"))
        DELAY_BETWEEN_BATCHES = float(os.getenv("DELAY_BETWEEN_BATCHES", "2.0"))
        DELAY_BETWEEN_DOCS = float(os.getenv("DELAY_BETWEEN_DOCS", "0.5"))
        MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "1"))  # Every N batches
        
        log_import(f"📊 Import configuration:")
        log_import(f"   Batch size: {BATCH_SIZE}")
        log_import(f"   Delay between batches: {DELAY_BETWEEN_BATCHES}s")
        log_import(f"   Delay between docs: {DELAY_BETWEEN_DOCS}s")
        log_import(f"   Memory cleanup interval: {MEMORY_CLEANUP_INTERVAL} batches")
        
        # Get initial memory usage
        initial_memory = None
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                log_import(f"💾 Initial memory usage: {initial_memory:.1f} MB")
            except Exception as e:
                log_import(f"⚠️ Memory monitoring failed: {e}")
                PSUTIL_AVAILABLE = False
        
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
        
        # Process evaluations with better error handling, backpressure, and memory management
        update_import_status("running", f"Processing {len(evaluations)} evaluations in batches of {BATCH_SIZE}")
        
        total_processed = 0
        total_chunks = 0
        errors = 0
        opensearch_errors = 0
        consecutive_opensearch_errors = 0
        batch_count = 0
        collections_processed = set()
        
        for batch_start in range(0, len(evaluations), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(evaluations))
            batch = evaluations[batch_start:batch_end]
            batch_count += 1
            
            log_import(f"📦 Processing batch {batch_count}/{(len(evaluations) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} documents)")
            update_import_status("running", f"Processing batch {batch_count}: documents {batch_start + 1}-{batch_end}/{len(evaluations)}")
            
            # Memory check before batch
            current_memory = None
            if PSUTIL_AVAILABLE:
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    log_import(f"💾 Memory before batch {batch_count}: {current_memory:.1f} MB")
                except Exception:
                    current_memory = None
            
            batch_opensearch_errors = 0
            batch_processed = 0
            batch_chunks = 0
            
            # Process documents in current batch
            for i, evaluation in enumerate(batch):
                actual_index = batch_start + i
                
                try:
                    result = await process_evaluation(evaluation)
                    
                    if result["status"] == "success":
                        batch_processed += 1
                        batch_chunks += result["chunks_indexed"]
                        consecutive_opensearch_errors = 0  # Reset counter on success
                        
                        # Track collections processed
                        if result.get("collection"):
                            collections_processed.add(result["collection"])
                        
                    elif result["status"] == "error":
                        errors += 1
                        error_msg = str(result.get("error", ""))
                        
                        # Check if it's an OpenSearch error
                        if any(keyword in error_msg.lower() for keyword in ["opensearch", "timeout", "connection", "unreachable"]):
                            opensearch_errors += 1
                            consecutive_opensearch_errors += 1
                            batch_opensearch_errors += 1
                            
                            log_import(f"⚠️ OpenSearch error {opensearch_errors} (consecutive: {consecutive_opensearch_errors}): {error_msg[:100]}")
                            
                            # If too many consecutive errors, increase delays
                            if consecutive_opensearch_errors >= 3:
                                delay = min(consecutive_opensearch_errors * 2, 10)
                                log_import(f"🔄 Increasing delay to {delay}s due to consecutive errors")
                                await asyncio.sleep(delay)
                        else:
                            log_import(f"⚠️ Non-OpenSearch error: {error_msg[:100]}")
                    
                    # If too many OpenSearch errors total, stop the import
                    if opensearch_errors > 15:  # Increased threshold
                        error_msg = f"Too many OpenSearch connection errors ({opensearch_errors}). Stopping import."
                        log_import(f"❌ {error_msg}")
                        update_import_status("failed", error=error_msg)
                        return
                    
                    # If too many consecutive errors, stop the import
                    if consecutive_opensearch_errors >= 8:
                        error_msg = f"Too many consecutive OpenSearch errors ({consecutive_opensearch_errors}). Cluster may be unavailable."
                        log_import(f"❌ {error_msg}")
                        update_import_status("failed", error=error_msg)
                        return
                    
                    # Add delay between documents
                    if actual_index < len(evaluations) - 1:  # Don't delay after last document
                        await asyncio.sleep(DELAY_BETWEEN_DOCS)
                
                except Exception as e:
                    errors += 1
                    log_import(f"❌ Unexpected error processing evaluation {actual_index}: {str(e)[:100]}")
            
            # Update totals after batch
            total_processed += batch_processed
            total_chunks += batch_chunks
            
            log_import(f"📊 Batch {batch_count} completed: {batch_processed}/{len(batch)} documents, {batch_chunks} chunks")
            
            # Memory cleanup after batch
            if batch_count % MEMORY_CLEANUP_INTERVAL == 0:
                log_import(f"🧹 Performing memory cleanup after batch {batch_count}")
                await cleanup_memory_after_batch()
                
                # Check memory after cleanup
                if PSUTIL_AVAILABLE:
                    try:
                        memory_after_cleanup = process.memory_info().rss / 1024 / 1024  # MB
                        memory_saved = current_memory - memory_after_cleanup if current_memory else 0
                        log_import(f"💾 Memory after cleanup: {memory_after_cleanup:.1f} MB (saved: {memory_saved:.1f} MB)")
                    except Exception:
                        pass
            
            # If this batch had many OpenSearch errors, increase delay
            if batch_opensearch_errors >= 2:
                extended_delay = DELAY_BETWEEN_BATCHES + (batch_opensearch_errors * 2)
                log_import(f"🔄 Batch had {batch_opensearch_errors} OpenSearch errors, extending delay to {extended_delay}s")
                await asyncio.sleep(extended_delay)
            else:
                # Normal delay between batches
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)
            
            # Clear batch references to help garbage collection
            batch.clear()
            del batch
        
        # Complete with final memory cleanup
        log_import("🧹 Performing final memory cleanup")
        await cleanup_memory_after_batch()
        
        # Final memory check
        final_memory = None
        memory_change = 0
        if PSUTIL_AVAILABLE:
            try:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_change = final_memory - initial_memory if initial_memory else 0
                log_import(f"💾 Final memory usage: {final_memory:.1f} MB (change: {memory_change:+.1f} MB)")
            except Exception:
                pass
        
        results = {
            "total_documents_processed": total_processed,
            "total_chunks_indexed": total_chunks,
            "errors": errors,
            "opensearch_errors": opensearch_errors,
            "import_type": "full",
            "completed_at": datetime.now().isoformat(),
            "success_rate": f"{(total_processed / len(evaluations) * 100):.1f}%" if evaluations else "0%",
            "batch_size": BATCH_SIZE,
            "total_batches": batch_count,
            "collections_processed": list(collections_processed),
            "memory_stats": {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_change_mb": memory_change
            }
        }
        
        log_import(f"🎉 Import completed:")
        log_import(f"   📄 Documents processed: {total_processed}/{len(evaluations)}")
        log_import(f"   🧩 Chunks indexed: {total_chunks}")
        log_import(f"   ❌ Total errors: {errors}")
        log_import(f"   🔌 OpenSearch errors: {opensearch_errors}")
        log_import(f"   📊 Success rate: {results['success_rate']}")
        log_import(f"   📦 Batches processed: {batch_count}")
        log_import(f"   📁 Collections: {len(collections_processed)}")
        log_import(f"   💾 Memory change: {memory_change:+.1f} MB")
        
        update_import_status("completed", results=results)
        
    except Exception as e:
        error_msg = f"Import failed: {str(e)}"
        
        # Check if it's an OpenSearch-related error
        if any(keyword in str(e).lower() for keyword in ["opensearch", "connection", "timeout", "unreachable"]):
            error_msg = f"OpenSearch connection issue: {str(e)}"
            log_import(f"❌ {error_msg}")
            log_import("💡 Suggestions:")
            log_import("   - Check if OpenSearch cluster is healthy")
            log_import("   - Verify network connectivity")
            log_import("   - Consider scaling up the cluster")
            log_import("   - Try reducing import batch size")
        else:
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
        "version": "3.3.0"
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
        <p><strong>Version:</strong> 3.3.0</p>
        <p><strong>New Features:</strong> Clean collection names, AgentId endpoint</p>
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
                "version": "3.3.0"
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
# Additional endpoints for app.py to support aligned filter system
# Add these endpoints to your existing app.py file

from typing import Dict, List, Any, Optional
from fastapi import Query
from collections import defaultdict

# Add these endpoints to your FastAPI app

@app.get("/filter_options_metadata")
async def get_filter_options_metadata():
    """
    Get dynamic filter options based on actual evaluation metadata
    Aligned with call detail structure
    """
    try:
        from opensearch_client import get_opensearch_manager
        manager = get_opensearch_manager()
        
        if not manager.test_connection():
            return create_fallback_filter_options()
        
        # Aggregation query to get unique values from metadata
        aggregation_body = {
            "size": 0,
            "aggs": {
                "programs": {
                    "terms": {"field": "metadata.template.keyword", "size": 50}
                },
                "partners": {
                    "terms": {"field": "metadata.program.keyword", "size": 50}
                },
                "sites": {
                    "terms": {"field": "metadata.site.keyword", "size": 50}
                },
                "lobs": {
                    "terms": {"field": "metadata.lob.keyword", "size": 50}
                },
                "call_dispositions": {
                    "terms": {"field": "metadata.disposition.keyword", "size": 100}
                },
                "call_sub_dispositions": {
                    "terms": {"field": "metadata.sub_disposition.keyword", "size": 200}
                },
                "agent_names": {
                    "terms": {"field": "metadata.agent.keyword", "size": 200}
                },
                "languages": {
                    "terms": {"field": "metadata.language.keyword", "size": 20}
                },
                # Hierarchical relationships
                "program_partners": {
                    "terms": {"field": "metadata.template.keyword", "size": 50},
                    "aggs": {
                        "partners": {
                            "terms": {"field": "metadata.program.keyword", "size": 50}
                        }
                    }
                },
                "partner_sites": {
                    "terms": {"field": "metadata.program.keyword", "size": 50},
                    "aggs": {
                        "sites": {
                            "terms": {"field": "metadata.site.keyword", "size": 50}
                        }
                    }
                },
                "site_lobs": {
                    "terms": {"field": "metadata.site.keyword", "size": 50},
                    "aggs": {
                        "lobs": {
                            "terms": {"field": "metadata.lob.keyword", "size": 50}
                        }
                    }
                },
                "disposition_subdispositions": {
                    "terms": {"field": "metadata.disposition.keyword", "size": 100},
                    "aggs": {
                        "sub_dispositions": {
                            "terms": {"field": "metadata.sub_disposition.keyword", "size": 200}
                        }
                    }
                }
            }
        }
        
        try:
            response = manager.client.search(
                index="*",  # Search all indices
                body=aggregation_body
            )
            
            aggs = response.get("aggregations", {})
            
            # Extract simple lists
            filter_options = {
                "programs": extract_terms(aggs.get("programs", {})),
                "partners": extract_terms(aggs.get("partners", {})),
                "sites": extract_terms(aggs.get("sites", {})),
                "lobs": extract_terms(aggs.get("lobs", {})),
                "callDispositions": extract_terms(aggs.get("call_dispositions", {})),
                "callSubDispositions": extract_terms(aggs.get("call_sub_dispositions", {})),
                "agentNames": extract_terms(aggs.get("agent_names", {})),
                "languages": extract_terms(aggs.get("languages", {})),
                "callTypes": ["Direct Connect", "Transfer", "Inbound", "Outbound"],  # Static for now
                "agentDispositions": ["Equipment", "Account Management", "Technical Support", "Customer Service"],  # Static for now
                "agentSubDispositions": ["NA", "Resolved", "Escalated", "Follow-up Required", "Transferred"]  # Static for now
            }
            
            # Extract hierarchical relationships
            hierarchy = {
                "program_partners": extract_hierarchy(aggs.get("program_partners", {}), "partners"),
                "partner_sites": extract_hierarchy(aggs.get("partner_sites", {}), "sites"),
                "site_lobs": extract_hierarchy(aggs.get("site_lobs", {}), "lobs"),
                "disposition_subdispositions": extract_hierarchy(aggs.get("disposition_subdispositions", {}), "sub_dispositions")
            }
            
            return {
                **filter_options,
                "hierarchy": hierarchy,
                "status": "success",
                "source": "dynamic_metadata"
            }
            
        except Exception as e:
            logger.error(f"OpenSearch aggregation failed: {e}")
            return create_fallback_filter_options()
            
    except Exception as e:
        logger.error(f"Filter options endpoint failed: {e}")
        return create_fallback_filter_options()

def extract_terms(aggregation_data: Dict) -> List[str]:
    """Extract terms from OpenSearch aggregation buckets"""
    buckets = aggregation_data.get("buckets", [])
    return [bucket["key"] for bucket in buckets if bucket["key"] and bucket["key"].strip()]

def extract_hierarchy(aggregation_data: Dict, sub_agg_name: str) -> Dict[str, List[str]]:
    """Extract hierarchical relationships from nested aggregations"""
    hierarchy = {}
    buckets = aggregation_data.get("buckets", [])
    
    for bucket in buckets:
        parent_key = bucket["key"]
        if parent_key and parent_key.strip():
            sub_agg = bucket.get(sub_agg_name, {})
            children = extract_terms(sub_agg)
            if children:
                hierarchy[parent_key] = children
    
    return hierarchy

def create_fallback_filter_options():
    """Fallback filter options when OpenSearch is not available"""
    return {
        "programs": [
            "Ai Corporate SPTR - TEST",
            "Customer Service Quality", 
            "Technical Support QA",
            "Billing Specialist Review"
        ],
        "partners": [
            "iQor", "Teleperformance", "Concentrix", "Alorica", "Sitel"
        ],
        "sites": [
            "Dasma", "Manila", "Cebu", "Davao", "Iloilo", "Bacolod"
        ],
        "lobs": [
            "WNP", "Prepaid", "Postpaid", "Business", "Enterprise"
        ],
        "callDispositions": [
            "Account", "Technical Support", "Billing", "Port Out", "Service Inquiry", "Equipment"
        ],
        "callSubDispositions": [
            "Rate Plan Or Plan Fit Analysis",
            "Port Out - Questions/pin/acct #",
            "Account - Profile Update",
            "Equipment - Troubleshooting"
        ],
        "agentNames": [
            "Rey Mendoza", "Maria Garcia", "John Smith", "Sarah Johnson"
        ],
        "languages": [
            "English", "Spanish", "Tagalog"
        ],
        "callTypes": [
            "Direct Connect", "Transfer", "Inbound", "Outbound"
        ],
        "agentDispositions": [
            "Equipment", "Account Management", "Technical Support"
        ],
        "agentSubDispositions": [
            "NA", "Resolved", "Escalated"
        ],
        "hierarchy": {},
        "status": "fallback",
        "source": "static_data"
    }

@app.post("/analytics/stats")
async def get_analytics_stats(request: Dict[str, Any]):
    """
    Get statistics based on aligned metadata filters
    """
    try:
        filters = request.get("filters", {})
        filter_version = request.get("filter_version", "4.0")
        
        logger.info(f"📊 Getting stats with filter version {filter_version}")
        logger.info(f"🔍 Applied filters: {list(filters.keys())}")
        
        from opensearch_client import get_opensearch_manager
        manager = get_opensearch_manager()
        
        if not manager.test_connection():
            return {"totalRecords": 0, "status": "opensearch_unavailable"}
        
        # Build OpenSearch query based on aligned filters
        query_body = build_aligned_filter_query(filters)
        
        # Get count
        try:
            response = manager.client.count(
                index="*",
                body={"query": query_body}
            )
            
            total_records = response.get("count", 0)
            
            return {
                "totalRecords": total_records,
                "status": "success",
                "filtersApplied": len(filters),
                "filterVersion": filter_version,
                "queryType": "aligned_metadata"
            }
            
        except Exception as e:
            logger.error(f"OpenSearch count query failed: {e}")
            return {"totalRecords": 0, "status": "query_failed", "error": str(e)}
            
    except Exception as e:
        logger.error(f"Stats endpoint failed: {e}")
        return {"totalRecords": 0, "status": "error", "error": str(e)}

def build_aligned_filter_query(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build OpenSearch query from aligned metadata filters
    """
    must_clauses = []
    
    # Date range filters
    if filters.get("call_date_start") or filters.get("call_date_end"):
        date_range = {}
        if filters.get("call_date_start"):
            date_range["gte"] = filters["call_date_start"]
        if filters.get("call_date_end"):
            date_range["lte"] = filters["call_date_end"]
        
        must_clauses.append({
            "range": {
                "metadata.call_date": date_range
            }
        })
    
    # Organizational hierarchy filters
    hierarchy_fields = {
        "program": "metadata.template.keyword",
        "partner": "metadata.program.keyword", 
        "site": "metadata.site.keyword",
        "lob": "metadata.lob.keyword"
    }
    
    for filter_key, es_field in hierarchy_fields.items():
        if filters.get(filter_key):
            must_clauses.append({
                "term": {es_field: filters[filter_key]}
            })
    
    # Call identifier filters (exact matches)
    id_fields = {
        "phone_number": "metadata.phone_number.keyword",
        "contact_id": "metadata.contact_id.keyword",
        "ucid": "metadata.ucid.keyword",
        "user_id": "metadata.agent_id.keyword"
    }
    
    for filter_key, es_field in id_fields.items():
        if filters.get(filter_key):
            must_clauses.append({
                "term": {es_field: filters[filter_key]}
            })
    
    # Call classification filters
    classification_fields = {
        "call_disposition": "metadata.disposition.keyword",
        "call_sub_disposition": "metadata.sub_disposition.keyword",
        "agent_disposition": "metadata.agent_disposition.keyword",
        "call_type": "metadata.call_type.keyword",
        "call_language": "metadata.language.keyword"
    }
    
    for filter_key, es_field in classification_fields.items():
        if filters.get(filter_key):
            must_clauses.append({
                "term": {es_field: filters[filter_key]}
            })
    
    # Agent name filter (partial match)
    if filters.get("agent_name"):
        must_clauses.append({
            "wildcard": {
                "metadata.agent.keyword": f"*{filters['agent_name']}*"
            }
        })
    
    # Duration range filter
    if filters.get("min_duration") or filters.get("max_duration"):
        duration_range = {}
        if filters.get("min_duration"):
            duration_range["gte"] = filters["min_duration"]
        if filters.get("max_duration"):
            duration_range["lte"] = filters["max_duration"]
        
        must_clauses.append({
            "range": {
                "metadata.call_duration": duration_range
            }
        })
    
    # Evaluation metadata filters
    eval_fields = {
        "evaluation_id": "metadata.evaluation_id.keyword",
        "internal_id": "metadata.internal_id.keyword",
        "template_id": "metadata.template_id.keyword"
    }
    
    for filter_key, es_field in eval_fields.items():
        if filters.get(filter_key):
            must_clauses.append({
                "term": {es_field: filters[filter_key]}
            })
    
    # If no filters, match all
    if not must_clauses:
        return {"match_all": {}}
    
    return {
        "bool": {
            "must": must_clauses
        }
    }

@app.get("/metadata/validation")
async def validate_metadata_structure():
    """
    Validate that the metadata structure aligns with expected call detail fields
    """
    try:
        from opensearch_client import get_opensearch_manager
        manager = get_opensearch_manager()
        
        if not manager.test_connection():
            return {"status": "opensearch_unavailable"}
        
        # Sample a few documents to check metadata structure
        sample_query = {
            "size": 10,
            "query": {"match_all": {}},
            "_source": ["metadata"]
        }
        
        response = manager.client.search(
            index="*",
            body=sample_query
        )
        
        hits = response.get("hits", {}).get("hits", [])
        
        if not hits:
            return {"status": "no_documents", "message": "No documents found for validation"}
        
        # Analyze metadata structure
        metadata_fields = set()
        sample_metadata = []
        
        for hit in hits:
            metadata = hit.get("_source", {}).get("metadata", {})
            if metadata:
                metadata_fields.update(metadata.keys())
                sample_metadata.append(metadata)
        
        # Expected fields based on call detail structure
        expected_fields = {
            "call_date", "disposition", "sub_disposition", "call_duration",
            "agent", "site", "program", "lob", "language", "template",
            "evaluation_id", "internal_id", "template_id"
        }
        
        # Optional fields that might be present
        optional_fields = {
            "phone_number", "contact_id", "ucid", "agent_id", "call_type",
            "agent_disposition", "agent_sub_disposition"
        }
        
        all_expected = expected_fields | optional_fields
        
        validation_result = {
            "status": "success",
            "total_metadata_fields": len(metadata_fields),
            "expected_fields_present": len(metadata_fields & expected_fields),
            "optional_fields_present": len(metadata_fields & optional_fields),
            "missing_expected": list(expected_fields - metadata_fields),
            "missing_optional": list(optional_fields - metadata_fields),
            "unexpected_fields": list(metadata_fields - all_expected),
            "sample_metadata": sample_metadata[:3],  # First 3 samples
            "field_coverage": {
                "expected": f"{len(metadata_fields & expected_fields)}/{len(expected_fields)}",
                "optional": f"{len(metadata_fields & optional_fields)}/{len(optional_fields)}",
                "total": f"{len(metadata_fields & all_expected)}/{len(all_expected)}"
            }
        }
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Metadata validation failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/filter/hierarchy/{level}")
async def get_hierarchy_options(
    level: str,
    program: Optional[str] = Query(None),
    partner: Optional[str] = Query(None),
    site: Optional[str] = Query(None)
):
    """
    Get hierarchical filter options based on parent selections
    Supports Program -> Partner -> Site -> LOB hierarchy
    """
    try:
        from opensearch_client import get_opensearch_manager
        manager = get_opensearch_manager()
        
        if not manager.test_connection():
            return {"options": [], "status": "opensearch_unavailable"}
        
        # Build query based on parent selections
        must_clauses = []
        
        if program:
            must_clauses.append({"term": {"metadata.template.keyword": program}})
        if partner:
            must_clauses.append({"term": {"metadata.program.keyword": partner}})
        if site:
            must_clauses.append({"term": {"metadata.site.keyword": site}})
        
        # Determine aggregation field based on level
        field_mapping = {
            "partner": "metadata.program.keyword",
            "site": "metadata.site.keyword", 
            "lob": "metadata.lob.keyword"
        }
        
        if level not in field_mapping:
            return {"options": [], "status": "invalid_level", "valid_levels": list(field_mapping.keys())}
        
        agg_field = field_mapping[level]
        
        query_body = {
            "size": 0,
            "query": {
                "bool": {"must": must_clauses} if must_clauses else {"match_all": {}}
            },
            "aggs": {
                "options": {
                    "terms": {"field": agg_field, "size": 100}
                }
            }
        }
        
        response = manager.client.search(
            index="*",
            body=query_body
        )
        
        options = extract_terms(response.get("aggregations", {}).get("options", {}))
        
        return {
            "options": options,
            "level": level,
            "parent_filters": {"program": program, "partner": partner, "site": site},
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Hierarchy options failed: {e}")
        return {"options": [], "status": "error", "error": str(e)}

# Add these utility functions if they don't exist
async def cleanup_memory_after_batch():
    """Memory cleanup function if not already defined"""
    import gc
    collected = gc.collect()
    logger.info(f"🧹 Garbage collected {collected} objects")

logger.info("✅ Aligned filter system endpoints loaded")
logger.info("📊 Supports hierarchical filtering: Program -> Partner -> Site -> LOB") 
logger.info("🔍 Supports call identifiers: Phone, Contact ID, UCID, User ID")
logger.info("📋 Supports call classifications: Dispositions, Types, Languages")
logger.info("📈 Supports metadata validation and dynamic option loading")

@app.get("/search")
async def search(q: str):
    """Production search with cleaned collection names"""
    try:
        # Search in your main data index
        results = search_opensearch(q, index_override="ai-corporate-sptr-test")
        formatted_results = []
        
        for hit in results:
            source = hit.get('_source', {})
            metadata = source.get('metadata', {})
            
            formatted_results.append({
                "id": hit.get('_id'),
                "title": f"{metadata.get('agent', 'Agent')} - {metadata.get('disposition', 'Call')}",
                "text": source.get('text', ''),
                "score": hit.get('_score', 0),
                "metadata": metadata,
                "collection": source.get('collection_cleaned', source.get('collection_raw', 'unknown')),
                "agent_id": metadata.get('agent_id', 'unknown')
            })
        
        return {
            "status": "success", 
            "results": formatted_results,
            "total_hits": len(formatted_results)
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"status": "error", "error": str(e), "results": []}

@app.get("/agent/{agent_id}")
async def get_agent_info(agent_id: str):
    """Get agent information and related evaluations"""
    try:
        # Search for evaluations by agent ID
        query = f"metadata.agent_id:\"{agent_id}\""
        
        # Use OpenSearch to find evaluations for this agent
        from opensearch_client import get_opensearch_manager
        manager = get_opensearch_manager()
        
        # Search across all collections
        results = manager.search(query, size=50)
        
        if not results:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"No evaluations found for agent ID: {agent_id}"
                }
            )
        
        # Extract agent information from results
        agent_info = {}
        evaluations = []
        
        for hit in results:
            source = hit.get('_source', {})
            metadata = source.get('metadata', {})
            
            # Build agent info from first result
            if not agent_info:
                agent_info = {
                    "agentId": agent_id,
                    "agentName": metadata.get('agent', 'Unknown'),
                    "site": metadata.get('site', ''),
                    "partner": metadata.get('program', ''),
                    "lob": metadata.get('lob', ''),
                    "total_evaluations": 0,
                    "collections": set(),
                    "date_range": {
                        "earliest": None,
                        "latest": None
                    }
                }
            
            # Add evaluation info
            evaluation_data = {
                "document_id": source.get('document_id'),
                "evaluation_id": metadata.get('evaluation_id'),
                "internal_id": metadata.get('internal_id'),
                "collection": source.get('collection_cleaned', 'unknown'),
                "disposition": metadata.get('disposition', ''),
                "sub_disposition": metadata.get('sub_disposition', ''),
                "call_date": metadata.get('call_date', ''),
                "call_duration": metadata.get('call_duration', 0),
                "content_type": source.get('content_type', ''),
                "score": hit.get('_score', 0)
            }
            
            evaluations.append(evaluation_data)
            agent_info["collections"].add(source.get('collection_cleaned', 'unknown'))
            
            # Update date range
            call_date = metadata.get('call_date')
            if call_date:
                if not agent_info["date_range"]["earliest"] or call_date < agent_info["date_range"]["earliest"]:
                    agent_info["date_range"]["earliest"] = call_date
                if not agent_info["date_range"]["latest"] or call_date > agent_info["date_range"]["latest"]:
                    agent_info["date_range"]["latest"] = call_date
        
        # Convert set to list
        agent_info["collections"] = list(agent_info["collections"])
        agent_info["total_evaluations"] = len(set(e["document_id"] for e in evaluations if e["document_id"]))
        
        return {
            "status": "success",
            "agent": agent_info,
            "evaluations": evaluations,
            "total_chunks": len(evaluations)
        }
        
    except Exception as e:
        logger.error(f"Agent info error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to get agent info: {str(e)}"
            }
        )

@app.post("/search/agents")
async def search_agents(request: AgentSearchRequest):
    """Search for agents with filters"""
    try:
        # Build search query
        must_clauses = []
        
        if request.agentId:
            must_clauses.append({"term": {"metadata.agent_id.keyword": request.agentId}})
        
        if request.agentName:
            must_clauses.append({"wildcard": {"metadata.agent.keyword": f"*{request.agentName}*"}})
        
        if request.site:
            must_clauses.append({"term": {"metadata.site.keyword": request.site}})
        
        if request.partner:
            must_clauses.append({"term": {"metadata.program.keyword": request.partner}})
        
        if request.lob:
            must_clauses.append({"term": {"metadata.lob.keyword": request.lob}})
        
        # If no specific filters, search all
        if not must_clauses:
            must_clauses = [{"match_all": {}}]
        
        # Use OpenSearch aggregations to get unique agents
        from opensearch_client import get_opensearch_manager
        manager = get_opensearch_manager()
        
        # Search with aggregations to get unique agents
        body = {
            "size": 0,  # We don't need individual hits
            "query": {
                "bool": {
                    "must": must_clauses
                }
            },
            "aggs": {
                "unique_agents": {
                    "terms": {
                        "field": "metadata.agent_id.keyword",
                        "size": request.limit or 10
                    },
                    "aggs": {
                        "agent_info": {
                            "top_hits": {
                                "size": 1,
                                "_source": ["metadata.agent", "metadata.site", "metadata.program", "metadata.lob", "metadata.agent_id"]
                            }
                        },
                        "evaluation_count": {
                            "cardinality": {
                                "field": "document_id.keyword"
                            }
                        }
                    }
                }
            }
        }
        
        if not manager.test_connection():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "OpenSearch connection not available"
                }
            )
        
        response = manager.client.search(
            index="*",  # Search all indices
            body=body
        )
        
        # Process aggregation results
        agents = []
        
        for bucket in response.get("aggregations", {}).get("unique_agents", {}).get("buckets", []):
            agent_id = bucket["key"]
            doc_count = bucket["doc_count"]
            eval_count = bucket.get("evaluation_count", {}).get("value", 0)
            
            # Get agent info from top hit
            top_hit = bucket.get("agent_info", {}).get("hits", {}).get("hits", [])
            if top_hit:
                metadata = top_hit[0].get("_source", {}).get("metadata", {})
                
                agents.append({
                    "agentId": agent_id,
                    "agentName": metadata.get("agent", "Unknown"),
                    "site": metadata.get("site", ""),
                    "partner": metadata.get("program", ""),
                    "lob": metadata.get("lob", ""),
                    "total_chunks": doc_count,
                    "total_evaluations": eval_count
                })
        
        return {
            "status": "success",
            "agents": agents,
            "total_found": len(agents)
        }
        
    except Exception as e:
        logger.error(f"Agent search error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": f"Agent search failed: {str(e)}"
            }
        )

@app.get("/agents/list")
async def list_all_agents(
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0)
):
    """List all agents with pagination"""
    try:
        from opensearch_client import get_opensearch_manager
        manager = get_opensearch_manager()
        
        if not manager.test_connection():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "OpenSearch connection not available"
                }
            )
        
        # Get all unique agents with pagination
        body = {
            "size": 0,
            "query": {"match_all": {}},
            "aggs": {
                "unique_agents": {
                    "terms": {
                        "field": "metadata.agent_id.keyword",
                        "size": limit + offset + 50  # Get more than needed for pagination
                    },
                    "aggs": {
                        "agent_info": {
                            "top_hits": {
                                "size": 1,
                                "_source": ["metadata"]
                            }
                        },
                        "evaluation_count": {
                            "cardinality": {
                                "field": "document_id.keyword"
                            }
                        }
                    }
                }
            }
        }
        
        response = manager.client.search(
            index="*",
            body=body
        )
        
        # Process results
        all_agents = []
        
        for bucket in response.get("aggregations", {}).get("unique_agents", {}).get("buckets", []):
            agent_id = bucket["key"]
            doc_count = bucket["doc_count"]
            eval_count = bucket.get("evaluation_count", {}).get("value", 0)
            
            top_hit = bucket.get("agent_info", {}).get("hits", {}).get("hits", [])
            if top_hit:
                metadata = top_hit[0].get("_source", {}).get("metadata", {})
                
                all_agents.append({
                    "agentId": agent_id,
                    "agentName": metadata.get("agent", "Unknown"),
                    "site": metadata.get("site", ""),
                    "partner": metadata.get("program", ""),
                    "lob": metadata.get("lob", ""),
                    "total_chunks": doc_count,
                    "total_evaluations": eval_count
                })
        
        # Apply pagination
        paginated_agents = all_agents[offset:offset + limit]
        
        return {
            "status": "success",
            "agents": paginated_agents,
            "pagination": {
                "offset": offset,
                "limit": limit,
                "total": len(all_agents),
                "returned": len(paginated_agents)
            }
        }
        
    except Exception as e:
        logger.error(f"List agents error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to list agents: {str(e)}"
            }
        )

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
        background_tasks.add_task(run_production_import, request.collection, request.max_docs, request.batch_size)
        
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

@app.get("/collections")
async def get_collections():
    """Get list of available collections"""
    try:
        from opensearch_client import get_opensearch_manager
        manager = get_opensearch_manager()
        
        if not manager.test_connection():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "OpenSearch connection not available"
                }
            )
        
        # Get all indices
        indices = manager.client.indices.get_alias(index="*")
        
        # Filter out system indices
        user_indices = [name for name in indices.keys() if not name.startswith('.')]
        
        # Get document counts for each index
        collections = []
        for index in user_indices:
            try:
                stats = manager.client.indices.stats(index=index)
                doc_count = stats["_all"]["primaries"]["docs"]["count"]
                
                collections.append({
                    "name": index,
                    "document_count": doc_count,
                    "status": "active"
                })
            except Exception as e:
                collections.append({
                    "name": index,
                    "document_count": 0,
                    "status": "error",
                    "error": str(e)[:100]
                })
        
        return {
            "status": "success",
            "collections": collections,
            "total_collections": len(collections)
        }
        
    except Exception as e:
        logger.error(f"Collections endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to get collections: {str(e)}"
            }
        )

@app.get("/test_collection_cleanup")
async def test_collection_cleanup():
    """Test collection name cleanup function"""
    test_cases = [
        "Ai Corporate SPTR - TEST",
        "Customer Service & Support",
        "Metro_T-Mobile_Evaluations",
        "Special Characters !@#$%^&*()",
        "   Spaces at Start and End   ",
        "Multiple---Hyphens",
        "UPPERCASE_MIXED_case",
        "",
        "1234567890123456789012345678901234567890123456789012345678901234567890",
        "Normal Collection Name"
    ]
    
    results = []
    for test_name in test_cases:
        cleaned = clean_collection_name(test_name)
        results.append({
            "original": test_name,
            "cleaned": cleaned,
            "length": len(cleaned)
        })
    
    return {
        "status": "success",
        "test_results": results,
        "function_info": {
            "description": "Collection names are cleaned by removing spaces, special characters, and ensuring OpenSearch compatibility",
            "rules": [
                "Convert to lowercase",
                "Replace spaces and underscores with hyphens",
                "Remove special characters except hyphens and alphanumeric",
                "Remove multiple consecutive hyphens",
                "Remove leading/trailing hyphens",
                "Ensure starts with alphanumeric character",
                "Limit length to 50 characters"
            ]
        }
    }

# Existing endpoints continue...
@app.get("/memory_stats")
async def get_memory_stats():
    """Get current memory usage statistics"""
    try:
        # Try to import psutil
        try:
            import psutil
            import gc
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            memory_usage = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "total_mb": psutil.virtual_memory().total / 1024 / 1024
            }
            
            garbage_collection = {
                "generation_0": gc.get_count()[0],
                "generation_1": gc.get_count()[1],
                "generation_2": gc.get_count()[2]
            }
            
        except ImportError:
            memory_usage = {
                "error": "psutil not available",
                "message": "Install psutil for memory monitoring: pip install psutil"
            }
            
            garbage_collection = {
                "error": "psutil not available"
            }
        
        # Get embedding service stats if available
        embedding_cache_size = 0
        embedding_cache_hits = 0
        if EMBEDDER_AVAILABLE:
            try:
                from embedder import get_embedding_stats
                stats = get_embedding_stats()
                embedding_cache_size = stats.get('lru_cache_info', {}).get('size', 0)
                embedding_cache_hits = stats.get('lru_cache_hits', 0)
            except:
                pass
        
        # Get OpenSearch stats
        opensearch_stats = {}
        try:
            from opensearch_client import get_opensearch_stats
            opensearch_stats = get_opensearch_stats()
        except:
            pass
        
        return {
            "status": "success",
            "memory_usage": memory_usage,
            "cache_stats": {
                "embedding_cache_size": embedding_cache_size,
                "embedding_cache_hits": embedding_cache_hits,
                "opensearch_operations": opensearch_stats.get('total_operations', 0),
                "opensearch_success_rate": opensearch_stats.get('success_rate', '0%')
            },
            "garbage_collection": garbage_collection
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/clear_cache")
async def clear_cache():
    """Manually clear caches and perform garbage collection"""
    try:
        await cleanup_memory_after_batch()
        return {"status": "success", "message": "Cache cleared and garbage collection performed"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/import_configuration")
async def get_import_configuration():
    """Get current import configuration settings"""
    return {
        "status": "success",
        "configuration": {
            "default_batch_size": int(os.getenv("IMPORT_BATCH_SIZE", "5")),
            "delay_between_batches": float(os.getenv("DELAY_BETWEEN_BATCHES", "2.0")),
            "delay_between_docs": float(os.getenv("DELAY_BETWEEN_DOCS", "0.5")),
            "memory_cleanup_interval": int(os.getenv("MEMORY_CLEANUP_INTERVAL", "1")),
            "max_opensearch_errors": 15,
            "max_consecutive_errors": 8,
            "collection_name_cleaning": True,
            "agent_id_generation": True,
            "environment_variables": {
                "IMPORT_BATCH_SIZE": os.getenv("IMPORT_BATCH_SIZE", "5"),
                "DELAY_BETWEEN_BATCHES": os.getenv("DELAY_BETWEEN_BATCHES", "2.0"),
                "DELAY_BETWEEN_DOCS": os.getenv("DELAY_BETWEEN_DOCS", "0.5"),
                "MEMORY_CLEANUP_INTERVAL": os.getenv("MEMORY_CLEANUP_INTERVAL", "1")
            }
        }
    }

@app.get("/import_statistics")
async def get_import_statistics():
    """Get import statistics with memory information"""
    try:
        # Get basic import stats
        basic_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "last_import": None,
            "success_rate": "0%",
            "batch_size": None,
            "total_batches": None,
            "collections_processed": [],
            "memory_stats": None
        }
        
        if import_status.get("results"):
            results = import_status["results"]
            basic_stats.update({
                "total_documents": results.get("total_documents_processed", 0),
                "total_chunks": results.get("total_chunks_indexed", 0),
                "last_import": import_status.get("end_time"),
                "success_rate": results.get("success_rate", "0%"),
                "batch_size": results.get("batch_size"),
                "total_batches": results.get("total_batches"),
                "collections_processed": results.get("collections_processed", []),
                "memory_stats": results.get("memory_stats")
            })
        
        # Get current memory stats
        try:
            import psutil
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / 1024 / 1024
            basic_stats["current_memory_mb"] = current_memory
        except ImportError:
            basic_stats["current_memory_mb"] = "psutil not available"
        except:
            pass
        
        return {
            "status": "success",
            "statistics": basic_stats
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/test_opensearch")
async def test_opensearch_connection():
    """Test OpenSearch connectivity and list indices"""
    try:
        from opensearch_client import test_connection, get_connection_status, get_opensearch_config, get_opensearch_manager
        
        config = get_opensearch_config()
        manager = get_opensearch_manager()
        
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
                    "ssl_enabled": config["use_ssl"],
                    "details": {
                        "tested": conn_status.get("tested", False),
                        "last_test": conn_status.get("last_test"),
                        "connection_working": False
                    }
                }
            )
        
        # Get list of indices
        try:
            indices = manager.client.indices.get_alias(index="*")
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
                    "ssl_enabled": config["use_ssl"],
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
                    "ssl_enabled": config["use_ssl"],
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
                "use_ssl": False
            }
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"OpenSearch test failed: {str(e)}",
                "host": config["host"],
                "port": config["port"],
                "url": config["url"],
                "ssl_enabled": config["use_ssl"],
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
        logger.info(f"   Version: 3.3.0")
        logger.info(f"   Port: {os.getenv('PORT', '8080')}")
        logger.info(f"   New Features: Collection cleanup, AgentId endpoints")
        
        # Check configuration without blocking
        api_configured = bool(API_AUTH_VALUE)
        genai_configured = bool(GENAI_ACCESS_KEY)
        opensearch_configured = bool(os.getenv("OPENSEARCH_HOST"))
        
        logger.info(f"   API Source: {'✅ Configured' if api_configured else '❌ Missing'}")
        logger.info(f"   GenAI: {'✅ Configured' if genai_configured else '❌ Missing'}")
        logger.info(f"   OpenSearch: {'✅ Configured' if opensearch_configured else '❌ Missing'}")
        
        # Test collection cleanup function
        test_collection = "Ai Corporate SPTR - TEST"
        cleaned_collection = clean_collection_name(test_collection)
        logger.info(f"   Collection cleanup: '{test_collection}' -> '{cleaned_collection}'")
        
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
        
        # Log memory management features
        try:
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024
            logger.info(f"💾 Initial memory usage: {initial_memory:.1f} MB")
            logger.info("🧹 Memory management features enabled")
        except ImportError:
            logger.warning("⚠️ psutil not available - memory monitoring disabled")
            logger.info("💡 Install psutil for memory monitoring: pip install psutil")
        except Exception as e:
            logger.warning(f"⚠️ Memory monitoring setup failed: {e}")
        
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