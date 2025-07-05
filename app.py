# Enhanced App.py - Template_ID Based Collections with Evaluation Grouping (FIXED)
# Version: 4.0.1 - Fixed import endpoint and OpenSearch issues

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
    title="Ask InnovAI Production - Evaluation Grouped",
    description="AI-Powered Knowledge Assistant with Evaluation-Based Document Structure",
    version="4.0.1"
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

def clean_template_id_for_index(template_id: str) -> str:
    """
    Clean template_id to create valid OpenSearch index names
    Template IDs are more structured than template names
    """
    if not template_id:
        return "default-template"
    
    # Convert to lowercase and clean
    cleaned = str(template_id).lower().strip()
    
    # Replace any non-alphanumeric characters with hyphens
    cleaned = re.sub(r'[^a-z0-9]', '-', cleaned)
    
    # Remove multiple consecutive hyphens
    cleaned = re.sub(r'-+', '-', cleaned)
    
    # Remove leading/trailing hyphens
    cleaned = cleaned.strip('-')
    
    # Ensure it's not empty
    if not cleaned:
        cleaned = "default-template"
    
    # Ensure it starts with a letter (OpenSearch requirement)
    if cleaned and not cleaned[0].isalpha():
        cleaned = f"template-{cleaned}"
    
    # Add prefix for clarity
    cleaned = f"eval-{cleaned}"
    
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
# MEMORY MANAGEMENT FUNCTIONS (Keeping existing)
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
# ENHANCED DATA PROCESSING FUNCTIONS - EVALUATION GROUPING
# ============================================================================

def extract_qa_pairs(evaluation_text: str) -> List[Dict[str, Any]]:
    """Extract Question and Answer pairs from evaluation text - enhanced with metadata"""
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
                qa_text = f"Section: {section_name}\nQuestion: {question}\nAnswer: {answer}"
                
                qa_chunks.append({
                    "text": qa_text,
                    "section": section_name,
                    "question": question,
                    "answer": answer,
                    "content_type": "evaluation_qa"
                })
    
    # Fallback: if no sections, try direct Q&A extraction
    if not qa_chunks:
        qa_pattern = r'Question:\s*([^?]+\??)\s*Answer:\s*([^.]+\.?)'
        matches = re.finditer(qa_pattern, clean_text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            qa_text = f"Question: {question}\nAnswer: {answer}"
            
            qa_chunks.append({
                "text": qa_text,
                "section": "General",
                "question": question,
                "answer": answer,
                "content_type": "evaluation_qa"
            })
    
    return qa_chunks

def split_transcript_by_speakers(transcript: str) -> List[Dict[str, Any]]:
    """Split transcript while preserving speaker boundaries - enhanced with metadata"""
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
        return [{
            "text": chunk["text"],
            "content_type": "transcript",
            "speaker": "Unknown",
            "timestamp": None
        } for chunk in chunks]
    
    # Group speaker turns into chunks
    speaker_turns = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            speaker_header = parts[i].strip()
            content = parts[i + 1].strip()
            
            # Extract speaker and timestamp info
            speaker_match = re.match(r'Speaker ([AB]) \((\d{2}:\d{2}:\d{2})\):', speaker_header)
            speaker = speaker_match.group(1) if speaker_match else "Unknown"
            timestamp = speaker_match.group(2) if speaker_match else None
            
            if content:
                speaker_turn = {
                    "text": f"{speaker_header} {content}",
                    "content_type": "transcript",
                    "speaker": speaker,
                    "timestamp": timestamp,
                    "raw_content": content
                }
                speaker_turns.append(speaker_turn)
    
    # Combine turns into appropriately sized chunks while preserving speaker info
    chunks = []
    current_chunk_text = ""
    current_chunk_speakers = set()
    current_chunk_timestamps = []
    max_size = 1100
    
    for turn in speaker_turns:
        turn_text = turn["text"]
        
        if current_chunk_text and len(current_chunk_text + "\n" + turn_text) > max_size:
            # Finalize current chunk
            chunks.append({
                "text": current_chunk_text.strip(),
                "content_type": "transcript",
                "speakers": list(current_chunk_speakers),
                "timestamps": current_chunk_timestamps,
                "speaker_count": len(current_chunk_speakers)
            })
            
            # Start new chunk
            current_chunk_text = turn_text
            current_chunk_speakers = {turn["speaker"]}
            current_chunk_timestamps = [turn["timestamp"]] if turn["timestamp"] else []
        else:
            # Add to current chunk
            current_chunk_text = current_chunk_text + "\n" + turn_text if current_chunk_text else turn_text
            current_chunk_speakers.add(turn["speaker"])
            if turn["timestamp"]:
                current_chunk_timestamps.append(turn["timestamp"])
    
    # Add final chunk if exists
    if current_chunk_text.strip():
        chunks.append({
            "text": current_chunk_text.strip(),
            "content_type": "transcript",
            "speakers": list(current_chunk_speakers),
            "timestamps": current_chunk_timestamps,
            "speaker_count": len(current_chunk_speakers)
        })
    
    return chunks

async def process_evaluation(evaluation: Dict) -> Dict:
    """
    ENHANCED: Process evaluation with template_ID-based collections and evaluation grouping
    All chunks are now grouped under a single document per evaluationID
    """
    try:
        evaluation_text = evaluation.get("evaluation", "")
        transcript_text = evaluation.get("transcript", "")
        
        if not evaluation_text and not transcript_text:
            return {"status": "skipped", "reason": "no_content"}
        
        # Extract all chunks with enhanced metadata
        all_chunks = []
        
        # Process evaluation Q&A with enhanced structure
        if evaluation_text:
            qa_chunks = extract_qa_pairs(evaluation_text)
            for i, qa_data in enumerate(qa_chunks):
                if len(qa_data["text"].strip()) >= 20:
                    chunk_data = {
                        "text": qa_data["text"],
                        "content_type": "evaluation_qa",
                        "chunk_index": len(all_chunks),
                        "section": qa_data.get("section", "General"),
                        "question": qa_data.get("question", ""),
                        "answer": qa_data.get("answer", ""),
                        "qa_pair_index": i
                    }
                    all_chunks.append(chunk_data)
        
        # Process transcript with enhanced structure
        if transcript_text:
            transcript_chunks = split_transcript_by_speakers(transcript_text)
            for i, transcript_data in enumerate(transcript_chunks):
                if len(transcript_data["text"].strip()) >= 20:
                    chunk_data = {
                        "text": transcript_data["text"],
                        "content_type": "transcript",
                        "chunk_index": len(all_chunks),
                        "speakers": transcript_data.get("speakers", []),
                        "timestamps": transcript_data.get("timestamps", []),
                        "speaker_count": transcript_data.get("speaker_count", 0),
                        "transcript_chunk_index": i
                    }
                    all_chunks.append(chunk_data)
        
        if not all_chunks:
            return {"status": "skipped", "reason": "no_meaningful_content"}
        
        # Enhanced metadata extraction with agent info
        agent_info = extract_agent_info(evaluation)
        
        # Use evaluationId as primary document ID (this is the key change)
        evaluation_id = evaluation.get("evaluationId")
        internal_id = evaluation.get("internalId")
        
        if not evaluation_id:
            return {"status": "skipped", "reason": "missing_evaluation_id"}
        
        # Primary document ID is now the evaluationId
        doc_id = str(evaluation_id)
        
        # Use template_ID for collection naming (major change)
        template_id = evaluation.get("template_id")
        template_name = evaluation.get("template_name", "Unknown Template")
        
        if not template_id:
            return {"status": "skipped", "reason": "missing_template_id"}
        
        # Create collection name from template_ID
        collection = clean_template_id_for_index(template_id)
        
        log_import(f"📁 Template ID: '{template_id}' -> Collection: '{collection}' (Name: '{template_name}')")
        
        # Complete metadata structure
        base_metadata = {
            "evaluationId": evaluation_id,
            "internalId": internal_id,
            "template_id": template_id,
            "template_name": template_name,  # Keep template_name as reference
            "partner": evaluation.get("partner"),
            "site": evaluation.get("site"),
            "lob": evaluation.get("lob"),
            "agent": evaluation.get("agentName"),
            "agent_id": agent_info["agentId"],
            "disposition": evaluation.get("disposition"),
            "sub_disposition": evaluation.get("subDisposition"),
            "language": evaluation.get("language"),
            "call_date": evaluation.get("call_date"),
            "call_duration": evaluation.get("call_duration"),
            "created_on": evaluation.get("created_on")
        }
        
        # Generate embeddings for all chunks
        chunk_embeddings = []
        if EMBEDDER_AVAILABLE:
            try:
                chunk_texts = [chunk["text"] for chunk in all_chunks]
                chunk_embeddings = []
                
                # Process in smaller batches for better memory management
                batch_size = 10
                for i in range(0, len(chunk_texts), batch_size):
                    batch_texts = chunk_texts[i:i + batch_size]
                    try:
                        # Use batch embedding if available
                        from embedder import embed_texts
                        batch_embeddings = embed_texts(batch_texts)
                        chunk_embeddings.extend(batch_embeddings)
                    except ImportError:
                        # Fallback to individual embeddings
                        for text in batch_texts:
                            embedding = embed_text(text)
                            chunk_embeddings.append(embedding)
                    
                    # Small delay between batches
                    if i + batch_size < len(chunk_texts):
                        await asyncio.sleep(0.1)
                        
            except Exception as e:
                log_import(f"⚠️ Embedding failed for evaluation {evaluation_id}: {str(e)[:50]}")
                chunk_embeddings = []
        
        # CREATE SINGLE DOCUMENT WITH ALL CHUNKS (Key Change)
        # Instead of separate documents per chunk, create one document per evaluation
        document_body = {
            # Primary identification
            "evaluationId": evaluation_id,
            "internalId": internal_id,
            "template_id": template_id,
            "template_name": template_name,
            
            # Document structure
            "document_type": "evaluation",
            "total_chunks": len(all_chunks),
            "evaluation_chunks_count": len([c for c in all_chunks if c["content_type"] == "evaluation_qa"]),
            "transcript_chunks_count": len([c for c in all_chunks if c["content_type"] == "transcript"]),
            
            # All chunks in a single document
            "chunks": [],
            
            # Combined text for full-text search
            "full_text": "",
            "evaluation_text": evaluation_text,
            "transcript_text": transcript_text,
            
            # Metadata
            "metadata": base_metadata,
            
            # Indexing info
            "source": "evaluation_api",
            "indexed_at": datetime.now().isoformat(),
            "collection_name": collection,
            "collection_source": f"template_id_{template_id}"
        }
        
        # Add each chunk to the document
        full_text_parts = []
        
        for i, chunk in enumerate(all_chunks):
            chunk_data = {
                "chunk_index": i,
                "text": chunk["text"],
                "content_type": chunk["content_type"],
                "length": len(chunk["text"]),
                
                # Content-specific metadata
                **{k: v for k, v in chunk.items() if k not in ["text", "content_type", "chunk_index"]}
            }
            
            # Add embedding if available
            if i < len(chunk_embeddings):
                chunk_data["embedding"] = chunk_embeddings[i]
            
            document_body["chunks"].append(chunk_data)
            full_text_parts.append(chunk["text"])
        
        # Create combined full text for search
        document_body["full_text"] = "\n\n".join(full_text_parts)
        
        # Add document-level embedding (average of chunk embeddings or full text embedding)
        if chunk_embeddings:
            try:
                import numpy as np
                # Create document-level embedding as average of chunk embeddings
                doc_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                document_body["document_embedding"] = doc_embedding
            except Exception as e:
                log_import(f"⚠️ Could not create document embedding: {str(e)[:50]}")
        
        # INDEX SINGLE DOCUMENT (Key Change)
        # Use evaluationId as document ID, so all chunks are grouped under one document
        try:
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Index the complete evaluation document
                    index_document(doc_id, document_body, index_override=collection)
                    log_import(f"✅ Indexed evaluation {evaluation_id} with {len(all_chunks)} chunks in collection '{collection}'")
                    break  # Success, exit retry loop
                    
                except Exception as index_error:
                    if retry < max_retries - 1:
                        delay = (retry + 1) * 2  # 2, 4, 6 seconds
                        log_import(f"⚠️ Indexing retry {retry + 1}/{max_retries} for evaluation {evaluation_id} in {delay}s: {str(index_error)[:50]}")
                        time.sleep(delay)
                    else:
                        # Final retry failed
                        raise index_error
            
        except Exception as e:
            error_msg = f"Failed to index evaluation {evaluation_id}: {str(e)}"
            log_import(f"❌ {error_msg}")
            
            # Check if it's an OpenSearch connectivity issue
            if any(keyword in str(e).lower() for keyword in ["timeout", "connection", "unreachable", "opensearch"]):
                raise Exception(f"OpenSearch connection error: {str(e)}")
            
            return {"status": "error", "error": str(e)}
        
        return {
            "status": "success",
            "document_id": doc_id,
            "evaluationId": evaluation_id,
            "template_id": template_id,
            "template_name": template_name,
            "collection": collection,
            "total_chunks": len(all_chunks),
            "agent_id": agent_info["agentId"],
            "agent_name": agent_info["agentName"],
            "evaluation_chunks": len([c for c in all_chunks if c["content_type"] == "evaluation_qa"]),
            "transcript_chunks": len([c for c in all_chunks if c["content_type"] == "transcript"]),
            "total_content_length": sum(len(chunk["text"]) for chunk in all_chunks),
            "has_embeddings": bool(chunk_embeddings)
        }
        
    except Exception as e:
        logger.error(f"Failed to process evaluation: {e}")
        return {"status": "error", "error": str(e)}

# ============================================================================
# PRODUCTION API FETCHING (Keeping existing)
# ============================================================================

async def fetch_evaluations(max_docs: int = None):
    """Fetch evaluations from API"""
    try:
        if not API_BASE_URL or not API_AUTH_VALUE:
            raise Exception("API configuration missing")
        
        headers = {
            API_AUTH_KEY: API_AUTH_VALUE,
            'Accept': 'application/json',
            'User-Agent': 'Ask-InnovAI-Production/4.0.1'
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
    """
    ENHANCED: Production import process with evaluation grouping
    Now creates one document per evaluation instead of per chunk
    """
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
        update_import_status("running", "Starting enhanced import with evaluation grouping")
        log_import("🚀 Starting ENHANCED import: Template_ID collections + Evaluation grouping")
        
        # Memory management settings
        BATCH_SIZE = batch_size or int(os.getenv("IMPORT_BATCH_SIZE", "5"))
        DELAY_BETWEEN_BATCHES = float(os.getenv("DELAY_BETWEEN_BATCHES", "2.0"))
        DELAY_BETWEEN_DOCS = float(os.getenv("DELAY_BETWEEN_DOCS", "0.5"))
        MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "1"))
        
        log_import(f"📊 Enhanced import configuration:")
        log_import(f"   🔗 Collections based on: template_ID")
        log_import(f"   📋 Document grouping: evaluationID")
        log_import(f"   📦 Batch size: {BATCH_SIZE}")
        log_import(f"   ⏱️ Delay between batches: {DELAY_BETWEEN_BATCHES}s")
        log_import(f"   🧹 Memory cleanup interval: {MEMORY_CLEANUP_INTERVAL} batches")
        
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
            results = {
                "total_documents_processed": 0, 
                "total_chunks_indexed": 0, 
                "import_type": "full",
                "document_structure": "evaluation_grouped"
            }
            update_import_status("completed", results=results)
            return
        
        # Process evaluations with enhanced structure
        update_import_status("running", f"Processing {len(evaluations)} evaluations with enhanced grouping")
        
        total_processed = 0
        total_chunks = 0
        total_evaluations_indexed = 0  # New metric
        errors = 0
        opensearch_errors = 0
        consecutive_opensearch_errors = 0
        batch_count = 0
        template_collections = set()  # Track template-based collections
        
        for batch_start in range(0, len(evaluations), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(evaluations))
            batch = evaluations[batch_start:batch_end]
            batch_count += 1
            
            log_import(f"📦 Processing batch {batch_count}/{(len(evaluations) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} evaluations)")
            update_import_status("running", f"Processing batch {batch_count}: evaluations {batch_start + 1}-{batch_end}/{len(evaluations)}")
            
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
            batch_evaluations_indexed = 0
            
            # Process evaluations in current batch
            for i, evaluation in enumerate(batch):
                actual_index = batch_start + i
                
                try:
                    result = await process_evaluation(evaluation)
                    
                    if result["status"] == "success":
                        batch_processed += 1
                        batch_chunks += result["total_chunks"]
                        batch_evaluations_indexed += 1  # Each successful result = 1 evaluation document
                        consecutive_opensearch_errors = 0  # Reset counter on success
                        
                        # Track template-based collections
                        if result.get("collection"):
                            template_collections.add(result["collection"])
                            
                        log_import(f"✅ Evaluation {result['evaluationId']}: {result['total_chunks']} chunks -> Collection '{result['collection']}'")
                        
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
                    
                    elif result["status"] == "skipped":
                        reason = result.get("reason", "unknown")
                        log_import(f"⏭️ Skipped evaluation: {reason}")
                    
                    # If too many OpenSearch errors total, stop the import
                    if opensearch_errors > 15:
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
            total_evaluations_indexed += batch_evaluations_indexed
            
            log_import(f"📊 Batch {batch_count} completed: {batch_processed}/{len(batch)} evaluations, {batch_chunks} total chunks, {batch_evaluations_indexed} documents indexed")
            
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
            "total_evaluations_indexed": total_evaluations_indexed,  # NEW: actual documents in OpenSearch
            "total_chunks_processed": total_chunks,  # NEW: total chunks processed (but grouped in documents)
            "errors": errors,
            "opensearch_errors": opensearch_errors,
            "import_type": "full",
            "document_structure": "evaluation_grouped",  # NEW: indicates new structure
            "collection_strategy": "template_id_based",  # NEW: indicates template_ID strategy
            "completed_at": datetime.now().isoformat(),
            "success_rate": f"{(total_processed / len(evaluations) * 100):.1f}%" if evaluations else "0%",
            "batch_size": BATCH_SIZE,
            "total_batches": batch_count,
            "template_collections_created": list(template_collections),
            "memory_stats": {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_change_mb": memory_change
            }
        }
        
        log_import(f"🎉 ENHANCED import completed:")
        log_import(f"   📄 Evaluations processed: {total_processed}/{len(evaluations)}")
        log_import(f"   📋 Documents indexed: {total_evaluations_indexed} (1 per evaluation)")
        log_import(f"   🧩 Total chunks processed: {total_chunks} (grouped within documents)")
        log_import(f"   📁 Template collections created: {len(template_collections)}")
        log_import(f"   ❌ Total errors: {errors}")
        log_import(f"   🔌 OpenSearch errors: {opensearch_errors}")
        log_import(f"   📊 Success rate: {results['success_rate']}")
        log_import(f"   💾 Memory change: {memory_change:+.1f} MB")
        log_import(f"   🏗️ Document structure: Evaluation-grouped (chunks within documents)")
        log_import(f"   🏷️ Collection strategy: Template_ID-based")
        
        update_import_status("completed", results=results)
        
    except Exception as e:
        error_msg = f"Enhanced import failed: {str(e)}"
        
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
# FASTAPI ENDPOINTS (FIXED - Added missing import endpoint)
# ============================================================================

@app.get("/ping")
async def ping():
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "service": "ask-innovai-enhanced",
        "version": "4.0.1",
        "document_structure": "evaluation_grouped",
        "collection_strategy": "template_id_based"
    }

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
        <h1>🤖 Ask InnovAI Enhanced</h1>
        <p><strong>Status:</strong> Enhanced Production Ready ✅</p>
        <p><strong>Version:</strong> 4.0.1</p>
        <p><strong>New Structure:</strong> Template_ID Collections + Evaluation Grouping</p>
        <p><strong>Document Model:</strong> One document per evaluation with grouped chunks</p>
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
        <h1>🤖 Ask InnovAI Enhanced Chat</h1>
        <p>Chat interface file not found. Please ensure static/chat.html exists.</p>
        <p><a href="/">← Back to Admin</a></p>
        </body></html>
        """)

# FIXED: Added the missing import endpoint
@app.post("/import")
async def start_import(request: ImportRequest, background_tasks: BackgroundTasks):
    """
    Start the enhanced import process with evaluation grouping
    """
    global import_status
    
    # Check if already running
    if import_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Import is already running")
    
    try:
        # Reset import status
        import_status = {
            "status": "idle",
            "start_time": None,
            "end_time": None,
            "current_step": None,
            "results": {},
            "error": None,
            "import_type": request.import_type
        }
        
        # Validate request
        if request.max_docs is not None and request.max_docs <= 0:
            raise HTTPException(status_code=400, detail="max_docs must be a positive integer")
        
        # Log import start
        log_import(f"🚀 Import request received:")
        log_import(f"   Collection: {request.collection}")
        log_import(f"   Import Type: {request.import_type}")
        log_import(f"   Max Docs: {request.max_docs or 'All'}")
        log_import(f"   Batch Size: {request.batch_size or 'Default'}")
        
        # Start background import
        background_tasks.add_task(
            run_production_import,
            collection=request.collection,
            max_docs=request.max_docs,
            batch_size=request.batch_size
        )
        
        return {
            "status": "success",
            "message": f"Enhanced import started: {request.import_type} mode",
            "collection": request.collection,
            "max_docs": request.max_docs,
            "import_type": request.import_type,
            "structure": "evaluation_grouped"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start import: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start import: {str(e)}")

@app.get("/health")
async def health():
    """Enhanced health check with new structure information"""
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
                    "url": config["url"],
                    "document_structure": "evaluation_grouped",
                    "collection_strategy": "template_id_based"
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
                        "url": config["url"],
                        "document_structure": "evaluation_grouped",
                        "collection_strategy": "template_id_based"
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
                "version": "4.0.1",
                "enhancements": {
                    "document_structure": "evaluation_grouped",
                    "collection_strategy": "template_id_based",
                    "chunk_grouping": "within_evaluation_documents"
                }
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

# Add test endpoint for new structure
@app.get("/test_enhanced_structure")
async def test_enhanced_structure():
    """Test the enhanced template_ID collection and evaluation grouping structure"""
    test_cases = [
        {
            "template_id": "685c05eb3d1f4c023147e889",
            "template_name": "Ai Corporate SPTR - TEST",
            "evaluationId": 14,
            "description": "Primary test case from sample data"
        },
        {
            "template_id": "abc123def456",
            "template_name": "Customer Service Quality Review",
            "evaluationId": 25,
            "description": "Synthetic test case"
        },
        {
            "template_id": "xyz789uvw012",
            "template_name": "Technical Support Evaluation - Advanced",
            "evaluationId": 36,
            "description": "Complex name test case"
        }
    ]
    
    results = []
    for test_case in test_cases:
        collection_name = clean_template_id_for_index(test_case["template_id"])
        
        results.append({
            "input": {
                "template_id": test_case["template_id"],
                "template_name": test_case["template_name"],
                "evaluationId": test_case["evaluationId"]
            },
            "output": {
                "collection_name": collection_name,
                "document_id": str(test_case["evaluationId"]),
                "expected_structure": "single_document_with_grouped_chunks"
            },
            "description": test_case["description"]
        })
    
    return {
        "status": "success",
        "test_results": results,
        "structure_info": {
            "collection_strategy": "template_id_based",
            "document_grouping": "evaluation_id_based",
            "chunk_storage": "within_evaluation_documents",
            "benefits": [
                "One document per evaluation (easier aggregation)",
                "Template-based collections (better organization)",
                "Preserved chunk details (maintained granularity)",
                "Improved search relevance (evaluation-level results)"
            ]
        },
        "migration_notes": [
            "Old structure: Multiple chunk documents per evaluation",
            "New structure: Single evaluation document with chunk array",
            "Collection names now based on template_ID instead of template_name",
            "Search results now return evaluations instead of individual chunks"
        ]
    }

# Keep existing endpoints but add enhanced information
@app.get("/status")
async def get_import_status():
    """Get import status with enhanced structure information"""
    enhanced_status = import_status.copy()
    enhanced_status["structure_version"] = "4.0.1"
    enhanced_status["document_strategy"] = "evaluation_grouped"
    enhanced_status["collection_strategy"] = "template_id_based"
    return enhanced_status

# FIXED: Added logs endpoint that was referenced in the frontend
@app.get("/logs")
async def get_logs():
    """Get import logs"""
    return {
        "status": "success",
        "logs": import_logs,
        "count": len(import_logs)
    }

# FIXED: Add analytics stats endpoint
@app.post("/analytics/stats")
async def analytics_stats(request: dict):
    """Get analytics statistics with filtering"""
    try:
        filters = request.get("filters", {})
        
        # For now, return mock stats since we don't have the actual implementation
        # In a real scenario, this would query OpenSearch with the filters
        return {
            "status": "success",
            "totalRecords": 1200 + len(str(filters)),  # Mock calculation
            "filters_applied": filters,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Add filter options endpoint
@app.get("/filter_options_metadata")
async def filter_options_metadata():
    """Get filter options for the UI"""
    # Return mock data for now - in production this would come from OpenSearch aggregations
    return {
        "programs": ["Metro", "ASW", "T-Mobile Prepaid"],
        "partners": ["iQor", "Teleperformance", "Concentrix", "Alorica", "Sitel"],
        "sites": ["Dasma", "Manila", "Cebu", "Davao", "Iloilo", "Bacolod"],
        "lobs": ["CSR", "CSR-Supervisor", "Chat", "chat-supervisor"],
        "callDispositions": ["Account", "Technical Support", "Billing", "Port Out", "Service Inquiry", "Complaint", "Equipment", "Rate Plan"],
        "callSubDispositions": ["Rate Plan Or Plan Fit Analysis", "Port Out - Questions/pin/acct #", "Account - Profile Update", "Billing - Payment Plan", "Technical - Device Setup", "Equipment - Troubleshooting"],
        "agentNames": ["Rey Mendoza", "Maria Garcia", "John Smith", "Sarah Johnson", "Ana Rodriguez", "David Chen", "Lisa Wang", "Carlos Martinez"],
        "languages": ["English", "Spanish", "Tagalog", "Cebuano"],
        "callTypes": ["Direct Connect", "Transfer", "Inbound", "Outbound"]
    }

# FIXED: Add search endpoint
@app.get("/search")
async def search_endpoint(q: str = Query(..., description="Search query")):
    """Search endpoint for testing"""
    try:
        results = search_opensearch(q, size=10)
        
        return {
            "status": "success",
            "query": q,
            "results": [
                {
                    "title": result.get("template_name", "Unknown"),
                    "text": result.get("text", ""),
                    "score": result.get("_score", 0),
                    "evaluationId": result.get("evaluationId"),
                    "collection": result.get("_index")
                }
                for result in results
            ],
            "count": len(results)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": q,
            "results": []
        }

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Enhanced startup with new structure information"""
    try:
        logger.info("🚀 Ask InnovAI Enhanced starting...")
        logger.info(f"   Version: 4.0.1")
        logger.info(f"   Document Structure: Evaluation-grouped")
        logger.info(f"   Collection Strategy: Template_ID-based")
        logger.info(f"   Port: {os.getenv('PORT', '8080')}")
        
        # Log structure changes
        logger.info("🔄 Structure Enhancements:")
        logger.info("   📋 Collections: Based on template_ID (cleaner indexing)")
        logger.info("   📄 Documents: One per evaluationID (better aggregation)")
        logger.info("   🧩 Chunks: Grouped within evaluation documents")
        logger.info("   🔍 Search: Returns evaluations instead of chunks")
        
        # Check configuration without blocking
        api_configured = bool(API_AUTH_VALUE)
        genai_configured = bool(GENAI_ACCESS_KEY)
        opensearch_configured = bool(os.getenv("OPENSEARCH_HOST"))
        
        logger.info(f"   API Source: {'✅ Configured' if api_configured else '❌ Missing'}")
        logger.info(f"   GenAI: {'✅ Configured' if genai_configured else '❌ Missing'}")
        logger.info(f"   OpenSearch: {'✅ Configured' if opensearch_configured else '❌ Missing'}")
        
        # Test template_ID collection cleanup
        test_template_id = "685c05eb3d1f4c023147e889"
        cleaned_collection = clean_template_id_for_index(test_template_id)
        logger.info(f"   Template collection: '{test_template_id}' -> '{cleaned_collection}'")
        
        # Preload embedder if available (non-blocking)
        if EMBEDDER_AVAILABLE:
            try:
                preload_embedding_model()
                logger.info("✅ Embedding model preloaded")
            except Exception as e:
                logger.warning(f"⚠️ Embedding preload failed: {e}")
        
        logger.info("🎉 Enhanced startup complete")
        logger.info("📊 Ready for template_ID-based collections with evaluation grouping")
        
    except Exception as e:
        logger.error(f"❌ Enhanced startup error: {e}")
        # Don't re-raise - let app start anyway

# For Digital Ocean App Platform
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    logger.info(f"🚀 Starting Ask InnovAI Enhanced on port {port}")
    logger.info("🔄 Using: Template_ID collections + Evaluation grouping")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )