#!/usr/bin/env python3
"""
app.py - Vector DB Importer - FIXED VECTOR SERIALIZATION
Properly converts numpy embeddings to JSON-serializable lists
"""

import os
import json
import re
import sys
import logging
import asyncio 
import gc
import numpy as np  # Added for proper numpy handling
from datetime import datetime
from typing import Optional, List, Dict, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from collections import defaultdict
import requests
from bs4 import BeautifulSoup

from embedder import embed_texts as get_embeddings, get_embedding_dimension, get_empty_vector
from sentence_splitter import split_into_chunks
from opensearch_client import index_chunks
from opensearch_client import client as opensearch_client

# Use the embedding service from embedder module directly
from embedder import get_embedding_service
from embedder import preload_embedding_model

from datetime import datetime, timezone
from dateutil.parser import parse as parse_date


app = FastAPI(title="Vector DB Importer", version="3.1.1-fixed-vector-serialization")

# Ensure static directory exists
import pathlib
static_dir = pathlib.Path("static")
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_origins=["http://localhost:3000", "http://localhost:8000"],  # Restrict to specific domains will be added later behind firewall right now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for tracking last import timestamp
last_import_info = {
    "timestamp": None,
    "type": None
}

@app.get("/last_import_info", response_class=JSONResponse)
async def get_last_import_info():
    """Get information about the last import"""
    try:
        return {
            "status": "success",
            "last_import_timestamp": last_import_info.get("timestamp"),
            "last_import_type": last_import_info.get("type")
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/clear_import_timestamp", response_class=JSONResponse)
async def clear_import_timestamp():
    """Clear the last import timestamp"""
    try:
        global last_import_info
        last_import_info = {"timestamp": None, "type": None}
        return {"status": "success", "message": "Import timestamp cleared"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/check_document_updates/{collection}", response_class=JSONResponse)
async def check_document_updates(collection: str):
    """Check for document updates in a collection"""
    try:
        # Mock implementation - you'll need to implement the actual logic
        # based on your MongoDB API
        
        api_base = os.getenv("API_BASE_URL")
        if not api_base:
            raise ValueError("API_BASE_URL not configured")
        
        headers = {
            os.getenv("API_AUTH_KEY", "Authorization"): os.getenv("API_AUTH_VALUE", "token")
        }
        
        # Fetch documents from API
        documents = fetch_collection_data(api_base, collection, headers)
        
        # Check against last import timestamp
        last_timestamp = last_import_info.get("timestamp")
        updated_documents = 0
        updated_document_list = []
        
        if last_timestamp:
            from dateutil.parser import parse as parse_date
            cutoff_date = parse_date(last_timestamp)
            
            for doc in documents:
                if doc.get("updated"):
                    try:
                        doc_updated = parse_date(doc["updated"])
                        if doc_updated > cutoff_date:
                            updated_documents += 1
                            if len(updated_document_list) < 10:  # Limit to 10 for display
                                updated_document_list.append({
                                    "id": doc.get("id"),
                                    "name": doc.get("name"),
                                    "updated": doc.get("updated")
                                })
                    except:
                        continue
        else:
            # No previous import, all documents are "updated"
            updated_documents = len(documents)
            updated_document_list = documents[:10]
        
        return {
            "status": "success",
            "collection": collection,
            "total_documents": len(documents),
            "updated_documents": updated_documents,
            "updates_available": updated_documents > 0,
            "last_import_timestamp": last_timestamp,
            "updated_document_list": updated_document_list,
            "total_updated_count": updated_documents
        }
        
    except Exception as e:
        logger.error(f"Error checking document updates: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/import_statistics", response_class=JSONResponse)
async def get_import_statistics():
    """Get import statistics from OpenSearch"""
    try:
        from opensearch_client import get_import_statistics
        stats = get_import_statistics()
        
        return {
            "status": "success",
            "statistics": stats,
            "last_import_timestamp": last_import_info.get("timestamp")
        }
    except Exception as e:
        logger.error(f"Error getting import statistics: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/cleanup_old_chunks", response_class=JSONResponse)
async def cleanup_old_chunks(request: dict):
    """Clean up old chunks from OpenSearch"""
    try:
        max_age_days = request.get("max_age_days", 30)
        
        # Validate input
        if not isinstance(max_age_days, int) or max_age_days < 1:
            raise ValueError("max_age_days must be a positive integer")
        
        from opensearch_client import cleanup_orphaned_chunks
        chunks_cleaned = cleanup_orphaned_chunks(max_age_days)
        
        return {
            "status": "success",
            "chunks_cleaned": chunks_cleaned,
            "max_age_days": max_age_days
        }
    except Exception as e:
        logger.error(f"Error cleaning up old chunks: {e}")
        return {"status": "error", "error": str(e)}
    
@app.get("/available_collections", response_class=JSONResponse)
async def get_available_collections():
    """Get available collections from API - simple version"""
    try:
        api_base = os.getenv("API_BASE_URL")
        if not api_base:
            return {"status": "error", "error": "API not configured"}
        
        headers = {os.getenv("API_AUTH_KEY", "Authorization"): os.getenv("API_AUTH_VALUE", "token")}
        collections = fetch_dynamic_endpoints(api_base, headers)
        collections.insert(0, "all")  # Add "all" option
        
        return {"status": "success", "collections": collections}
    except Exception as e:
        # Return fallback collections if API fails
        fallback = ["all", "announcements", "call-assist", "devices", "mpower", "plans", 
                   "quality-pages", "services", "ttk", "wnp-pages", "apps", "promotions"]
        return {"status": "success", "collections": fallback, "note": "Using fallback collections"}

@app.get("/preview_cleanup/{max_age_days}", response_class=JSONResponse)
async def preview_cleanup(max_age_days: int):
    """Preview what would be cleaned up"""
    try:
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        
        # Query OpenSearch for cleanup preview
        from opensearch_client import client as opensearch_client
        
        # Get total chunks
        total_response = opensearch_client.count(index="kb-*")
        total_chunks = total_response.get("count", 0)
        
        # Get old chunks count
        old_query = {
            "query": {
                "range": {
                    "indexed_at": {
                        "lt": cutoff_date.isoformat()
                    }
                }
            }
        }
        
        old_response = opensearch_client.count(index="kb-*", body=old_query)
        old_chunks = old_response.get("count", 0)
        
        chunks_to_keep = total_chunks - old_chunks
        percentage_to_delete = (old_chunks / total_chunks * 100) if total_chunks > 0 else 0
        
        return {
            "status": "success",
            "preview": {
                "cutoff_date": cutoff_date.isoformat(),
                "total_chunks": total_chunks,
                "old_chunks_to_delete": old_chunks,
                "chunks_to_keep": chunks_to_keep,
                "percentage_to_delete": round(percentage_to_delete, 2),
                "is_safe": percentage_to_delete < 50,  # Safe if deleting less than 50%
                "collection_breakdown": {}  # Could add per-collection breakdown
            }
        }
        
    except Exception as e:
        logger.error(f"Error previewing cleanup: {e}")
        return {"status": "error", "error": str(e)}
    
def is_document_updated_since_last_import(doc_updated: str, last_import: str) -> bool:
    """
    FIXED: Check if a document has been updated since the last import
    Enhanced with better logging for debugging
    """
    try:
        add_log(f"[DATE_DEBUG] === DEBUGGING DATE COMPARISON ===")
        add_log(f"[DATE_DEBUG] doc_updated type: {type(doc_updated)}")
        add_log(f"[DATE_DEBUG] doc_updated repr: {repr(doc_updated)}")
        add_log(f"[DATE_DEBUG] last_import type: {type(last_import)}")
        add_log(f"[DATE_DEBUG] last_import repr: {repr(last_import)}")

        # Handle missing timestamps
        if not doc_updated or not last_import:
            add_log(f"[DATE_DEBUG] Missing timestamps - PROCESSING (fallback)")
            return True

        # Normalize input
        doc_updated_str = str(doc_updated).strip()
        last_import_str = str(last_import).strip()
        
        add_log(f"[DATE_DEBUG] After normalization:")
        add_log(f"[DATE_DEBUG] doc_updated_str: '{doc_updated_str}'")
        add_log(f"[DATE_DEBUG] last_import_str: '{last_import_str}'")

        # Check for default/placeholder timestamps
        default_timestamps = [
            "2025-01-01T00:00:00.000Z", "2025-01-01T00:00:00.000z",
            "2025-01-01T00:00:00Z", "2025-01-01T00:00:00z",
            "2025-01-01 00:00:00.000", "2025-01-01 00:00:00",
            "2024-01-01T00:00:00.000Z", "2023-01-01T00:00:00.000Z",
            "1970-01-01T00:00:00.000Z", "0000-00-00T00:00:00.000Z"
        ]

        # Test exact matches (case-insensitive)
        for default_ts in default_timestamps:
            if doc_updated_str.lower() == default_ts.lower():
                add_log(f"[DATE_DEBUG] MATCHED default timestamp '{default_ts}' - SKIPPING document")
                return False
        
        # Test prefix matches for common default dates
        default_date_prefixes = ["2025-01-01", "2024-01-01", "2023-01-01", "1970-01-01"]
        for prefix in default_date_prefixes:
            if doc_updated_str.startswith(prefix):
                add_log(f"[DATE_DEBUG] Document date starts with default prefix '{prefix}' - SKIPPING document")
                return False

        add_log(f"[DATE_DEBUG] Not a default timestamp - proceeding with date comparison")

        # Parse and compare dates
        try:
            doc_date = parse_date(doc_updated_str)
            import_date = parse_date(last_import_str)
            add_log(f"[DATE_DEBUG] Successfully parsed dates")
        except Exception as parse_error:
            add_log(f"[DATE_DEBUG] Failed to parse dates: {parse_error}")
            add_log(f"[DATE_DEBUG] Falling back to PROCESSING document")
            return True

        # Normalize timezones to UTC
        if doc_date.tzinfo is None:
            doc_date = doc_date.replace(tzinfo=timezone.utc)
        else:
            doc_date = doc_date.astimezone(timezone.utc)
        
        if import_date.tzinfo is None:
            import_date = import_date.replace(tzinfo=timezone.utc)
        else:
            import_date = import_date.astimezone(timezone.utc)

        add_log(f"[DATE_DEBUG] Normalized dates:")
        add_log(f"[DATE_DEBUG] doc_date (UTC): {doc_date.isoformat()}")
        add_log(f"[DATE_DEBUG] import_date (UTC): {import_date.isoformat()}")

        # Compare the dates
        result = doc_date > import_date
        
        add_log(f"[DATE_DEBUG] Final comparison: {doc_date.isoformat()} > {import_date.isoformat()} = {result}")
        add_log(f"[DATE_DEBUG] Decision: {'PROCESS' if result else 'SKIP'} document")
        add_log(f"[DATE_DEBUG] === END DEBUG ===")
        
        return result

    except Exception as e:
        add_log(f"[DATE_DEBUG] UNEXPECTED ERROR in date comparison: {e}")
        add_log(f"[DATE_DEBUG] Falling back to PROCESSING document")
        add_log(f"[DATE_DEBUG] === END DEBUG (ERROR) ===")
        return True


def get_last_import_timestamp():
    """
    Get the last import timestamp with proper timezone handling
    FIXED: Return timezone-aware timestamp consistently
    """
    global last_import_info
    timestamp = last_import_info.get("timestamp")
    
    add_log(f"[TIMESTAMP_DEBUG] Global last_import_info: {last_import_info}")
    add_log(f"[TIMESTAMP_DEBUG] Retrieved timestamp: {timestamp}")
    
    if not timestamp:
        return None
    
    # Ensure the timestamp is timezone-aware
    try:
        parsed_ts = parse_date(timestamp)
        if parsed_ts.tzinfo is None:
            # If timezone-naive, assume UTC
            parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC if it has a different timezone
            parsed_ts = parsed_ts.astimezone(timezone.utc)
        
        # Return as ISO format string
        return parsed_ts.isoformat()
    except Exception as e:
        add_log(f"[TIMESTAMP] Warning: Could not parse stored timestamp '{timestamp}': {e}")
        return timestamp  # Return as-is if parsing fails


def set_last_import_timestamp():
    """
    Set the last import timestamp with proper timezone
    FIXED: Always store timezone-aware UTC timestamp
    """
    global last_import_info
    
    # Always use UTC timezone-aware timestamp
    current_timestamp = datetime.now(timezone.utc).isoformat()
    last_import_info["timestamp"] = current_timestamp
    last_import_info["type"] = "incremental"  # Track the type
    
    add_log(f"[TIMESTAMP] Last import timestamp set to: {current_timestamp}")
    return current_timestamp


# OPTIONAL: Helper function to test date comparison without running full import
def test_date_comparison(doc_updated: str, last_import: str) -> dict:
    """
    Test function to debug date comparison without running import
    """
    try:
        result = is_document_updated_since_last_import(doc_updated, last_import)
        return {
            "doc_updated": doc_updated,
            "last_import": last_import,
            "result": result,
            "decision": "PROCESS" if result else "SKIP",
            "status": "success"
        }
    except Exception as e:
        return {
            "doc_updated": doc_updated,
            "last_import": last_import,
            "result": None,
            "decision": "ERROR",
            "error": str(e),
            "status": "error"
        }

import_status = {
    "status": "idle",
    "current_step": "",
    "start_time": None,
    "end_time": None,
    "results": {},
    "logs": [],
    "error": None
}

# LOB mapping dictionary as requested
LOB_MAPPING = {
    "5615548fc5ecb7841c40eda5": "CSR",
    "5615548fc5ecb7841c40eda6": "DSG",
    "5615548fc5ecb7841c40eda7": "Chat",
    "5615548fc5ecb7841c40eda8": "Exigent/911 Team",
    "5615548fc5ecb7841c40eda9": "Mentor",
    "5615548fc5ecb7841c40edaa": "Welcome Calls",
    "5615548fc5ecb7841c40edab": "WNP",
    "5615548fc5ecb7841c40edac": "Supervisor",
    "5615548fc5ecb7841c40edad": "NRG",
    "5615548fc5ecb7841c40edae": "CAT",
    "5615548fc5ecb7841c40edaf": "QA",
    "58286433cd9b4ef7003820ea": "Release Cycle 0",
    "58e6c3a1eaf21ff600229382": "Smartphone Support",
    "5a144f44f539ff16008565ef": "Training",
    "5ae9db459ee83e06bbd66d32": "Support",
    "5af46f2b20a10e06a2c79cc9": "Service Experts Chat",
    "5af46f4c20a10e06a2c79cca": "Troubleshooting-SPL",
    "5af46f5b20a10e06a2c79ccb": "ID Management",
    "5af46f9420a10e06a2c79ccc": "Troubleshooting",
    "5af475ba20a10e06a2c79ccd": "Social Media",
    "5b1a9d6e4a79d506b0f6a8eb": "Landline",
    "5b1aa1614a79d506b0f6a8f9": "Provisioning",
    "5b3b7eb147fff706b0fcdde1": "Metro Sales",
    "5bbfa069471cd6581fc92189": "Leadership",
    "5bbfaa0ea258e059daf87c78": "Error",
    "5cc85a31b54c940b80ac894a": "Training Manager",
    "5cc85a3ab54c940b80ac894b": "QA Manager",
    "5cc85a4cb54c940b80ac894c": "Operations Manager",
    "5cc85a97b54c940b80ac894d": "Site Director",
    "5cf816ef29060d062df83812": "Client Care",
    "5d015ecaf3395e1233514b77": "AOY",
    "5e78bdbb6e7e50297d25c0ef": "WFH - CSR",
    "5e78bdd06e7e50297d25c0f0": "WFH - Specialty",
    "5e78c5db6e7e50297d25c0f4": "Direct Connect",
    "5ed7c9899aceda7645dd539f": "Rebate",
    "60784ba843d4100016d82bd8": "DAS",
    "61d6082e1f35000017f82f60": "Corp - QA",
    "61d608381f35000017f82f64": "Corp - Training",
    "61d6083f1f35000017f82f6b": "Corp - Support",
    "62966af049fada001e5688db": "Rebate - Supervisor",
    "62fff8085e3ee9001c8d9866": "Chat - Supervisor",
    "62fff8154c68dc0023038e6f": "CSR - Supervisor",
    "62fff8525e3ee9001c8d990c": "SPEC - Supervisor",
    "62fff8666a0a83001d10e84a": "Corporate - User",
    "62fff87c4c68dc0023038f20": "Corp - Admin",
    "631a50802d85b20023fcbf79": "Web Sales",
    "648b7e9861a948001e0496ac": "VR - CSR",
    "648b7ec143b5ee001f4d9759": "VR - Support",
    "64ff8e0e704ed5001a35cfb3": "Training - Supervisor",
    "64ff8e1a704ed5001a35cfcf": "QA - Supervisor",
    "6500beafdb56c1001baa07eb": "Websales",
    "6500c25aa5ff8e001a53912b": "VR - Supervisor",
    "655d1a069351230024b32c91": "Sr. Operations Manager",
    "66880c95113c14014963800c": "WNP - Supervisor",
    "66880cb7b498a70154fbb742": "DSG - Supervisor",
    "66881badca2e83016a7beb94": "Social Media - Supervisor"
}

def ensure_json_serializable(vector):
    """
    FIXED: Ensure vector is JSON serializable with better error handling
    """
    if vector is None:
        return get_empty_vector()
    
    try:
        # If it's already a list of regular Python numbers, return as-is
        if isinstance(vector, list) and all(isinstance(x, (int, float)) for x in vector):
            return vector
        
        # If it's a numpy array, convert to list
        if hasattr(vector, 'astype') and hasattr(vector, 'tolist'):
            return vector.astype(float).tolist()
        
        # If it's a list of numpy values, convert each element
        if isinstance(vector, list):
            return [float(x) if hasattr(x, 'item') else float(x) for x in vector]
        
        # If it's a single numpy value, convert to list
        if hasattr(vector, 'tolist'):
            return vector.tolist()
        
        # Fallback: try to convert to list of floats
        return [float(x) for x in vector]
        
    except (TypeError, ValueError, Exception) as e:
        add_log(f"[WARNING] Could not convert vector to JSON serializable format: {type(vector)}, error: {e}")
        return get_empty_vector()

def add_log(message: str):
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    import_status["logs"].append(entry)
    logger.info(entry)
    # Only keep last 1000 entries
    if len(import_status["logs"]) > 1000:
        import_status["logs"] = import_status["logs"][-1000:]

def log_memory():
    """Added error handling for resource module"""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Adjust for different platforms
        if sys.platform == 'darwin':  # macOS
            mem = usage / 1024 / 1024  # Convert to MB
        else:  # Linux
            mem = usage / 1024  # Already in KB, convert to MB
        add_log(f"Memory usage: {mem:.2f} MB")
    except ImportError:
        add_log("Memory logging not available on this platform")
    except Exception as e:
        add_log(f"Memory logging failed: {e}")

def normalize_program(program_value):
    """Normalize program values according to endpoint specification"""
    if not program_value or str(program_value).strip() == "":
        return "All"
    
    program_str = str(program_value).strip()
    
    # Direct mapping for known values
    if program_str.lower() in ["metro", "asw", "t-mobile pre-paid", "all"]:
        if program_str.lower() == "metro":
            return "Metro"
        elif program_str.lower() == "asw":
            return "ASW"
        elif program_str.lower() == "t-mobile pre-paid":
            return "T-Mobile Pre-Paid"
        else:
            return "All"
    
    # Default to "All" for any unclear values
    return "All"

def get_lob_name(lob):
    """Get LOB name from mapping dictionary"""
    if not lob:
        return "All"
    
    lob_str = str(lob).strip()
    return LOB_MAPPING.get(lob_str, f"LOB_{lob_str}")

def extract_clean_content(content):
    """Extract clean text content from HTML"""
    if not content:
        return ""
    
    if isinstance(content, dict):
        content = json.dumps(content)
    elif not isinstance(content, str):
        content = str(content)
    
    # Clean HTML and normalize whitespace
    clean_text = BeautifulSoup(content, "html.parser").get_text(separator="\n").strip()
    
    # Remove excessive whitespace
    clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
    clean_text = re.sub(r' +', ' ', clean_text)
    
    return clean_text

def ensure_json_serializable(vector):
    """
    FIXED: Ensure vector is JSON serializable by converting numpy arrays to Python lists
    """
    if vector is None:
        return []
    
    # If it's already a list of regular Python numbers, return as-is
    if isinstance(vector, list) and all(isinstance(x, (int, float)) for x in vector):
        return vector
    
    # If it's a numpy array, convert to list
    if isinstance(vector, np.ndarray):
        return vector.astype(float).tolist()
    
    # If it's a list of numpy values, convert each element
    if isinstance(vector, list):
        return [float(x) if hasattr(x, 'item') else float(x) for x in vector]
    
    # If it's a single numpy value, convert to list
    if hasattr(vector, 'tolist'):
        return vector.tolist()
    
    # Fallback: try to convert to list of floats
    try:
        return [float(x) for x in vector]
    except (TypeError, ValueError):
        add_log(f"[WARNING] Could not convert vector to JSON serializable format: {type(vector)}")
        return get_empty_vector()

# FIXED: Add a unified vector search function used by both endpoints
async def unified_vector_search(query: str, programs: List[str] = None, max_results: int = 25, hybrid: bool = True) -> List[dict]:
    """Unified vector search function using AI embeddings for semantic similarity"""
    try:
        # Generate embedding for the query using our embedding service
        query_embedding = get_embeddings([query])[0] if query else []
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # FIXED: Ensure query embedding is JSON serializable
        query_embedding = ensure_json_serializable(query_embedding)
        
        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions for: {query}")
        
        # Build the search query with vector similarity
        if hybrid:
            # Hybrid search: combine vector similarity with keyword matching
            query_body = {
                "query": {
                    "bool": {
                        "should": [
                            # Vector similarity search (primary)
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "1 + dotProduct(params.query_vector, 'vector')",
                                        "params": {"query_vector": query_embedding}
                                    },
                                    "boost": 2.0
                                }
                            },
                            # Keyword fallback (secondary)
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["name^1.5", "chunk.text"],
                                    "fuzziness": "AUTO",
                                    "boost": 0.5
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": max_results,
                "_source": [
                    "id", "name", "url", "program", "lob_name", 
                    "collection", "chunk", "updated", "vector"
                ],
                "sort": [
                    {"_score": {"order": "desc"}}
                ]
            }
        else:
            # Pure vector search
            query_body = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "1 + dotProduct(params.query_vector, 'vector')",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "size": max_results,
                "_source": [
                    "id", "name", "url", "program", "lob_name", 
                    "collection", "chunk", "updated"
                ],
                "sort": [
                    {"_score": {"order": "desc"}}
                ]
            }
        
        # Add program filter if specified
        if programs:
            program_filter = []
            for prog in programs:
                normalized_prog = normalize_program(prog)
                program_filter.append(normalized_prog)
            
            if "bool" not in query_body["query"]:
                # Wrap existing query in bool for pure vector search
                query_body["query"] = {
                    "bool": {
                        "must": [query_body["query"]],
                        "filter": {
                            "terms": {"program.keyword": program_filter}
                        }
                    }
                }
            else:
                # Add filter to existing bool query
                query_body["query"]["bool"]["filter"] = {
                    "terms": {"program.keyword": program_filter}
                }
        
        # Search OpenSearch using vector similarity
        response = opensearch_client.search(index="kb-*", body=query_body)
        hits = response.get("hits", {}).get("hits", [])
        
        # Log similarity scores for debugging
        if hits:
            top_score = hits[0].get("_score", 0)
            logger.info(f"Vector search found {len(hits)} results. Top similarity score: {top_score:.4f}")
        
        return hits
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        # FIXED: Fallback to keyword search if vector search fails
        try:
            logger.info("Falling back to keyword search")
            return await fallback_keyword_search(query, programs, max_results)
        except Exception as fallback_error:
            logger.error(f"Fallback search also failed: {fallback_error}")
            return []

# FIXED: Ensure fallback_keyword_search is async
async def fallback_keyword_search(query: str, programs: List[str] = None, max_results: int = 25) -> List[dict]:
    """Fallback keyword search when vector search fails"""
    query_body = {
        "query": {
            "bool": {
                "should": [
                    {"multi_match": {
                        "query": query,
                        "fields": ["name^3"],
                        "fuzziness": "AUTO",
                        "boost": 3
                    }},
                    {"multi_match": {
                        "query": query,
                        "fields": ["chunk.text"],
                        "fuzziness": "AUTO",
                        "boost": 1
                    }},
                    {"term": {"id.keyword": {"value": query, "boost": 5}}},
                    {"wildcard": {"id.keyword": {"value": f"*{query}*", "boost": 2}}}
                ],
                "minimum_should_match": 1
            }
        },
        "size": max_results,
        "_source": [
            "id", "name", "url", "program", "lob_name", 
            "collection", "chunk", "updated"
        ],
        "sort": [
            {"_score": {"order": "desc"}},
            {"updated": {"order": "desc"}}
        ]
    }
    
    if programs:
        program_filter = [normalize_program(prog) for prog in programs]
        query_body["query"]["bool"]["filter"] = {
            "terms": {"program.keyword": program_filter}
        }
    
    response = opensearch_client.search(index="kb-*", body=query_body)
    return response.get("hits", {}).get("hits", [])

# FIXED: Process document function using EXACT endpoint field names
async def process_document(doc: dict, collection: str, import_type: str, last_import_timestamp: Optional[str]) -> tuple:
    """
    Process a single document with FIXED incremental import logic and empty content deletion
    Returns (chunks_count, empty_document_count)
    """
    
    # Debug logging for incremental imports
    add_log(f"[DEBUG] import_type = '{import_type}'")
    add_log(f"[DEBUG] last_import_timestamp = '{last_import_timestamp}'")
    add_log(f"[DEBUG] Will call date function? {import_type == 'incremental' and last_import_timestamp}")
    
    # Extract fields using EXACT endpoint field names
    id = doc.get("id") or f"doc_{hash(str(doc))}"
    url = doc.get("url", "")
    program_raw = doc.get("program", "")
    name = doc.get("name", "[No Name]")
    lob = doc.get("lob", "")
    updated = doc.get("updated", "")
    content_raw = doc.get("content", "")

    # FIXED: Incremental import check - only if we have both incremental type AND valid timestamp
    if import_type == "incremental" and last_import_timestamp:
        if not is_document_updated_since_last_import(updated, last_import_timestamp):
            add_log(f"[INCREMENTAL] Document '{name}' not updated since last import - SKIPPING")
            return 0, 0  # 0 chunks, 0 empty documents (document was skipped, not empty)
        else:
            add_log(f"[INCREMENTAL] Document '{name}' has been updated - PROCESSING")
    else:
        # Full import mode - process all documents
        add_log(f"[FULL] Processing document '{name}' (full import mode)")
    
    add_log(f"[RAW DOC] Processing document from collection '{collection}'")
    add_log(f"[RAW DOC] Available fields: {list(doc.keys()) if isinstance(doc, dict) else 'Not a dict'}")
       
    add_log(f"[EXTRACTED] EXACT endpoint fields:")
    add_log(f"[EXTRACTED] - id: '{id}'")
    add_log(f"[EXTRACTED] - url: '{url}'")
    add_log(f"[EXTRACTED] - program: '{program_raw}'")
    add_log(f"[EXTRACTED] - name: '{name}'")
    add_log(f"[EXTRACTED] - lob: '{lob}'")
    add_log(f"[EXTRACTED] - updated: '{updated}'")
    add_log(f"[EXTRACTED] - content length: {len(str(content_raw))}")
    
    # Normalize values
    program = normalize_program(program_raw)
    lob_name = get_lob_name(lob)
    
    add_log(f"[NORMALIZED] program: '{program}' | lob_name: '{lob_name}'")
    
    # Extract and clean content
    clean_content = extract_clean_content(content_raw)
    add_log(f"[CONTENT] Clean content length: {len(clean_content)} characters")
    
    # FIXED: Check for empty content and DELETE from OpenSearch if document exists
    if len(clean_content.strip()) < 10:
        add_log(f"[EMPTY_CONTENT] Document '{name}' has insufficient content (length: {len(clean_content)})")
        
        try:
            # Import here to avoid circular imports
            from opensearch_client import delete_document_chunks
            
            # Delete existing chunks for this document from OpenSearch
            deleted_count = await delete_document_chunks(id, collection)
            
            if deleted_count > 0:
                add_log(f"[EMPTY_DELETE] ✅ Deleted {deleted_count} chunks for empty document '{name}' (ID: {id})")
                return 0, 1  # 0 new chunks, 1 empty document (deleted)
            else:
                add_log(f"[EMPTY_SKIP] Document '{name}' was already empty or not indexed - no deletion needed")
                return 0, 1  # 0 chunks, 1 empty document (was already empty)
                
        except Exception as e:
            add_log(f"[EMPTY_DELETE_ERROR] Failed to delete empty document '{name}': {e}")
            # Continue treating as empty even if deletion failed
            return 0, 1

    # Continue with chunking and embedding (your existing logic)
    try:
        chunks = split_into_chunks(clean_content)
        add_log(f"[CHUNKS] Split into {len(chunks)} chunks for '{name}'")
        
        if not chunks:
            add_log(f"[CHUNKS] No chunks generated for '{name}' - treating as empty content")
            
            # Also delete if no chunks could be generated
            try:
                from opensearch_client import delete_document_chunks
                deleted_count = await delete_document_chunks(id, collection)
                if deleted_count > 0:
                    add_log(f"[NO_CHUNKS_DELETE] ✅ Deleted {deleted_count} chunks for document '{name}' with no processable chunks")
            except Exception as e:
                add_log(f"[NO_CHUNKS_DELETE_ERROR] Failed to delete document '{name}' with no chunks: {e}")
            
            return 0, 1

        # Get embeddings for all chunks
        try:
            vectors = get_embeddings([chunk["text"] for chunk in chunks])
            add_log(f"[EMBEDDINGS] Generated {len(vectors)} embeddings for '{name}'")
        except Exception as e:
            add_log(f"[EMBEDDINGS ERROR] Failed to generate embeddings for '{name}': {e}")
            return 0, 1

        chunks_indexed = 0
        
        # Index each chunk
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            # FIXED: Ensure vector is JSON serializable
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            elif not isinstance(vector, list):
                vector = list(vector)
            
            chunk_payload = {
                "collection": collection,
                "id": id,  # FIXED: Changed from document_id to id
                "text": chunk["text"],  # FIXED: Moved text field up for required field validation
                "name": name,  # FIXED: Moved name field up for required field validation
                "offset": chunk.get("offset", i * 1000),  # FIXED: Use chunk offset instead of start_char
                "url": url,
                "program": program,
                "lob": lob_name,
                "updated": updated,
                "vector": vector,
                "indexed_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Debug log for first chunk
            if i == 0:
                add_log(f"[INDEX PAYLOAD] First chunk vector type: {type(vector)}")
                add_log(f"[INDEX PAYLOAD] First chunk vector sample: {vector[:3] if vector else 'empty'}")
                add_log(f"[INDEX PAYLOAD] First chunk with EXACT endpoint fields: {json.dumps({k: v for k, v in chunk_payload.items() if k != 'vector'}, indent=2)[:300]}...")
            
            try:
                # FIXED: Pass replace_existing parameter for incremental imports
                replace_existing = (import_type == "incremental")
                await index_chunks(chunk_payload, replace_existing=replace_existing)
                chunks_indexed += 1
                add_log(f"[SUCCESS] Indexed chunk {i+1}/{len(chunks)} for '{name}'")
            except Exception as e:
                add_log(f"[ERROR] Failed to index chunk {i+1} for '{name}': {e}")
                add_log(f"[ERROR] Chunk payload keys: {list(chunk_payload.keys())}")
                add_log(f"[ERROR] Vector type: {type(chunk_payload['vector'])}")
                continue
        
        add_log(f"[SUCCESS] '{name}' indexed with {chunks_indexed}/{len(chunks)} chunks using EXACT endpoint field names")
        return chunks_indexed, 0

    except Exception as e:
        add_log(f"[DOCUMENT ERROR] Failed to process document '{name}': {e}")
        return 0, 1
    
async def cleanup_deleted_documents_simple(collection: str, api_documents: list):
    """
    SIMPLE SOLUTION: Delete documents from OpenSearch that are no longer in the API response
    Works for both full and incremental imports without requiring timestamps
    """
    index = f"kb-{collection}"
    
    try:
        # Get all documents currently in OpenSearch
        search_query = {
            "size": 10000,  # Increase if you have more documents
            "_source": ["id", "name"],  # We only need id and name for comparison
            "query": {"match_all": {}}
        }
        
        response = opensearch_client.search(index=index, body=search_query)
        opensearch_documents = response.get("hits", {}).get("hits", [])
        
        # Create set of current API document IDs (convert to string for comparison)
        api_doc_ids = set(str(doc.get("id")) for doc in api_documents if doc.get("id"))
        add_log(f"[DELETE_SIMPLE] API has {len(api_doc_ids)} documents")
        
        # Find documents in OpenSearch that are NOT in the API response
        documents_to_delete = []
        for os_doc in opensearch_documents:
            source = os_doc["_source"]
            doc_id = str(source.get("id"))
            doc_name = source.get("name", "Unknown")
            
            if doc_id not in api_doc_ids:
                documents_to_delete.append({
                    "id": doc_id,
                    "name": doc_name
                })
                add_log(f"[DELETE_MARK] '{doc_name}' (ID: {doc_id}) - NOT FOUND in API, marking for deletion")
        
        if not documents_to_delete:
            add_log(f"[DELETE_SIMPLE] No orphaned documents found in {collection}")
            return 0
        
        add_log(f"[DELETE_SIMPLE] Found {len(documents_to_delete)} documents to delete from {collection}")
        
        # Delete the orphaned documents
        total_deleted = 0
        for doc_info in documents_to_delete:
            try:
                doc_id = doc_info["id"]
                doc_name = doc_info["name"]
                
                # Delete all chunks for this document
                delete_query = {
                    "query": {
                        "term": {
                            "id.keyword": doc_id
                        }
                    }
                }
                
                delete_response = opensearch_client.delete_by_query(
                    index=index,
                    body=delete_query,
                    refresh=True
                )
                
                deleted_count = delete_response.get("deleted", 0)
                total_deleted += deleted_count
                add_log(f"[DELETE_SUCCESS] Deleted {deleted_count} chunks for '{doc_name}' (ID: {doc_id})")
                
            except Exception as e:
                add_log(f"[DELETE_ERROR] Failed to delete document '{doc_name}': {e}")
                continue
        
        add_log(f"[DELETE_SIMPLE] Successfully deleted {total_deleted} chunks from {len(documents_to_delete)} orphaned documents in {collection}")
        return total_deleted
        
    except Exception as e:
        add_log(f"[DELETE_ERROR] Failed to cleanup deleted documents in {collection}: {e}")
        return 0
    
# FIXED: Fetch collection data with better error handling and field validation
def fetch_collection_data(api_base, collection, headers):
    """
    FIXED: Fetch data from endpoint with better field validation
    """
    url = f"{api_base}/api/content/{collection}"
    add_log(f"[API] Fetching from URL: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        
        raw_data = response.json()
        add_log(f"[API] Raw response type: {type(raw_data)}")
        add_log(f"[API] Raw response keys: {list(raw_data.keys()) if isinstance(raw_data, dict) else 'Not a dict'}")
        add_log(f"[API] Raw response sample: {str(raw_data)[:300]}...")
        
        # Normalize data structure - handle different response formats
        documents = []
        
        if isinstance(raw_data, dict):
            if "documents" in raw_data:
                if isinstance(raw_data["documents"], list):
                    documents = raw_data["documents"]
                elif isinstance(raw_data["documents"], dict):
                    documents = [raw_data["documents"]]
            elif any(key in raw_data for key in ["id", "name", "content"]):
                # Single document response
                documents = [raw_data]
            else:
                # Check if raw_data itself contains document-like data
                add_log(f"[API] Unexpected dict structure, treating as single document")
                documents = [raw_data]
        elif isinstance(raw_data, list):
            documents = raw_data
        else:
            add_log(f"[API] Unexpected response type: {type(raw_data)}")
            documents = [raw_data]
        
        add_log(f"[API] Extracted {len(documents)} documents from collection '{collection}'")
        
        # Validate document structure
        if documents:
            sample_doc = documents[0]
            add_log(f"[API] Sample document fields: {list(sample_doc.keys()) if isinstance(sample_doc, dict) else 'Not a dict'}")
            add_log(f"[API] Sample document: {str(sample_doc)[:200]}...")
            
            # Check for required endpoint fields
            required_fields = ["id", "url", "program", "name", "lob", "updated", "content"]
            missing_fields = []
            present_fields = []
            
            if isinstance(sample_doc, dict):
                for field in required_fields:
                    if field in sample_doc:
                        present_fields.append(field)
                    else:
                        missing_fields.append(field)
                
                add_log(f"[API] Present fields: {present_fields}")
                if missing_fields:
                    add_log(f"[API] WARNING - Missing expected fields: {missing_fields}")
        
        return documents
        
    except requests.exceptions.RequestException as e:
        add_log(f"[API ERROR] Request failed: {e}")
        raise
    except json.JSONDecodeError as e:
        add_log(f"[API ERROR] JSON decode failed: {e}")
        raise
    except Exception as e:
        add_log(f"[API ERROR] Unexpected error: {e}")
        raise

def fetch_dynamic_endpoints(api_base, headers):
    try:
        url = f"{api_base}/api/content/availableItems"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if "availableItems" in data:
            endpoints = []
            full_endpoint_info = {}
            for item in data["availableItems"]:
                if "apiKey" not in item or "name" not in item:
                    logger.warning(f"Skipping invalid item: {item}")
                    continue
                api_key = item["apiKey"]
                endpoints.append(api_key)
                full_endpoint_info[api_key] = {
                    "display_name": item["name"],
                    "path": f"/api/content/{api_key}",
                    "method": "GET",
                    "params": {},
                    "id_field": "id",
                    "content_fields": ["id", "url", "program", "name", "content", "lobs"]
                }
            import_status["endpoint_metadata"] = full_endpoint_info
            return endpoints
    except Exception as e:
        logger.warning(f"Failed to fetch dynamic endpoints: {e}")
    return ["plans", "phones", "features", "add-ons", "faq", "stores", "policies"]



# UPDATED: Modified import process to use the simple cleanup
async def run_import_process(max_docs: Optional[int], collection: str, import_type: str = "full"):
    """Main import process with FIXED deletion cleanup for both full and incremental imports"""
    total_chunks = 0
    total_documents = 0
    empty_documents = 0
    batch_size = 10

    # Get last import timestamp for incremental imports
    last_import_timestamp = None
    if import_type == "incremental":
        last_import_timestamp = get_last_import_timestamp()
        add_log(f"[INCREMENTAL] Retrieved last import timestamp: {last_import_timestamp}")
        
        if not last_import_timestamp:
            add_log(f"[INCREMENTAL] No previous import timestamp found - converting to FULL import")
            import_type = "full"
        else:
            add_log(f"[INCREMENTAL] Starting incremental import since: {last_import_timestamp}")
    else:
        add_log(f"[FULL] Starting full import")

    api_base = os.getenv("API_BASE_URL")
    if not api_base:
        raise RuntimeError("API_BASE_URL not set")
    headers = {os.getenv("API_AUTH_KEY", "Authorization"): os.getenv("API_AUTH_VALUE", "token")}

    try:
        targets = [collection] if collection != "all" else fetch_dynamic_endpoints(api_base, headers)
        import_status["empty_documents_by_collection"] = {}
        import_status["import_type"] = import_type
        
        add_log(f" Starting {import_type} import for collections: {targets}")
        add_log(f" Using batch size: {batch_size} documents per batch")
        if max_docs:
            add_log(f" Limiting to {max_docs} documents per collection")
        log_memory()

        for coll in targets:
            add_log(f"📡 Fetching from collection: {coll}")
            import_status["current_step"] = f"Fetching {coll}"
            
            try:
                documents = fetch_collection_data(api_base, coll, headers)
            except Exception as e:
                add_log(f"[ERROR] Failed to fetch collection '{coll}': {e}")
                continue

            add_log(f" Retrieved {len(documents)} documents from collection '{coll}'")
            
            # SIMPLE DELETION CLEANUP - Works for both full and incremental imports
            try:
                import_status["current_step"] = f"Cleaning up deleted documents in {coll}"
                deleted_chunks = await cleanup_deleted_documents_simple(coll, documents)
                add_log(f"[DELETE_CLEANUP] Removed {deleted_chunks} chunks from orphaned documents in {coll}")
            except Exception as e:
                add_log(f"[DELETE_ERROR] Deletion cleanup failed for {coll}: {e}")
            
            collection_docs = 0
            collection_chunks = 0
            collection_empty = 0

            docs_to_process = documents[:max_docs] if max_docs else documents
            add_log(f"⚙️ Processing {len(docs_to_process)} documents from '{coll}' (limit: {max_docs or 'none'})")

            # Process documents in batches (your existing batch logic continues...)
            for batch_start in range(0, len(docs_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(docs_to_process))
                batch_docs = docs_to_process[batch_start:batch_end]
                import_status["current_step"] = f"Processing {coll} - batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end} of {len(docs_to_process)})"
                
                for i, doc in enumerate(batch_docs):
                    try:
                        import_status["current_step"] = f"Processing {coll} - document {batch_start + i + 1}/{len(docs_to_process)}"
                        
                        d_chunks, d_empty = await process_document(doc, coll, import_type=import_type, last_import_timestamp=last_import_timestamp)
                        collection_chunks += d_chunks
                        collection_docs += 1
                        collection_empty += d_empty

                        if ((batch_start + i + 1) % 10) == 0:
                            add_log(f"📈 Progress: {batch_start + i + 1}/{len(docs_to_process)} documents processed")

                    except Exception as e:
                        add_log(f"[ERROR] Failed to process document {batch_start + i + 1}: {e}")
                        collection_docs += 1
                        empty_documents += 1

            # Collection summary
            import_status["empty_documents_by_collection"][coll] = collection_empty
            total_documents += collection_docs
            total_chunks += collection_chunks
            empty_documents += collection_empty
            
            add_log(f"✅ Collection '{coll}' completed: {collection_docs} docs, {collection_chunks} chunks, {collection_empty} empty")

        # Final summary
        import_status["status"] = "completed"
        import_status["documents_processed"] = total_documents
        import_status["chunks_created"] = total_chunks
        import_status["empty_documents"] = empty_documents
        import_status["current_step"] = "Import completed"
        
        # Update last import timestamp
        last_import_info["timestamp"] = datetime.now(timezone.utc).isoformat()
        last_import_info["type"] = import_type
        
        add_log(f"🎉 Import completed: {total_documents} documents, {total_chunks} chunks, {empty_documents} empty")
        log_memory()

    except Exception as e:
        import_status["status"] = "error"
        import_status["error"] = str(e)
        import_status["current_step"] = f"Error: {str(e)}"
        add_log(f"❌ Import failed: {e}")
        raise

@app.get("/debug/date_from_api/{collection}")
async def debug_date_from_api(collection: str):
    """Debug endpoint to check exact date format from API"""
    try:
        api_base = os.getenv("API_BASE_URL")
        if not api_base:
            return {"status": "error", "error": "API_BASE_URL not configured"}
        
        headers = {os.getenv("API_AUTH_KEY", "Authorization"): os.getenv("API_AUTH_VALUE", "token")}
        
        # Fetch documents from API
        documents = fetch_collection_data(api_base, collection, headers)
        
        # Find documents with the problematic date
        problematic_docs = []
        
        for doc in documents[:10]:  # Check first 10 documents
            doc_updated = doc.get("updated", "")
            doc_name = doc.get("name", "Unknown")
            
            # Check if this looks like the problematic date
            if "2025-01-01" in str(doc_updated):
                problematic_docs.append({
                    "name": doc_name,
                    "updated_raw": doc_updated,
                    "updated_type": type(doc_updated).__name__,
                    "updated_repr": repr(doc_updated),
                    "updated_length": len(str(doc_updated)),
                    "updated_str": str(doc_updated),
                    "test_exact_match": str(doc_updated).strip() == "2025-01-01T00:00:00.000Z",
                    "char_by_char": [ord(c) for c in str(doc_updated)][:30]  # First 30 chars as ASCII codes
                })
        
        return {
            "status": "success",
            "collection": collection,
            "total_documents": len(documents),
            "problematic_documents": problematic_docs,
            "expected_date": "2025-01-01T00:00:00.000Z",
            "expected_char_codes": [ord(c) for c in "2025-01-01T00:00:00.000Z"]
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Simplified test endpoint to verify the comparison logic
@app.get("/debug/test_date_match")
async def test_date_match():
    """Simple test of date matching logic"""
    
    test_date = "2025-01-01T00:00:00.000Z"
    test_variations = [
        "2025-01-01T00:00:00.000Z",    # Exact match
        " 2025-01-01T00:00:00.000Z",   # Leading space
        "2025-01-01T00:00:00.000Z ",   # Trailing space
        "2025-01-01T00:00:00.000z",    # Lowercase z
        "2025-01-01T00:00:00Z",        # No milliseconds
        "2025-01-01 00:00:00",         # Space instead of T
    ]
    
    results = []
    for variation in test_variations:
        # Test our comparison logic
        matches_exact = variation == test_date
        matches_stripped = variation.strip() == test_date
        
        results.append({
            "test_string": variation,
            "repr": repr(variation),
            "length": len(variation),
            "matches_exact": matches_exact,
            "matches_stripped": matches_stripped,
            "char_codes": [ord(c) for c in variation]
        })
    
    return {
        "status": "success",
        "expected_date": test_date,
        "expected_repr": repr(test_date),
        "expected_length": len(test_date),
        "expected_char_codes": [ord(c) for c in test_date],
        "test_results": results
    }

@app.get("/search", response_class=JSONResponse)
async def search_opensearch(q: str = Query(..., description="Search query")):
    """Enhanced search using AI embeddings for semantic similarity"""
    try:
        # Use vector similarity search
        hits = await unified_vector_search(q, programs=None, max_results=25, hybrid=True)
        results = []
        
        for h in hits:
            doc = h["_source"]
            collection_name = h["_index"].replace("kb-", "")
            
            # Build comprehensive result with similarity score
            result = {
                "title": doc.get("name", "Unknown Document"),
                "id": doc.get("id"),
                "url": doc.get("url", ""),  # From 'url' endpoint field
                "collection": collection_name,
                "chunk": doc.get("chunk", {}),
                "program": doc.get("program"),
                "lob_name": doc.get("lob_name"),
                "score": h.get("_score", 0),
                "similarity_score": h.get("_score", 0),
                "updated": doc.get("updated"),
                "has_clickable_url": bool(doc.get("url") and doc.get("url").strip())
            }
            
            results.append(result)
        
        # Group by document to avoid showing multiple chunks from same doc
        grouped_results = {}
        for result in results:
            doc_id = result["id"]
            if doc_id not in grouped_results or result["score"] > grouped_results[doc_id]["score"]:
                grouped_results[doc_id] = result
        
        final_results = list(grouped_results.values())
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Add similarity score info to response
        similarity_info = {
            "search_type": "vector_similarity",
            "embedding_model": get_embedding_service().model_name if hasattr(get_embedding_service(), 'model_name') else "unknown",
            "top_score": final_results[0]["score"] if final_results else 0,
            "threshold_used": "hybrid_search"
        }
        
        return {
            "status": "success", 
            "results": final_results,
            "total_hits": len(final_results),
            "total_chunks": len(results),
            "search_info": similarity_info
        }
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return {"status": "error", "error": str(e)}
    
@app.get("/search_enhanced", response_class=JSONResponse)
async def search_enhanced(q: str = Query(..., description="Enhanced search query")):
    """Enhanced search with additional results for 'show more' functionality"""
    try:
        # Use vector similarity search with more results
        hits = await unified_vector_search(q, programs=None, max_results=50, hybrid=True)
        results = []
        
        for h in hits:
            doc = h["_source"]
            collection_name = h["_index"].replace("kb-", "")
            
            # Build comprehensive result with similarity score
            result = {
                "title": doc.get("name", "Unknown Document"),
                "id": doc.get("id"),
                "url": doc.get("url", ""),  # From 'url' endpoint field
                "collection": collection_name,
                "chunk": doc.get("chunk", {}),
                "program": doc.get("program"),
                "lob_name": doc.get("lob_name"),
                "score": h.get("_score", 0),
                "similarity_score": h.get("_score", 0),
                "updated": doc.get("updated"),
                "has_clickable_url": bool(doc.get("url") and doc.get("url").strip())
            }
            
            results.append(result)
        
        # Group by document to avoid showing multiple chunks from same doc
        grouped_results = {}
        for result in results:
            doc_id = result["id"]
            if doc_id not in grouped_results or result["score"] > grouped_results[doc_id]["score"]:
                grouped_results[doc_id] = result
        
        final_results = list(grouped_results.values())
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "status": "success", 
            "results": final_results,
            "total_hits": len(final_results),
            "total_chunks": len(results),
            "search_info": {
                "search_type": "enhanced_vector_similarity",
                "embedding_model": get_embedding_service().model_name if hasattr(get_embedding_service(), 'model_name') else "unknown",
                "max_results": 50
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/count_by_program", response_class=JSONResponse)
async def count_documents_by_program():
    try:
        query = {
            "size": 0,
            "aggs": {
                "programs": {
                    "terms": {"field": "program.keyword", "size": 50, "missing": "All"}
                }
            }
        }        
        response = opensearch_client.search(index="kb-*", body=query)
        buckets = response.get("aggregations", {}).get("programs", {}).get("buckets", [])

        return {
            "status": "success", 
            "program_counts": {b["key"]: b["doc_count"] for b in buckets}
        }
    except Exception as e:
        logger.error(f"Count by program error: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/count_by_collection_and_program", response_class=JSONResponse)
async def count_by_collection_and_program():
    try:
        query = {
            "size": 0,
            "aggs": {
                "collections": {
                    "terms": {"field": "_index", "size": 50},
                    "aggs": {
                        "programs": {
                            "terms": {"field": "program.keyword", "size": 20, "missing": "All"}
                        }
                    }
                }
            }
        }
        response = opensearch_client.search(index="kb-*", body=query)
        counts = {}

        for bucket in response["aggregations"]["collections"]["buckets"]:
            collection = bucket["key"].replace("kb-", "")
            counts[collection] = {}
            for sub in bucket["programs"]["buckets"]:
                program_name = sub["key"] if sub["key"] else "All"
                counts[collection][program_name] = sub["doc_count"]

        return {"status": "success", "collection_program_counts": counts}
    except Exception as e:
        logger.error(f"Count by collection and program error: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        return FileResponse("static/chat.html")
        
    except FileNotFoundError:
        return HTMLResponse("<h1>Chat Interface Not Found</h1><p>Please ensure chat.html exists in root directory</p>")

@app.get("/admin", response_class=HTMLResponse)
async def admin_interface():
    """Serve the original admin/import interface"""
    try:
        return FileResponse("static/index.html")
    except FileNotFoundError:
        return HTMLResponse("""
            <h1>Admin Interface Not Found</h1>
            <p>Please ensure index.html exists in the static folder</p>
        """)

@app.get("/status", response_class=JSONResponse)
async def get_status():
    return import_status

@app.get("/logs", response_class=JSONResponse)
async def get_logs():
    return {"logs": import_status.get("logs", [])}

@app.get("/health", response_class=JSONResponse)
async def check_health():
    """Health check with actual component testing"""
    health_status = {"status": "ok", "components": {}}
    
    # Check embeddings
    try:
        service = get_embedding_service()
        test_embedding = service.embed_single("health check")
        health_status["components"]["embeddings"] = {
            "status": "ok", 
            "model": service.model_name,
            "dimension": len(test_embedding)
        }
    except Exception as e:
        health_status["components"]["embeddings"] = {"status": "error", "error": str(e)}
    
    # Check OpenSearch
    try:
        info = opensearch_client.info()
        health_status["components"]["opensearch"] = {
            "status": "ok",
            "version": info.get("version", {}).get("number", "unknown")
        }
    except Exception as e:
        health_status["components"]["opensearch"] = {"status": "error", "error": str(e)}
    
    # Check GENAI endpoint configuration
    genai_endpoint = os.getenv("GENAI_ENDPOINT")
    genai_access_key = os.getenv("GENAI_ACCESS_KEY")
    
    if genai_endpoint and genai_access_key:
        health_status["components"]["genai"] = {
            "status": "configured",
            "endpoint": genai_endpoint.replace(genai_access_key, "***") if genai_access_key in genai_endpoint else genai_endpoint,
            "note": "Ready for AI-powered chat responses"
        }
    else:
        health_status["components"]["genai"] = {
            "status": "not_configured",
            "note": "GENAI_ENDPOINT and GENAI_ACCESS_KEY not set - chat will fall back to search results"
        }
    
    # Check API configuration for import
    api_base = os.getenv("API_BASE_URL")
    api_auth_key = os.getenv("API_AUTH_KEY")
    api_auth_value = os.getenv("API_AUTH_VALUE")
    
    if api_base and api_auth_key and api_auth_value:
        health_status["components"]["api_source"] = {
            "status": "configured",
            "base_url": api_base,
            "auth_header": api_auth_key,
            "note": "Ready for document import"
        }
    else:
        missing_vars = []
        if not api_base: missing_vars.append("API_BASE_URL")
        if not api_auth_key: missing_vars.append("API_AUTH_KEY") 
        if not api_auth_value: missing_vars.append("API_AUTH_VALUE")
        
        health_status["components"]["api_source"] = {
            "status": "not_configured",
            "missing_variables": missing_vars,
            "note": "Required for document import from external API"
        }
    
    return health_status

class ImportRequest(BaseModel):
    max_docs: Optional[int] = None
    collection: Optional[str] = "all"
    import_type: Optional[str] = "full"  # Add support for incremental


# Chat endpoint
from openai import OpenAI

class ChatRequest(BaseModel):
    message: str
    programs: Optional[List[str]] = []
    history: Optional[List[Dict[str, str]]] = []

@app.post("/chat")
async def chat_with_agent(req: ChatRequest):
    """Chat endpoint using vector similarity search with proper OpenSearch integration"""
    
    logger.info(f"Chat request received: '{req.message}' with programs: {req.programs}")
    
    # Step 1: Search using vector similarity with AI embeddings from YOUR OpenSearch
    try:
        local_search_results = await unified_vector_search(
            req.message, 
            programs=None,
            max_results=25, 
            hybrid=True
        )
        logger.info(f"OpenSearch returned {len(local_search_results)} results for: {req.message}")
        
        # 🔍 DEBUG: Show raw search results
        logger.info("=== RAW SEARCH RESULTS ===")
        for i, hit in enumerate(local_search_results[:5]):
            source = hit["_source"]
            text = source.get("chunk", {}).get("text", "")
            logger.info(f"{i+1}. {source.get('name')} - Text: '{text[:100]}...' - Length: {len(text)}")
        logger.info("=== END RAW SEARCH RESULTS ===")
        
        # Debug: Log what we found
        if local_search_results:
            for i, hit in enumerate(local_search_results[:3]):
                source = hit["_source"]
                logger.info(f"Result {i+1}: {source.get('name', 'Unknown')} - Score: {hit.get('_score', 0)}")
        else:
            logger.warning(f"No results found in OpenSearch for query: {req.message}")
            
    except Exception as e:
        logger.error(f"OpenSearch vector search failed: {e}")
        local_search_results = []
        
    # Step 2: Build context from local search results with better filtering
    context_documents = []
    references = []
    seen_document_ids = set()  # Track which documents we've already added
        
    # Use a lower similarity threshold for security-related queries
    min_score_threshold = 0.1  # Adjust this based on your data
    
    relevant_results = []
    for hit in local_search_results:
        score = hit.get("_score", 0)
        if score >= min_score_threshold:
            relevant_results.append(hit)
        else:
            #logger.info(f"Filtered out result with low score: {score}")
            logger.info(f"Filtered out result with low score: {score} (threshold: {min_score_threshold})")

    if not relevant_results and local_search_results:
        logger.info("No results after threshold filtering, using all results")
        relevant_results = local_search_results
    
    for hit in relevant_results[:10]:  # Use top relevant results
        source = hit["_source"]
        
        # Extract document info for context
        doc_name = source.get("name", "Document")
        doc_url = source.get("url", "")
        doc_text = source.get("chunk", {}).get("text", "")
        program = source.get("program", "All")
        lob_name = source.get("lob_name", "All")
        doc_id = source.get("id", "unknown")
        collection_name = hit["_index"].replace("kb-", "") if "_index" in hit else "unknown"  # extract collection from index name
        
        if doc_text and len(doc_text.strip()) > 20 and doc_text != "[empty document]":  # Skip empty docs
            context_documents.append({
                "name": doc_name,
                "content": doc_text,
                "program": program,
                "lob": lob_name,
                "url": doc_url,
                "id": doc_id,
                "collection": collection_name,  # ADDED for use in results
                "score": hit.get("_score", 0)
            })
            
            # Only add reference if we haven't seen this document ID before
            if doc_id not in seen_document_ids:
                references.append({
                    "title": doc_name,
                    "program": program,
                    "lob": lob_name,
                    "url": doc_url if doc_url else None,
                    "id": doc_id,
                    "collection": collection_name,  # ADDED for use in results
                    "score": hit.get("_score", 0)  # ADDED Include score for consistency
                })
                seen_document_ids.add(doc_id)
    
    logger.info(f"Built context from {len(context_documents)} documents")
    logger.info(f"Built {len(references)} unique references")

    # DEBUG: Show context documents used
    logger.info("=== CONTEXT DOCUMENTS USED ===")
    for i, doc in enumerate(context_documents):
        logger.info(f"Context {i+1}: {doc['name']} - Text: '{doc['content'][:100]}...'")
    logger.info("=== END CONTEXT DOCUMENTS ===")
    
    # If no relevant context found, try a broader search
    if not context_documents:
        logger.info("No context found, trying broader search without program filters...")
        try:
            broader_results = await unified_vector_search(
                req.message, 
                programs=None,  # Remove program filter
                max_results=15, 
                hybrid=True
            )
            
            for hit in broader_results[:5]:
                source = hit["_source"]
                doc_text = source.get("chunk", {}).get("text", "")
                if doc_text and len(doc_text.strip()) > 20 and doc_text != "[empty document]":
                    context_documents.append({
                        "name": source.get("name", "Document"),
                        "content": doc_text,
                        "program": source.get("program", "All"),
                        "lob": source.get("lob_name", "All"),
                        "url": source.get("url", ""),
                        "id": source.get("id", "unknown"),
                        "score": hit.get("_score", 0)
                    })
                    
            logger.info(f"Broader search added {len(context_documents)} documents")
            
        except Exception as e:
            logger.error(f"Broader search also failed: {e}")
    
    # Step 3: Use external GenAI service for response generation with local context
    agent_endpoint = os.getenv("GENAI_ENDPOINT")
    agent_access_key = os.getenv("GENAI_ACCESS_KEY")
    
    if not agent_endpoint or not agent_access_key:
        logger.warning("GenAI service not configured, returning search results only")
        # Fallback: return search results if no GenAI service
        if context_documents:
            return {
                "summary": f"I found {len(context_documents)} relevant documents in the knowledge base:",
                "steps": [
                    f"**{doc['name']}** ({doc['program']} - {doc['lob']}): {doc['content'][:200]}..." 
                    for doc in context_documents[:3]
                ],
                "suggestions": [
                    f"• {doc['name']} ({doc['program']} - {doc['lob']})" + 
                    (f": {doc['url']}" if doc.get('url') else "") 
                    for doc in context_documents
                    if doc['id'] not in [ref['id'] for ref in references[:len(context_documents)]]  # Deduplicate
                ][:5],
                
                "references": references,
                "debug_info": {
                    "search_results_count": len(local_search_results),
                    "context_documents_used": len(context_documents),
                    "programs_searched": req.programs or ["All"],
                    "genai_available": False
                }
            }
        else:
            return {
                "summary": "I couldn't find relevant information in the knowledge base for your question.",
                "steps": [
                    "Try rephrasing your question",
                    "Check if the information exists in the knowledge base",
                    "Contact support for assistance"
                ],
                "suggestions": [],
                "references": [],
                "debug_info": {
                    "search_results_count": len(local_search_results),
                    "context_documents_used": 0,
                    "programs_searched": req.programs or ["All"],
                    "genai_available": False
                }
            }

    # Initialize GenAI client
    try:
        client = OpenAI(
            base_url=agent_endpoint.rstrip("/") + "/api/v1/",
            api_key=agent_access_key,
        )
        logger.info("GenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GenAI client: {e}")
        # Return search results as fallback
        if context_documents:
            return {
                "summary": f"Found information from {len(context_documents)} documents (GenAI unavailable):",
                "steps": [],
                "suggestions": [
                    f"• {doc['name']} ({doc['program']} - {doc['lob']})" + 
                    (f": {doc['url']}" if doc.get('url') else "") 
                    for doc in context_documents
                ],
                "references": references,
                "debug_info": {
                    "search_results_count": len(local_search_results),
                    "context_documents_used": len(context_documents),
                    "programs_searched": req.programs or ["All"],
                    "genai_available": False,
                    "genai_error": str(e)
                }
            }

    # Build enhanced system prompt with local context
    context_text = ""
    if context_documents:
        context_text = "\n\n=== KNOWLEDGE BASE CONTEXT ===\n"
        for doc in context_documents:
            doc_title = doc['name']
            context_text += f"\nDocument: {doc_title}\n"
            context_text += f"Program: {doc['program']} | LOB: {doc['lob']}\n"
            if doc.get('url'):
                context_text += f"URL: {doc['url']}\n"
            context_text += f"Content: {doc['content']}\n"
            context_text += "-" * 50 + "\n"
    
    SYSTEM_PROMPT = f"""
Identity: You are a helpful Metro by T-Mobile customer service assistant. You must ONLY use the provided context from the knowledge base below.

{context_text}

Instructions:
- Answer questions based ONLY on the context provided above
- If the context doesn't contain relevant information, say "I don't have information about that in the knowledge base"
- When referencing information, mention which document it came from
- Be helpful, accurate, and thorough in your responses
- Focus on customer service and support information
- Format your response as JSON with the following structure:

{{
  "summary": "Provide a detailed response explaining the question, this is your main answer based on the provided context",
  "steps": ["Step 1 if applicable", "Step 2 if applicable"],
  "suggestions": ["Additional suggestions or related information from the knowledge base if applicable"],
}}

Rules:
- Return only valid JSON
- Never invent information not in the provided context
- If no relevant context is provided, acknowledge this limitation clearly
- Never invent information not in the provided context
- Use actual document titles when referencing sources
"""

    # Compose message history
    messages = req.history.copy() if req.history else []
    if not (messages and messages[-1].get("role") == "user" and messages[-1].get("content") == req.message):
        messages.append({"role": "user", "content": req.message})
    messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    try:
        # Call external GenAI service
        logger.info(f"Sending request to GenAI with {len(context_documents)} context documents")
        response = client.chat.completions.create(
            model="llama-3.3-70b-instruct",
            messages=messages,
            temperature=0.7,  # Lower temperature for more consistent responses
            top_p=0.9,
            max_tokens=1500,
        )
        
        answer = response.choices[0].message.content
        logger.info(f"GenAI response received for query: {req.message}")

        # Parse JSON response
        try:
            answer_json = json.loads(answer)
        except Exception:
            # Try to extract JSON from response
            match = re.search(r'(\{.*\})', answer, re.DOTALL)
            if match:
                try:
                    answer_json = json.loads(match.group(1))
                except:
                    answer_json = None
            else:
                answer_json = None

        if answer_json and "summary" in answer_json:
            answer_json.setdefault("steps", [])
            answer_json.setdefault("suggestions", [])
            answer_json["references"] = references
            
            # Add debug info
            answer_json["debug_info"] = {
                "search_results_count": len(local_search_results),
                "context_documents_used": len(context_documents),
                "programs_searched": req.programs or ["All"],
                "genai_available": True
            }
            
            return answer_json

        # Fallback response format if JSON parsing fails
        return {
            "summary": answer if isinstance(answer, str) else "Response received but couldn't parse properly",
            "steps": [],
            "suggestions": [
                f"• {doc['name']} ({doc['program']} - {doc['lob']})" + 
                (f": {doc['url']}" if doc.get('url') else "") 
                for doc in context_documents
            ],
            "references": references,
            "debug_info": {
                "search_results_count": len(local_search_results),
                "context_documents_used": len(context_documents),
                "programs_searched": req.programs or ["All"],
                "genai_available": True,
                "json_parse_failed": True
            }
        }

    except Exception as e:
        logger.error(f"Chat GenAI error: {e}")
        # Return search results as final fallback
        if context_documents:
            return {
                "summary": f"I found {len(context_documents)} relevant documents about your question, but couldn't generate a response due to a technical issue. Here are the sources:",
                "steps": [
                    "Review the suggested documents below",
                    "Contact support if you need immediate assistance"
                ],
                "suggestions": [
                    f"• {doc['name']} ({doc['program']} - {doc['lob']})" + 
                    (f": {doc['url']}" if doc.get('url') else "") 
                    for doc in context_documents
                ],
                "references": references,
                "debug_info": {
                    "search_results_count": len(local_search_results),
                    "context_documents_used": len(context_documents),
                    "programs_searched": req.programs or ["All"],
                    "genai_available": True,
                    "genai_error": str(e)
                }
            }
        else:
            return {
                "summary": "I couldn't find relevant information in the knowledge base and encountered a technical issue.",
                "steps": [
                    "Try rephrasing your question",
                    "Check the search function to see if relevant documents exist",
                    "Contact support for assistance"
                ],
                "suggestions": [],
                "references": [],
                "debug_info": {
                    "search_results_count": len(local_search_results),
                    "context_documents_used": 0,
                    "programs_searched": req.programs or ["All"],
                    "genai_available": True,
                    "genai_error": str(e)
                }
            }
        
        

@app.post("/import", response_class=JSONResponse)
async def start_import(request: ImportRequest, background_tasks: BackgroundTasks):
    """
    FIXED: Start import and return immediately to prevent timeouts
    
    The key fix: This endpoint now returns immediately after starting the background task.
    No more waiting for the entire import to complete before sending a response.
    """
    try:
        if import_status["status"] == "running":
            raise HTTPException(status_code=409, detail="Import already running")

        # FIXED: Reset import status and return immediately
        import_status.update({
            "status": "running",
            "current_step": "initializing",
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
            "results": {},
            "logs": [],
            "error": None,
            "import_type": request.import_type or "full"
        })

        # FIXED: Start background task and return immediately (no waiting!)
        background_tasks.add_task(
            run_import_process, 
            request.max_docs, 
            request.collection or "all",
            request.import_type or "full"
        )
        
        # FIXED: Return immediately - don't wait for processing to complete
        return {
            "message": f"✅ {(request.import_type or 'full').title()} import started with chunked processing",
            "status": "running",
            "import_type": request.import_type or "full",
            "max_docs": request.max_docs,
            "collection": request.collection or "all",
            "note": "Import is running in background with chunked processing to prevent timeouts"
        }
        
    except Exception as e:
        import_status.update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.utcnow().isoformat()
        })
        logger.error(f"Import start failed: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

@app.post("/rate")
async def rate_response(request: dict):
    """Rate a chat response"""
    try:
        article_id = request.get("article_id", "unknown")
        value = request.get("value", "unknown")
        
        # Log the rating (you can store this in a database)
        add_log(f"Rating received: {value} for article {article_id}")
        
        return {"status": "success", "message": "Rating recorded"}
    except Exception as e:
        logger.error(f"Rating error: {e}")
        return {"status": "error", "error": str(e)}
    
# DEBUG ENDPOINTS

@app.get("/debug/search/{query}")
async def debug_search(query: str):
    """Debug endpoint to test search functionality"""
    try:
        logger.info(f"Debug search for: {query}")
        
        # Test different search approaches
        results = {
            "query": query,
            "embedding_info": {},
            "search_results": {},
            "opensearch_info": {}
        }
        
        # 1. Test embedding generation
        try:
            query_embedding = get_embeddings([query])[0] if query else []
            # FIXED: Ensure embedding is serializable for debug output
            query_embedding = ensure_json_serializable(query_embedding)
            results["embedding_info"] = {
                "status": "success",
                "embedding_length": len(query_embedding),
                "first_few_values": query_embedding[:5] if query_embedding else [],
                "embedding_model": get_embedding_service().model_name if hasattr(get_embedding_service(), 'model_name') else "unknown"
            }
        except Exception as e:
            results["embedding_info"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # 2. Test OpenSearch connection
        try:
            opensearch_info = opensearch_client.info()
            results["opensearch_info"] = {
                "status": "connected",
                "version": opensearch_info.get("version", {}).get("number", "unknown"),
                "cluster_name": opensearch_info.get("cluster_name", "unknown")
            }
        except Exception as e:
            results["opensearch_info"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # 3. Test different search types
        
        # Vector search
        try:
            vector_results = await unified_vector_search(query, programs=None, max_results=5, hybrid=False)
            results["search_results"]["vector_search"] = {
                "status": "success",
                "result_count": len(vector_results),
                "top_results": []
            }
            
            for hit in vector_results[:3]:
                source = hit["_source"]
                results["search_results"]["vector_search"]["top_results"].append({
                    "name": source.get("name", "Unknown"),
                    "score": hit.get("_score", 0),
                    "program": source.get("program", "Unknown"),
                    "collection": hit["_index"].replace("kb-", ""),
                    "chunk_preview": source.get("chunk", {}).get("text", "")[:100] + "..." if source.get("chunk", {}).get("text") else "No text"
                })
                
        except Exception as e:
            results["search_results"]["vector_search"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Hybrid search
        try:
            hybrid_results = await unified_vector_search(query, programs=None, max_results=5, hybrid=True)
            results["search_results"]["hybrid_search"] = {
                "status": "success",
                "result_count": len(hybrid_results),
                "top_results": []
            }
            
            for hit in hybrid_results[:3]:
                source = hit["_source"]
                results["search_results"]["hybrid_search"]["top_results"].append({
                    "name": source.get("name", "Unknown"),
                    "score": hit.get("_score", 0),
                    "program": source.get("program", "Unknown"),
                    "collection": hit["_index"].replace("kb-", ""),
                    "chunk_preview": source.get("chunk", {}).get("text", "")[:100] + "..." if source.get("chunk", {}).get("text") else "No text"
                })
                
        except Exception as e:
            results["search_results"]["hybrid_search"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Keyword search
        try:
            keyword_results = await fallback_keyword_search(query, programs=None, max_results=5)
            results["search_results"]["keyword_search"] = {
                "status": "success",
                "result_count": len(keyword_results),
                "top_results": []
            }
            
            for hit in keyword_results[:3]:
                source = hit["_source"]
                results["search_results"]["keyword_search"]["top_results"].append({
                    "name": source.get("name", "Unknown"),
                    "score": hit.get("_score", 0),
                    "program": source.get("program", "Unknown"),
                    "collection": hit["_index"].replace("kb-", ""),
                    "chunk_preview": source.get("chunk", {}).get("text", "")[:100] + "..." if source.get("chunk", {}).get("text") else "No text"
                })
                
        except Exception as e:
            results["search_results"]["keyword_search"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # 4. Test index statistics
        try:
            index_stats = opensearch_client.indices.stats(index="kb-*")
            results["index_stats"] = {
                "total_docs": index_stats.get("_all", {}).get("total", {}).get("docs", {}).get("count", 0),
                "indices": list(index_stats.get("indices", {}).keys())
            }
        except Exception as e:
            results["index_stats"] = {
                "error": str(e)
            }
        
        return {
            "status": "success",
            "debug_results": results
        }
        
    except Exception as e:
        logger.error(f"Debug search failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.on_event("startup")
async def startup_event():
    """Preload models during app startup to avoid slow first search"""
    logger.info("🚀 Starting Metro AI application with FIXED vector serialization...")
    
    try:
        # Preload the embedding model
        logger.info("🔄 Preloading embedding model...")
        preload_embedding_model()
        logger.info("✅ Embedding model preloaded - searches will be fast!")
        
        # Test OpenSearch connection
        try:
            from opensearch_client import client as opensearch_client
            info = opensearch_client.info()
            logger.info(f"✅ OpenSearch connected: {info.get('version', {}).get('number', 'unknown')}")
        except Exception as e:
            logger.warning(f"⚠️ OpenSearch connection issue: {e}")
        
        logger.info("🎉 Application startup complete - ready for fast searches with FIXED vector serialization!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        # Don't fail startup, but log the error
        logger.warning("⚠️ Continuing with lazy model loading...")

@app.get("/debug/chat/{query}")
async def debug_chat(query: str):
    """Debug endpoint to test chat functionality with detailed logging"""
    try:
        # Simulate a chat request
        chat_request = ChatRequest(
            message=query,
            programs=["Metro"],  # Test with Metro program
            history=[]
        )
        
        logger.info(f"Debug chat test for: {query}")
        
        # Call the chat function with debug logging
        response = await chat_with_agent(chat_request)
        
        return {
            "status": "success",
            "query": query,
            "chat_response": response
        }
        
    except Exception as e:
        logger.error(f"Debug chat failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }

@app.get("/debug/opensearch/indices")
async def debug_opensearch_indices():
    """Debug endpoint to check OpenSearch indices and document counts"""
    try:
        # Get all indices
        indices = opensearch_client.indices.get(index="kb-*")
        
        results = {
            "indices": {},
            "total_documents": 0
        }
        
        for index_name in indices.keys():
            try:
                # Get document count for each index
                count_response = opensearch_client.count(index=index_name)
                doc_count = count_response.get("count", 0)
                
                # Get a sample document
                search_response = opensearch_client.search(
                    index=index_name,
                    body={"size": 1, "query": {"match_all": {}}},
                    _source=["name", "program", "collection", "chunk.text", "vector"]
                )
                
                sample_doc = None
                if search_response.get("hits", {}).get("hits"):
                    sample_doc = search_response["hits"]["hits"][0]["_source"]
                
                results["indices"][index_name] = {
                    "document_count": doc_count,
                    "sample_document": sample_doc
                }
                
                results["total_documents"] += doc_count
                
            except Exception as e:
                results["indices"][index_name] = {
                    "error": str(e)
                }
        
        return {
            "status": "success",
            "results": results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)