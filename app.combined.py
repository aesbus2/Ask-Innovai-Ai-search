# Enhanced Production App.py - VECTOR SEARCH ENABLED
# Version: 4.8.0 - Full vector search integration with hybrid text+vector search
# MAJOR UPDATE: Vector search capabilities aligned with chat_handlers.py v4.8.0 and opensearch_client.py v4.7.0
# NEW: Vector embeddings, hybrid search testing, enhanced debugging with vector support

import os
import logging
import requests
import asyncio
import json
import sys
import re
import time
import gc
import hashlib
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
from collections import defaultdict
from chat_handlers import chat_router, health_router
from fastapi import APIRouter

# Create FastAPI app
app = FastAPI(
    title="Ask InnovAI Production - VECTOR SEARCH ENABLED",
    description="AI-Powered Knowledge Assistant with Vector Search + Real-Time Data Filters + Hybrid Search Strategy",
    version="4.8.0"
)

app.include_router(chat_router, prefix="/api")
app.include_router(health_router)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Production logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ask-innovai-production-vector")

# Load environment variables
load_dotenv()

# Memory monitoring setup
PSUTIL_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
    logger.info("✅ psutil available for memory monitoring")
except ImportError:
    logger.warning("⚠️ psutil not available - memory monitoring disabled")

# Import modules with error handling
try:
    from sentence_splitter import split_into_chunks
    logger.info("✅ sentence_splitter imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import sentence_splitter: {e}")
    sys.exit(1)

try:
    from opensearch_client import search_opensearch, index_document, search_vector, hybrid_search
    logger.info("✅ opensearch_client with VECTOR SEARCH imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import opensearch_client: {e}")
    sys.exit(1)

# Import embedder with fallback - VECTOR SEARCH DEPENDENCY
EMBEDDER_AVAILABLE = False
VECTOR_SEARCH_READY = False
try:
    from embedder import embed_text, get_embedding_stats, preload_embedding_model
    EMBEDDER_AVAILABLE = True
    VECTOR_SEARCH_READY = True
    logger.info("✅ embedder imported successfully - VECTOR SEARCH READY")
except ImportError as e:
    logger.warning(f"⚠️ embedder import failed: {e} - vector search will be disabled")

class ImportRequest(BaseModel):
    collection: str = "all"
    max_docs: Optional[int] = None
    import_type: str = "full"
    batch_size: Optional[int] = None

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"⚠️ Failed to mount static files: {e}")

# Production Configuration - ENHANCED FOR VECTOR SEARCH
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

# Global cache for efficient filter metadata
_filter_metadata_cache = {
    "data": None,
    "timestamp": None,
    "ttl_seconds": 300  # 5 minutes cache
}

def log_import(message: str):
    """Add message to import logs with production formatting"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    import_logs.append(log_entry)
    logger.info(message)
    if len(import_logs) > 100:
        import_logs.pop(0)

def update_import_status(status: str, step: str = None, results: dict = None, error: str = None):
    """Update import status with production tracking"""
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

# ============================================================================
# ENHANCED METADATA LOADING SYSTEM WITH VECTOR SEARCH AWARENESS
# ============================================================================

@health_router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "components": {
            "opensearch": {"status": "connected"},
            "embedding_service": {"status": "healthy" if EMBEDDER_AVAILABLE else "disabled"},
            "genai_agent": {"status": "configured"},
            "vector_search": {"status": "enabled" if VECTOR_SEARCH_READY else "disabled"}
        },
        "enhancements": {
            "document_structure": "enhanced v4.8.0",
            "vector_search": "enabled" if VECTOR_SEARCH_READY else "disabled",
            "hybrid_search": "enabled" if VECTOR_SEARCH_READY else "disabled",
            "semantic_similarity": "enabled" if VECTOR_SEARCH_READY else "disabled"
        }
    }

@health_router.get("/last_import_info")
async def last_import_info():
    return {
        "status": "success",
        "last_import_timestamp": datetime.now().isoformat(),
        "vector_search_enabled": VECTOR_SEARCH_READY
    }

@app.get("/filter_options_metadata")
async def filter_options_metadata():
    """
    ENHANCED: Get filter options with vector search capabilities detection
    """
    try:
        # Check cache first
        cached_data = get_cached_filter_metadata()
        if cached_data:
            logger.info(f"📋 Returning cached filter metadata (age: {cached_data.get('cache_age_seconds', 0):.1f}s)")
            return cached_data
        
        from opensearch_client import get_opensearch_client, test_connection, detect_vector_support
        
        if not test_connection():
            logger.warning("OpenSearch not available for filter options")
            return create_empty_filter_response("opensearch_unavailable")
        
        client = get_opensearch_client()
        if not client:
            logger.error("Could not create OpenSearch client for filter options")
            return create_empty_filter_response("client_unavailable")
        
        logger.info("🚀 Loading filter metadata with VECTOR SEARCH detection...")
        start_time = time.time()
        
        # ✅ STEP 1: Check vector search support
        vector_support = detect_vector_support(client) if client else False
        logger.info(f"🔮 Vector search support: {'✅ ENABLED' if vector_support else '❌ DISABLED'}")
        
        # STEP 2: Get all evaluation indices efficiently
        indices_info = await get_evaluation_indices_info(client)
        if not indices_info:
            return create_empty_filter_response("no_indices")
        
        # STEP 3: Extract templates from index names (much faster than aggregation)
        templates_from_indices = extract_templates_from_indices(indices_info)
        
        # STEP 4: Get field mappings to understand available metadata fields (including vector fields)
        available_fields = await get_available_metadata_fields(client, indices_info)
        
        # STEP 5: Use targeted sampling per index for metadata values
        metadata_values = await get_metadata_values_efficiently(client, indices_info, available_fields)
        
        # STEP 6: Build final response with vector search capabilities
        filter_options = {
            # Templates from index structure (fastest)
            "templates": templates_from_indices,
            
            # Metadata from efficient sampling
            "programs": metadata_values.get("programs", []),
            "partners": metadata_values.get("partners", []),
            "sites": metadata_values.get("sites", []),
            "lobs": metadata_values.get("lobs", []),
            "callDispositions": metadata_values.get("dispositions", []),
            "callSubDispositions": metadata_values.get("sub_dispositions", []),
            "agentNames": metadata_values.get("agents", []),
            "languages": metadata_values.get("languages", []),
            "callTypes": metadata_values.get("call_types", []),
            
            # Template IDs from index names
            "template_ids": [info["template_id"] for info in indices_info],
            
            # Enhanced metadata with vector search info
            "total_evaluations": sum(info["doc_count"] for info in indices_info),
            "total_indices": len(indices_info),
            "data_freshness": datetime.now().isoformat(),
            "status": "success",
            "version": "4.8.0_vector_enabled",
            "load_method": "index_structure_based",
            "load_time_ms": round((time.time() - start_time) * 1000, 2),
            "cached": False,
            
            # ✅ NEW: Vector search capabilities
            "vector_search_enabled": vector_support,
            "hybrid_search_available": vector_support and EMBEDDER_AVAILABLE,
            "semantic_similarity": vector_support and EMBEDDER_AVAILABLE,
            "search_enhancements": {
                "vector_support": vector_support,
                "embedder_available": EMBEDDER_AVAILABLE,
                "hybrid_search": vector_support and EMBEDDER_AVAILABLE,
                "search_quality": "enhanced_with_vector_similarity" if vector_support and EMBEDDER_AVAILABLE else "text_only"
            }
        }
        
        # Cache the result
        cache_filter_metadata(filter_options)
        
        # ENHANCED logging with vector search info
        logger.info(f"✅ ENHANCED metadata loading completed in {filter_options['load_time_ms']}ms:")
        logger.info(f"   📁 Indices analyzed: {len(indices_info)}")
        logger.info(f"   📋 Templates: {len(templates_from_indices)} (from index names)")
        logger.info(f"   🏢 Programs: {len(metadata_values.get('programs', []))}")
        logger.info(f"   🤝 Partners: {len(metadata_values.get('partners', []))}")
        logger.info(f"   📊 Total evaluations: {filter_options['total_evaluations']:,}")
        logger.info(f"   🔮 Vector search: {'✅ ENABLED' if vector_support else '❌ DISABLED'}")
        logger.info(f"   🔥 Hybrid search: {'✅ AVAILABLE' if filter_options['hybrid_search_available'] else '❌ NOT AVAILABLE'}")
        
        return filter_options
        
    except Exception as e:
        logger.error(f"ENHANCED: Failed to load filter options: {e}")
        return create_empty_filter_response("error", str(e))

async def get_evaluation_indices_info(client):
    """Get information about all evaluation indices efficiently"""
    try:
        # Get index stats for eval-* pattern
        stats_response = client.indices.stats(index="eval-*")
        indices_info = []
        
        for index_name, stats in stats_response.get("indices", {}).items():
            # Extract template_id from index name (eval-template-123 -> template-123)
            template_id = index_name.replace("eval-", "") if index_name.startswith("eval-") else "unknown"
            
            doc_count = stats.get("primaries", {}).get("docs", {}).get("count", 0)
            size_bytes = stats.get("primaries", {}).get("store", {}).get("size_in_bytes", 0)
            
            indices_info.append({
                "index_name": index_name,
                "template_id": template_id,
                "doc_count": doc_count,
                "size_bytes": size_bytes
            })
        
        logger.info(f"📊 Found {len(indices_info)} evaluation indices")
        return indices_info
        
    except Exception as e:
        logger.error(f"Failed to get indices info: {e}")
        return []

def extract_templates_from_indices(indices_info):
    """Extract template names efficiently by sampling one document per index"""
    templates = []
    seen_templates = set()
    
    try:
        from opensearch_client import get_opensearch_client
        client = get_opensearch_client()
        
        for index_info in indices_info:
            index_name = index_info["index_name"]
            
            # Sample one document from each index to get template_name
            try:
                sample_response = client.search(
                    index=index_name,
                    body={
                        "size": 1,
                        "query": {"match_all": {}},
                        "_source": ["template_name", "template_id"]
                    }
                )
                
                hits = sample_response.get("hits", {}).get("hits", [])
                if hits:
                    source = hits[0].get("_source", {})
                    template_name = source.get("template_name", f"Template {index_info['template_id']}")
                    
                    if template_name and template_name not in seen_templates:
                        templates.append(template_name)
                        seen_templates.add(template_name)
                        
            except Exception as e:
                logger.debug(f"Could not sample from {index_name}: {e}")
                # Fallback: create template name from index
                fallback_name = f"Template {index_info['template_id']}"
                if fallback_name not in seen_templates:
                    templates.append(fallback_name)
                    seen_templates.add(fallback_name)
        
        logger.info(f"📋 Extracted {len(templates)} templates from index sampling")
        return sorted(templates)
        
    except Exception as e:
        logger.error(f"Failed to extract templates from indices: {e}")
        return []

async def get_available_metadata_fields(client, indices_info):
    """Check index mappings to see what metadata fields are actually available (including vector fields)"""
    try:
        # Get mapping for a representative index
        sample_index = indices_info[0]["index_name"] if indices_info else "eval-*"
        
        mapping_response = client.indices.get_mapping(index=sample_index)
        
        available_fields = set()
        vector_fields = set()
        
        for index_name, mapping_data in mapping_response.items():
            properties = mapping_data.get("mappings", {}).get("properties", {})
            metadata_props = properties.get("metadata", {}).get("properties", {})
            
            # Collect available metadata fields
            for field_name in metadata_props.keys():
                available_fields.add(field_name)
            
            # ✅ NEW: Check for vector fields
            if "document_embedding" in properties:
                vector_fields.add("document_embedding")
            
            if "chunks" in properties:
                chunk_props = properties["chunks"].get("properties", {})
                if "embedding" in chunk_props:
                    vector_fields.add("chunks.embedding")
        
        logger.info(f"🔍 Available metadata fields: {sorted(available_fields)}")
        logger.info(f"🔮 Vector fields detected: {sorted(vector_fields)}")
        
        return list(available_fields)
        
    except Exception as e:
        logger.warning(f"Could not check field mappings: {e}")
        # Return expected fields as fallback
        return ["program", "partner", "site", "lob", "agent", "disposition", 
                "sub_disposition", "language", "call_type", "weighted_score", "url"]

async def get_metadata_values_efficiently(client, indices_info, available_fields):
    """Get metadata values using targeted sampling instead of full aggregation"""
    metadata_values = {
        "programs": set(),
        "partners": set(), 
        "sites": set(),
        "lobs": set(),
        "dispositions": set(),
        "sub_dispositions": set(),
        "agents": set(),
        "languages": set(),
        "call_types": set(),
        "weighted_scores": set(),
        "urls": set()
    }
    
    field_mapping = {
        "program": "programs",
        "partner": "partners",
        "site": "sites", 
        "lob": "lobs",
        "disposition": "dispositions",
        "sub_disposition": "sub_dispositions",
        "agent": "agents",
        "language": "languages",
        "call_type": "call_types",
        "weighted_score": "weighted_scores",
        "url": "urls"
    }
    
    try:
        # Sample from each index (limited samples for speed)
        samples_per_index = 10
        
        for index_info in indices_info[:10]:  # Limit to top 10 indices for speed
            index_name = index_info["index_name"]
            
            try:
                # Get sample documents with metadata
                sample_response = client.search(
                    index=index_name,
                    body={
                        "size": samples_per_index,
                        "query": {"match_all": {}},
                        "_source": ["metadata"],
                        "sort": [{"_doc": {"order": "asc"}}]  # Fast sampling
                    }
                )
                
                hits = sample_response.get("hits", {}).get("hits", [])
                
                for hit in hits:
                    metadata = hit.get("_source", {}).get("metadata", {})
                    
                    # Extract values for each field
                    for opensearch_field, result_key in field_mapping.items():
                        if opensearch_field in available_fields:
                            value = metadata.get(opensearch_field)
                            if value and isinstance(value, str) and value.strip():
                                cleaned_value = value.strip()
                                if cleaned_value.lower() not in ["unknown", "null", "", "n/a"]:
                                    metadata_values[result_key].add(cleaned_value)
                
            except Exception as e:
                logger.debug(f"Could not sample metadata from {index_name}: {e}")
        
        # Convert sets to sorted lists
        result = {}
        for key, value_set in metadata_values.items():
            result[key] = sorted(list(value_set))
            logger.info(f"   {key}: {len(result[key])} unique values")
        
        logger.info(f"📊 Metadata sampling completed from {len(indices_info)} indices")
        return result
        
    except Exception as e:
        logger.error(f"Failed to sample metadata efficiently: {e}")
        return {key: [] for key in metadata_values.keys()}

def get_cached_filter_metadata():
    """Get cached filter metadata if still valid"""
    try:
        cache = _filter_metadata_cache
        
        if not cache["data"] or not cache["timestamp"]:
            return None
        
        # Check if cache is expired
        cache_age = time.time() - cache["timestamp"]
        if cache_age > cache["ttl_seconds"]:
            logger.info(f"📋 Filter cache expired (age: {cache_age:.1f}s > {cache['ttl_seconds']}s)")
            return None
        
        # Add cache metadata to response
        cached_data = cache["data"].copy()
        cached_data["cached"] = True
        cached_data["cache_age_seconds"] = cache_age
        cached_data["cache_expires_in_seconds"] = cache["ttl_seconds"] - cache_age
        
        return cached_data
        
    except Exception as e:
        logger.warning(f"Cache retrieval failed: {e}")
        return None

def cache_filter_metadata(data):
    """Cache filter metadata for faster subsequent requests"""
    try:
        _filter_metadata_cache["data"] = data.copy()
        _filter_metadata_cache["timestamp"] = time.time()
        
        logger.info(f"📋 Filter metadata cached for {_filter_metadata_cache['ttl_seconds']}s")
        
    except Exception as e:
        logger.warning(f"Failed to cache filter metadata: {e}")

def create_empty_filter_response(status="no_data", error_msg=""):
    """Create empty filter response for error cases"""
    return {
        "templates": [],
        "programs": [],
        "partners": [],
        "sites": [],
        "lobs": [],
        "callDispositions": [],
        "callSubDispositions": [],
        "agentNames": [],
        "languages": [],
        "callTypes": [],
        "template_ids": [],
        "total_evaluations": 0,
        "total_indices": 0,
        "status": status,
        "error": error_msg,
        "message": f"No filter data available: {error_msg}" if error_msg else "No data available",
        "version": "4.8.0_vector_enabled",
        "load_method": "fallback",
        "vector_search_enabled": False,
        "hybrid_search_available": False
    }

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

@app.post("/clear_filter_cache")
async def clear_filter_cache():
    """Clear the filter metadata cache (useful after data imports)"""
    try:
        _filter_metadata_cache["data"] = None
        _filter_metadata_cache["timestamp"] = None
        
        return {
            "status": "success",
            "message": "Filter metadata cache cleared",
            "timestamp": datetime.now().isoformat(),
            "vector_search_ready": VECTOR_SEARCH_READY
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# ENHANCED DEBUG ENDPOINTS WITH VECTOR SEARCH TESTING
# ============================================================================

@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint to show all available routes"""
    routes = []
    for route in app.routes:
        routes.append({
            "path": getattr(route, 'path', 'Unknown'),
            "methods": getattr(route, 'methods', 'Unknown'),
            "name": getattr(route, 'name', 'Unknown')
        })
    
    return {
        "total_routes": len(routes),
        "routes": routes,
        "chat_endpoint_status": "directly_defined_in_app_py",
        "vector_search_enabled": VECTOR_SEARCH_READY,
        "hybrid_search_available": VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE,
        "version": "4.8.0_vector_enabled"
    }

@app.get("/debug/test_vector_search")
async def debug_test_vector_search(query: str = "customer service"):
    """
    ✅ NEW: Test vector search functionality
    """
    if not VECTOR_SEARCH_READY:
        return {
            "status": "disabled",
            "message": "Vector search not available - embedder not imported",
            "requirements": ["embedder module", "OpenSearch vector support"],
            "version": "4.8.0_vector_enabled"
        }
    
    try:
        from opensearch_client import get_opensearch_client, detect_vector_support, search_vector
        from embedder import embed_text
        
        client = get_opensearch_client()
        if not client:
            return {"error": "OpenSearch client not available"}
        
        # Test vector support
        vector_support = detect_vector_support(client)
        if not vector_support:
            return {
                "status": "unsupported",
                "message": "OpenSearch cluster does not support vector search",
                "cluster_support": False,
                "version": "4.8.0_vector_enabled"
            }
        
        # Generate query vector
        logger.info(f"🔮 Testing vector search with query: '{query}'")
        query_vector = embed_text(query)
        
        # Perform vector search
        vector_results = search_vector(query_vector, size=5)
        
        # Analyze results
        result_analysis = []
        for i, result in enumerate(vector_results):
            analysis = {
                "result_index": i + 1,
                "evaluation_id": result.get("evaluationId"),
                "score": result.get("_score", 0),
                "search_type": result.get("search_type"),
                "template_name": result.get("template_name"),
                "has_vector_dimension": "vector_dimension" in result,
                "vector_dimension": result.get("vector_dimension"),
                "best_matching_chunks": len(result.get("best_matching_chunks", [])),
                "metadata_program": result.get("metadata", {}).get("program"),
                "metadata_disposition": result.get("metadata", {}).get("disposition")
            }
            result_analysis.append(analysis)
        
        return {
            "status": "success",
            "vector_search_test": {
                "query": query,
                "query_vector_dimension": len(query_vector),
                "cluster_vector_support": vector_support,
                "results_found": len(vector_results),
                "embedding_generation": "successful",
                "search_execution": "successful"
            },
            "results_analysis": result_analysis,
            "recommendations": [
                "Vector search is working correctly" if vector_results else "No vector results found - check if data has embeddings",
                "Results should have vector scores and dimensions",
                "Check best_matching_chunks for relevant content pieces"
            ],
            "version": "4.8.0_vector_enabled"
        }
        
    except Exception as e:
        logger.error(f"❌ Vector search test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query,
            "version": "4.8.0_vector_enabled"
        }

@app.get("/debug/test_hybrid_search")
async def debug_test_hybrid_search(query: str = "call dispositions"):
    """
    ✅ NEW: Test hybrid search functionality (text + vector)
    """
    if not VECTOR_SEARCH_READY:
        return {
            "status": "disabled",
            "message": "Hybrid search not available - requires vector search",
            "version": "4.8.0_vector_enabled"
        }
    
    try:
        from opensearch_client import get_opensearch_client, detect_vector_support, hybrid_search
        from embedder import embed_text
        
        client = get_opensearch_client()
        if not client:
            return {"error": "OpenSearch client not available"}
        
        # Test vector support
        vector_support = detect_vector_support(client)
        if not vector_support:
            return {
                "status": "unsupported",
                "message": "Hybrid search requires vector support in OpenSearch cluster",
                "version": "4.8.0_vector_enabled"
            }
        
        # Generate query vector
        logger.info(f"🔥 Testing hybrid search with query: '{query}'")
        query_vector = embed_text(query)
        
        # Perform hybrid search with different vector weights
        test_weights = [0.3, 0.6, 0.8]
        hybrid_test_results = {}
        
        for weight in test_weights:
            try:
                hybrid_results = hybrid_search(
                    query=query,
                    query_vector=query_vector,
                    size=5,
                    vector_weight=weight
                )
                
                hybrid_test_results[f"weight_{weight}"] = {
                    "results_count": len(hybrid_results),
                    "average_score": sum(r.get("_score", 0) for r in hybrid_results) / len(hybrid_results) if hybrid_results else 0,
                    "search_types": list(set(r.get("search_type") for r in hybrid_results)),
                    "sample_results": [
                        {
                            "evaluation_id": r.get("evaluationId"),
                            "score": r.get("_score", 0),
                            "hybrid_score": r.get("hybrid_score", 0),
                            "search_type": r.get("search_type"),
                            "template_name": r.get("template_name")
                        }
                        for r in hybrid_results[:3]
                    ]
                }
                
            except Exception as e:
                hybrid_test_results[f"weight_{weight}"] = {"error": str(e)}
        
        return {
            "status": "success",
            "hybrid_search_test": {
                "query": query,
                "query_vector_dimension": len(query_vector),
                "weights_tested": test_weights,
                "cluster_support": vector_support
            },
            "weight_comparison": hybrid_test_results,
            "analysis": {
                "best_weight": max(test_weights, key=lambda w: hybrid_test_results.get(f"weight_{w}", {}).get("results_count", 0)),
                "total_unique_results": len(set().union(*[
                    [r["evaluation_id"] for r in hybrid_test_results.get(f"weight_{w}", {}).get("sample_results", [])]
                    for w in test_weights
                ])),
                "search_quality": "hybrid text+vector search active"
            },
            "recommendations": [
                "Compare different vector weights to see which gives best results for your data",
                "Higher vector weights (0.6-0.8) emphasize semantic similarity",
                "Lower vector weights (0.2-0.4) emphasize text matching",
                "Use weight 0.6 as a good balance for most queries"
            ],
            "version": "4.8.0_vector_enabled"
        }
        
    except Exception as e:
        logger.error(f"❌ Hybrid search test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query,
            "version": "4.8.0_vector_enabled"
        }

@app.get("/debug/vector_capabilities")
async def debug_vector_capabilities():
    """
    ✅ NEW: Check overall vector search capabilities and status
    """
    try:
        from opensearch_client import get_opensearch_client, detect_vector_support, get_available_fields
        
        capabilities = {
            "embedder_module": EMBEDDER_AVAILABLE,
            "vector_search_ready": VECTOR_SEARCH_READY,
            "opensearch_client": False,
            "cluster_vector_support": False,
            "vector_fields_detected": [],
            "hybrid_search_available": False,
            "search_enhancements": []
        }
        
        # Test OpenSearch client
        client = get_opensearch_client()
        if client:
            capabilities["opensearch_client"] = True
            
            # Test vector support
            vector_support = detect_vector_support(client)
            capabilities["cluster_vector_support"] = vector_support
            
            if vector_support:
                # Get available fields including vector fields
                available_fields = get_available_fields(client)
                capabilities["vector_fields_detected"] = available_fields.get("vector_fields", [])
                capabilities["has_vector_mapping"] = available_fields.get("has_vector_support", False)
                
                # Check hybrid search availability
                if EMBEDDER_AVAILABLE:
                    capabilities["hybrid_search_available"] = True
                    capabilities["search_enhancements"] = [
                        "text_search",
                        "vector_similarity_search", 
                        "hybrid_text_vector_search",
                        "semantic_similarity_matching"
                    ]
        
        # Test embedding generation if available
        embedding_test = {"available": False, "dimension": 0, "generation_time": 0}
        if EMBEDDER_AVAILABLE:
            try:
                from embedder import embed_text
                start_time = time.time()
                test_vector = embed_text("test query")
                embedding_test = {
                    "available": True,
                    "dimension": len(test_vector),
                    "generation_time": round((time.time() - start_time) * 1000, 2)
                }
            except Exception as e:
                embedding_test["error"] = str(e)
        
        # Overall status assessment
        overall_status = "disabled"
        if capabilities["embedder_module"] and capabilities["opensearch_client"]:
            if capabilities["cluster_vector_support"] and capabilities["hybrid_search_available"]:
                overall_status = "fully_enabled"
            elif capabilities["cluster_vector_support"]:
                overall_status = "vector_only"
            else:
                overall_status = "text_only"
        
        return {
            "status": "success",
            "overall_vector_status": overall_status,
            "capabilities": capabilities,
            "embedding_test": embedding_test,
            "recommendations": {
                "fully_enabled": "All vector search features are available and working",
                "vector_only": "Vector search available but embedder missing - install embedder module",
                "text_only": "Only text search available - cluster doesn't support vectors",
                "disabled": "Vector search completely disabled - check OpenSearch connection and embedder module"
            }.get(overall_status, "Unknown status"),
            "next_steps": get_vector_setup_recommendations(overall_status),
            "version": "4.8.0_vector_enabled"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "4.8.0_vector_enabled"
        }

def get_vector_setup_recommendations(status):
    """Get recommendations based on vector search status"""
    recommendations = {
        "fully_enabled": [
            "Vector search is fully operational",
            "Test with /debug/test_vector_search and /debug/test_hybrid_search",
            "Use hybrid search for best results"
        ],
        "vector_only": [
            "Install embedder module: pip install sentence-transformers",
            "Restart the application after installing embedder",
            "Vector search will be enabled after embedder installation"
        ],
        "text_only": [
            "OpenSearch cluster needs vector search plugin",
            "Contact your OpenSearch administrator to enable kNN plugin",
            "Text search will continue to work normally"
        ],
        "disabled": [
            "Check OpenSearch connection in environment variables",
            "Install embedder module if not available", 
            "Verify OpenSearch cluster supports vector search",
            "Check application logs for specific errors"
        ]
    }
    return recommendations.get(status, ["Check application logs for specific issues"])

# Update existing debug endpoints to include vector search information

@app.get("/debug/opensearch_data")
async def debug_opensearch_data():
    """Enhanced debug endpoint that checks for vector data"""
    try:
        from opensearch_client import get_opensearch_client, debug_search_simple, detect_vector_support
        
        client = get_opensearch_client()
        if not client:
            return {"error": "OpenSearch client not available"}
        
        # Get basic stats
        simple_result = debug_search_simple()
        
        # ✅ NEW: Check vector support
        vector_support = detect_vector_support(client)
        
        # Get sample documents with vector field analysis
        try:
            response = client.search(
                index="eval-*",
                body={
                    "query": {"match_all": {}},
                    "size": 3,
                    "_source": True
                }
            )
            
            hits = response.get("hits", {}).get("hits", [])
            sample_docs = []
            
            for hit in hits:
                source = hit.get("_source", {})
                doc_summary = {
                    "index": hit.get("_index"),
                    "id": hit.get("_id"),
                    "evaluationId": source.get("evaluationId"),
                    "template_name": source.get("template_name"),
                    "template_id": source.get("template_id"),
                    "has_full_text": bool(source.get("full_text")),
                    "has_evaluation_text": bool(source.get("evaluation_text")),
                    "has_transcript_text": bool(source.get("transcript_text")),
                    "has_evaluation": bool(source.get("evaluation")),
                    "has_transcript": bool(source.get("transcript")),
                    "total_chunks": source.get("total_chunks", 0),
                    "chunks_count": len(source.get("chunks", [])),
                    
                    # ✅ NEW: Vector field analysis
                    "has_document_embedding": bool(source.get("document_embedding")),
                    "document_embedding_dimension": len(source.get("document_embedding", [])),
                    "has_chunk_embeddings": False,
                    "chunk_embeddings_count": 0,
                    
                    "metadata_keys": list(source.get("metadata", {}).keys()),
                    "metadata_sample": {
                        "program": source.get("metadata", {}).get("program"),
                        "partner": source.get("metadata", {}).get("partner"),
                        "site": source.get("metadata", {}).get("site"),
                        "lob": source.get("metadata", {}).get("lob"),
                        "agent": source.get("metadata", {}).get("agent"),
                        "disposition": source.get("metadata", {}).get("disposition"),
                        "language": source.get("metadata", {}).get("language"),
                        "weighted_score": source.get("metadata", {}).get("weighted_score"),
                        "url": source.get("metadata", {}).get("url")
                    },
                    "content_preview": {
                        "full_text": source.get("full_text", "")[:200],
                        "evaluation": source.get("evaluation", "")[:200],
                        "transcript": source.get("transcript", "")[:200],
                        "first_chunk": source.get("chunks", [{}])[0].get("text", "")[:200] if source.get("chunks") else ""
                    }
                }
                
                # Check for chunk embeddings
                chunks = source.get("chunks", [])
                if chunks and isinstance(chunks, list):
                    chunk_embeddings = [chunk for chunk in chunks if isinstance(chunk, dict) and "embedding" in chunk]
                    doc_summary["has_chunk_embeddings"] = len(chunk_embeddings) > 0
                    doc_summary["chunk_embeddings_count"] = len(chunk_embeddings)
                
                sample_docs.append(doc_summary)
            
            return {
                "status": "success",
                "simple_search_result": simple_result,
                "vector_analysis": {
                    "cluster_vector_support": vector_support,
                    "vector_search_ready": VECTOR_SEARCH_READY,
                    "embedder_available": EMBEDDER_AVAILABLE
                },
                "sample_documents": sample_docs,
                "total_indices": len(response.get("hits", {}).get("hits", [])),
                "message": "Enhanced data verification with vector search analysis",
                "version": "4.8.0_vector_enabled"
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "simple_search_result": simple_result,
                "vector_analysis": {
                    "cluster_vector_support": vector_support,
                    "vector_search_ready": VECTOR_SEARCH_READY,
                    "embedder_available": EMBEDDER_AVAILABLE
                },
                "version": "4.8.0_vector_enabled"
            }
            
    except Exception as e:
        return {"error": str(e), "version": "4.8.0_vector_enabled"}

# ============================================================================
# ENHANCED STATISTICS WITH VECTOR SEARCH INFO
# ============================================================================

@app.get("/opensearch_statistics")
async def get_opensearch_statistics():
    """Enhanced OpenSearch statistics with vector search information"""
    try:
        from opensearch_client import get_opensearch_client, test_connection, detect_vector_support, get_available_fields
        
        if not test_connection():
            return {
                "status": "error",
                "error": "OpenSearch connection failed",
                "timestamp": datetime.now().isoformat()
            }
        
        client = get_opensearch_client()
        start_time = time.time()
        
        # STEP 1: Get basic document count
        try:
            count_response = client.count(
                index="eval-*", 
                body={"query": {"match_all": {}}},
                request_timeout=10
            )
            total_documents = count_response.get("count", 0)
            logger.info(f"✅ Document count: {total_documents}")
        except Exception as e:
            logger.warning(f"Count query failed: {e}")
            total_documents = 0
        
        # STEP 2: Get indices information
        try:
            indices_response = client.cat.indices(index="eval-*", format="json", request_timeout=10)
            active_indices = len(indices_response) if indices_response else 0
        except Exception as e:
            logger.warning(f"Indices query failed: {e}")
            active_indices = 0
        
        # ✅ STEP 3: Check vector search capabilities
        vector_support = detect_vector_support(client)
        available_fields = get_available_fields(client) if vector_support else {"vector_fields": [], "has_vector_support": False}
        
        # STEP 4: Get available metadata fields from mapping
        try:
            mapping_response = client.indices.get_mapping(index="eval-*", request_timeout=10)
            available_fields_mapping = set()
            
            for index_name, mapping_data in mapping_response.items():
                properties = mapping_data.get("mappings", {}).get("properties", {})
                metadata_props = properties.get("metadata", {}).get("properties", {})
                
                # Collect available metadata fields
                for field_name in metadata_props.keys():
                    available_fields_mapping.add(field_name)
                    
        except Exception as e:
            logger.warning(f"Mapping query failed: {e}")
            available_fields_mapping = set()
        
        # STEP 5: Sample documents to extract real statistics
        statistics = {
            "templates": set(),
            "programs": set(),
            "partners": set(),
            "sites": set(),
            "lobs": set(),
            "dispositions": set(),
            "sub_dispositions": set(),
            "agents": set(),
            "languages": set(),
            "call_types": set(),
            "weighted_scores": [],
            "urls": set(),
            "call_durations": [],
            
            # ✅ NEW: Vector search statistics
            "documents_with_vector_embeddings": 0,
            "documents_with_chunk_embeddings": 0,
            "vector_dimensions": set(),
            "vector_search_ready": vector_support and EMBEDDER_AVAILABLE
        }
        
        evaluations_sampled = set()
        chunks_sampled = 0
        
        try:
            # Sample a good number of documents to get comprehensive statistics
            sample_response = client.search(
                index="eval-*",
                body={
                    "size": 200,  # Sample more documents for better statistics
                    "query": {"match_all": {}},
                    "_source": [
                        "template_name", "template_id", "evaluationId", "internalId",
                        "metadata.program", "metadata.partner", "metadata.site", 
                        "metadata.lob", "metadata.disposition", "metadata.sub_disposition",
                        "metadata.subDisposition", "metadata.agent", "metadata.agentName",
                        "metadata.language", "metadata.call_type", "metadata.weighted_score",
                        "metadata.url", "metadata.call_duration",
                        # ✅ NEW: Vector fields
                        "document_embedding", "chunks.embedding"
                    ]
                },
                request_timeout=15
            )
            
            hits = sample_response.get("hits", {}).get("hits", [])
            chunks_sampled = len(hits)
            
            for hit in hits:
                source = hit.get("_source", {})
                metadata = source.get("metadata", {})
                
                # Track unique evaluations
                evaluation_id = source.get("evaluationId") or source.get("internalId")
                if evaluation_id:
                    evaluations_sampled.add(evaluation_id)
                
                # Extract template information
                if source.get("template_name"):
                    statistics["templates"].add(source["template_name"])
                
                # Extract metadata values using the same logic as filter system
                if metadata.get("program"):
                    statistics["programs"].add(metadata["program"])
                if metadata.get("partner"):
                    statistics["partners"].add(metadata["partner"])
                if metadata.get("site"):
                    statistics["sites"].add(metadata["site"])
                if metadata.get("lob"):
                    statistics["lobs"].add(metadata["lob"])
                if metadata.get("disposition"):
                    statistics["dispositions"].add(metadata["disposition"])
                    
                # Handle both sub_disposition and subDisposition
                sub_disp = metadata.get("sub_disposition") or metadata.get("subDisposition")
                if sub_disp:
                    statistics["sub_dispositions"].add(sub_disp)
                    
                # Handle both agent and agentName
                agent = metadata.get("agent") or metadata.get("agentName")
                if agent:
                    statistics["agents"].add(agent)
                    
                if metadata.get("language"):
                    statistics["languages"].add(metadata["language"])
                if metadata.get("call_type"):
                    statistics["call_types"].add(metadata["call_type"])
                    
                # Collect weighted scores for analysis
                if metadata.get("weighted_score"):
                    try:
                        score = float(metadata["weighted_score"])
                        statistics["weighted_scores"].append(score)
                    except (ValueError, TypeError):
                        pass
                        
                # Collect URLs
                if metadata.get("url"):
                    statistics["urls"].add(metadata["url"])
                    
                # Collect call durations
                if metadata.get("call_duration"):
                    try:
                        duration = float(metadata["call_duration"])
                        statistics["call_durations"].append(duration)
                    except (ValueError, TypeError):
                        pass
                
                # ✅ NEW: Check for vector embeddings
                if source.get("document_embedding"):
                    statistics["documents_with_vector_embeddings"] += 1
                    embedding = source.get("document_embedding")
                    if isinstance(embedding, list):
                        statistics["vector_dimensions"].add(len(embedding))
                
                # Check for chunk embeddings
                chunks = source.get("chunks", [])
                if chunks and isinstance(chunks, list):
                    has_chunk_embeddings = any(
                        isinstance(chunk, dict) and "embedding" in chunk 
                        for chunk in chunks
                    )
                    if has_chunk_embeddings:
                        statistics["documents_with_chunk_embeddings"] += 1
        
        except Exception as e:
            logger.warning(f"Document sampling failed: {e}")
            # Keep empty sets as defaults
        
        # Calculate additional statistics
        avg_weighted_score = None
        if statistics["weighted_scores"]:
            avg_weighted_score = sum(statistics["weighted_scores"]) / len(statistics["weighted_scores"])
            
        avg_call_duration = None
        if statistics["call_durations"]:
            avg_call_duration = sum(statistics["call_durations"]) / len(statistics["call_durations"])
        
        # Get cluster health for additional info
        try:
            cluster_health = client.cluster.health(request_timeout=5)
            cluster_status = cluster_health.get("status", "unknown")
        except Exception as e:
            logger.warning(f"Cluster health check failed: {e}")
            cluster_status = "unknown"
        
        # Build enhanced response with vector search information
        response_data = {
            "status": "success",
            "data": {
                # Basic counts
                "total_documents": total_documents,
                "active_indices": active_indices,
                "available_fields": sorted(list(available_fields_mapping)),
                "cluster_status": cluster_status,
                
                # ✅ NEW: Vector search capabilities
                "vector_search": {
                    "cluster_support": vector_support,
                    "embedder_available": EMBEDDER_AVAILABLE,
                    "vector_search_ready": vector_support and EMBEDDER_AVAILABLE,
                    "vector_fields_detected": available_fields.get("vector_fields", []),
                    "documents_with_vectors": statistics["documents_with_vector_embeddings"],
                    "documents_with_chunk_vectors": statistics["documents_with_chunk_embeddings"],
                    "vector_dimensions": sorted(list(statistics["vector_dimensions"])),
                    "vector_coverage": round(
                        (statistics["documents_with_vector_embeddings"] / chunks_sampled * 100), 1
                    ) if chunks_sampled > 0 else 0
                },
                
                # Statistics from sampling (what the dashboard needs)
                "templates": len(statistics["templates"]),
                "programs": len(statistics["programs"]),
                "partners": len(statistics["partners"]),
                "sites": len(statistics["sites"]),
                "lobs": len(statistics["lobs"]),
                "dispositions": len(statistics["dispositions"]),
                "sub_dispositions": len(statistics["sub_dispositions"]),
                "agents": len(statistics["agents"]),
                "languages": len(statistics["languages"]),
                "call_types": len(statistics["call_types"]),
                "urls": len(statistics["urls"]),
                
                # Evaluation vs chunk analysis
                "evaluations_sampled": len(evaluations_sampled),
                "chunks_sampled": chunks_sampled,
                "avg_chunks_per_evaluation": round(chunks_sampled / len(evaluations_sampled), 1) if evaluations_sampled else 0,
                
                # Additional metrics
                "avg_weighted_score": round(avg_weighted_score, 1) if avg_weighted_score else None,
                "avg_call_duration": round(avg_call_duration, 1) if avg_call_duration else None,
                "weighted_scores_available": len(statistics["weighted_scores"]),
                "call_durations_available": len(statistics["call_durations"]),
                
                # Lists for debugging (first 10 items each)
                "templates_list": sorted(list(statistics["templates"]))[:10],
                "programs_list": sorted(list(statistics["programs"]))[:10],
                "partners_list": sorted(list(statistics["partners"]))[:10],
                "sites_list": sorted(list(statistics["sites"]))[:10],
                "lobs_list": sorted(list(statistics["lobs"]))[:10],
                "dispositions_list": sorted(list(statistics["dispositions"]))[:10],
                "agents_list": sorted(list(statistics["agents"]))[:10],
                "languages_list": sorted(list(statistics["languages"]))[:10]
            },
            "timestamp": datetime.now().isoformat(),
            "processing_time": round(time.time() - start_time, 2),
            "method": "document_sampling_with_vector_analysis",
            "sample_size": chunks_sampled,
            "version": "4.8.0_vector_enabled"
        }
        
        logger.info("✅ Enhanced statistics with vector search analysis completed")
        return response_data
        
    except Exception as e:
        logger.error(f"Statistics generation failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "method": "document_sampling_with_vector_analysis",
            "version": "4.8.0_vector_enabled"
        }

# ============================================================================
# ENHANCED PING AND HEALTH WITH VECTOR SEARCH INFO
# ============================================================================

@app.get("/ping")
async def ping():
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "service": "ask-innovai-production-vector",
        "version": "4.8.0_vector_enabled",
        "chat_fix": "endpoint_moved_to_app_py",
        "features": {
            "chat_routing": "direct_in_app_py",
            "method_allowed": "POST",
            "cors_enabled": True,
            "405_error": "fixed",
            "real_data_filters": True,
            "evaluation_grouping": True,
            "template_id_collections": True,
            "program_extraction": True,
            "efficient_metadata_loading": True,
            "filter_caching": True,
            
            # ✅ NEW: Vector search features
            "vector_search_enabled": VECTOR_SEARCH_READY,
            "hybrid_search_available": VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE,
            "semantic_similarity": VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE,
            "enhanced_relevance": VECTOR_SEARCH_READY
        }
    }

@app.get("/health")
async def health():
    """Enhanced health check with comprehensive vector search information"""
    try:
        components = {}
        
        # GenAI status
        components["genai"] = {
            "status": "configured" if GENAI_ACCESS_KEY else "not configured"
        }
        
        # Enhanced OpenSearch status with vector search
        try:
            from opensearch_client import get_connection_status, test_connection, get_opensearch_config, detect_vector_support, get_available_fields
            
            config = get_opensearch_config()
            conn_status = get_connection_status()
            
            if config["host"] == "not_configured":
                components["opensearch"] = {
                    "status": "not configured",
                    "message": "OPENSEARCH_HOST not set"
                }
            elif conn_status["connected"]:
                # Get vector search capabilities
                client = get_opensearch_client()
                vector_support = detect_vector_support(client) if client else False
                vector_fields = []
                
                if vector_support:
                    try:
                        available_fields = get_available_fields(client)
                        vector_fields = available_fields.get("vector_fields", [])
                    except Exception:
                        pass
                
                components["opensearch"] = {
                    "status": "connected",
                    "host": config["host"],
                    "port": config["port"],
                    "document_structure": "evaluation_grouped",
                    "collection_strategy": "template_id_based",
                    "real_data_filters": True,
                    "efficient_metadata": True,
                    
                    # ✅ NEW: Vector search status
                    "vector_search_support": vector_support,
                    "vector_fields_detected": vector_fields,
                    "hybrid_search_ready": vector_support and EMBEDDER_AVAILABLE
                }
            else:
                components["opensearch"] = {
                    "status": "connection_failed",
                    "host": config["host"],
                    "port": config["port"],
                    "error": conn_status.get("last_error", "Unknown error")[:100]
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
        
        # Enhanced embedder status with vector search info
        if EMBEDDER_AVAILABLE:
            try:
                stats = get_embedding_stats()
                components["embeddings"] = {
                    "status": "healthy", 
                    "model_loaded": stats.get("model_loaded", False),
                    "vector_search_enabled": True,
                    "embedding_dimension": stats.get("embedding_dimension", "unknown")
                }
            except Exception:
                components["embeddings"] = {
                    "status": "warning",
                    "vector_search_enabled": True
                }
        else:
            components["embeddings"] = {
                "status": "not available",
                "vector_search_enabled": False,
                "recommendation": "Install sentence-transformers for vector search"
            }
        
        # Filter cache status
        cache = _filter_metadata_cache
        cache_status = "empty"
        if cache["data"] and cache["timestamp"]:
            cache_age = time.time() - cache["timestamp"]
            if cache_age <= cache["ttl_seconds"]:
                cache_status = f"valid ({cache_age:.0f}s old)"
            else:
                cache_status = f"expired ({cache_age:.0f}s old)"
        
        components["filter_cache"] = {
            "status": cache_status,
            "ttl_seconds": cache["ttl_seconds"]
        }
        
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
                "version": "4.8.0_vector_enabled",
                "features": {
                    "real_data_filters": True,
                    "evaluation_grouping": True,
                    "template_id_collections": True,
                    "program_extraction": True,
                    "comprehensive_metadata": True,
                    "efficient_metadata_loading": True,
                    "filter_caching": True,
                    
                    # ✅ NEW: Vector search features
                    "vector_search": VECTOR_SEARCH_READY,
                    "hybrid_search": VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE,
                    "semantic_similarity": VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE,
                    "enhanced_search_relevance": VECTOR_SEARCH_READY
                },
                "search_capabilities": {
                    "text_search": "enabled",
                    "vector_search": "enabled" if VECTOR_SEARCH_READY else "disabled",
                    "hybrid_search": "enabled" if VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE else "disabled",
                    "search_quality": "enhanced_with_vector_similarity" if VECTOR_SEARCH_READY else "text_only"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"PRODUCTION: Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

# Keep existing endpoints with enhanced vector search awareness
@app.get("/status")
async def get_import_status():
    """Get import status with vector search information"""
    enhanced_status = import_status.copy()
    enhanced_status["structure_version"] = "4.8.0_vector_enabled"
    enhanced_status["document_strategy"] = "evaluation_grouped"
    enhanced_status["collection_strategy"] = "template_id_based"
    enhanced_status["real_data_filters"] = True
    enhanced_status["efficient_metadata"] = True
    enhanced_status["vector_search_enabled"] = VECTOR_SEARCH_READY
    enhanced_status["hybrid_search_available"] = VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE
    return enhanced_status

@app.get("/logs")
async def get_logs():
    """Get import logs"""
    return {
        "status": "success",
        "logs": import_logs,
        "count": len(import_logs),
        "version": "4.8.0_vector_enabled",
        "vector_search_ready": VECTOR_SEARCH_READY
    }

# Enhanced startup event with vector search info
@app.on_event("startup")
async def startup_event():
    """Enhanced startup with comprehensive vector search logging"""
    try:
        logger.info("🚀 Ask InnovAI PRODUCTION with VECTOR SEARCH starting...")
        logger.info(f"   Version: 4.8.0_vector_enabled")
        logger.info(f"   🔮 VECTOR SEARCH: {'✅ ENABLED' if VECTOR_SEARCH_READY else '❌ DISABLED'}")
        logger.info(f"   🔥 HYBRID SEARCH: {'✅ AVAILABLE' if VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE else '❌ NOT AVAILABLE'}")
        logger.info(f"   📚 EMBEDDER: {'✅ LOADED' if EMBEDDER_AVAILABLE else '❌ NOT AVAILABLE'}")
        logger.info(f"   Features: Vector Search + Real Data Filters + Hybrid Search + Semantic Similarity")
        logger.info(f"   Collection Strategy: Template_ID-based")
        logger.info(f"   Document Strategy: Evaluation-grouped with vector embeddings")
        logger.info(f"   Search Enhancement: {'Vector similarity enabled' if VECTOR_SEARCH_READY else 'Text-only search'}")
        logger.info(f"   Port: {os.getenv('PORT', '8080')}")
        logger.info(f"   Memory Monitoring: {'✅ Available' if PSUTIL_AVAILABLE else '❌ Disabled'}")
        
        # Check configuration
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
                logger.info("✅ Embedding model preloaded for vector search")
            except Exception as e:
                logger.warning(f"⚠️ Embedding preload failed: {e}")
        
        logger.info("🎉 PRODUCTION startup complete with VECTOR SEARCH CAPABILITIES")
        logger.info("📊 Ready for enhanced search with semantic similarity matching")
        logger.info("🔮 Vector search enables finding semantically similar content beyond keyword matching")
        logger.info("🔥 Hybrid search combines text matching with vector similarity for best results")
        
    except Exception as e:
        logger.error(f"❌ PRODUCTION startup error: {e}")

# For Digital Ocean App Platform
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    logger.info(f"🚀 Starting Ask InnovAI PRODUCTION with VECTOR SEARCH on port {port}")
    logger.info("🎯 Features: Vector Search + Hybrid Search + Real Data Filters + Semantic Similarity")
    logger.info(f"💾 Memory monitoring: {'✅ Enabled' if PSUTIL_AVAILABLE else '❌ Disabled'}")
    logger.info(f"🔮 Vector search: {'✅ ENABLED' if VECTOR_SEARCH_READY else '❌ DISABLED'}")
    logger.info(f"🔥 Hybrid search: {'✅ AVAILABLE' if VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE else '❌ NOT AVAILABLE'}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )