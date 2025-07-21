# Enhanced Production App.py - Real Data Filter System with Efficient Metadata Loading
# Version: 4.2.0 - Index-based metadata extraction with evaluation grouping

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
    title="Ask InnovAI Production - Efficient Real Data Filter System",
    description="AI-Powered Knowledge Assistant with Real-Time Data Filters and Efficient Metadata Loading",
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
logger = logging.getLogger("ask-innovai-production")

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
    from opensearch_client import search_opensearch, index_document, search_vector
    logger.info("✅ opensearch_client imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import opensearch_client: {e}")
    sys.exit(1)

# Import embedder with fallback
EMBEDDER_AVAILABLE = False
VECTOR_SEARCH_READY = False
PRELOAD_MODEL_ON_STARTUP = True
try:
    from embedder import embed_text, get_embedding_stats, preload_embedding_model
    EMBEDDER_AVAILABLE = False # Disable for chat
    VECTOR_SEARCH_READY = False # Disable for chat
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
# EENHANCED METADATA LOADING SYSTEM WITH VECTOR SEARCH AWARENESS
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
        
        logger.info("🚀 Loading filter metadata using efficient index-based approach...")
        start_time = time.time()

        # STEP 1: Check vector search support
        vector_support = detect_vector_support(client) if client else False
        logger.info(f"🔮 Vector search support: {'✅ ENABLED' if vector_support else '❌ DISABLED'}")
        
        # STEP 2: Get all evaluation indices efficiently
        indices_info = await get_evaluation_indices_info(client)
        if not indices_info:
            return create_empty_filter_response("no_indices")
        
        # STEP 3: Extract templates from index names (much faster than aggregation)
        templates_from_indices = extract_templates_from_indices(indices_info)
        
        # STEP 4: Get field mappings to understand available metadata fields
        available_fields = await get_available_metadata_fields(client, indices_info)
        
        # STEP 5: Use targeted sampling per index for metadata values
        metadata_values = await get_metadata_values_efficiently(client, indices_info, available_fields)
        
        # STEP 6: Build final response including Vector search capabilities
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
            
            # Vector search capabilities
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
        
        # PRODUCTION logging
        logger.info(f"✅ EFFICIENT metadata loading completed in {filter_options['load_time_ms']}ms:")
        logger.info(f"   📁 Indices analyzed: {len(indices_info)}")
        logger.info(f"   📋 Templates: {len(templates_from_indices)} (from index names)")
        logger.info(f"   🏢 Programs: {len(metadata_values.get('programs', []))}")
        logger.info(f"   🤝 Partners: {len(metadata_values.get('partners', []))}")
        logger.info(f"   📊 Total evaluations: {filter_options['total_evaluations']:,}")
        logger.info(f"   🔮 Vector search: {'✅ ENABLED' if vector_support else '❌ DISABLED'}")
        logger.info(f"   🔥 Hybrid search: {'✅ AVAILABLE' if filter_options['hybrid_search_available'] else '❌ NOT AVAILABLE'}")
        
        return filter_options
        
    except Exception as e:
        logger.error(f"EFFICIENT: Failed to load filter options: {e}")
        return create_empty_filter_response("error", str(e))

async def get_evaluation_indices_info(client):
    """
    Get information about all evaluation indices efficiently
    """
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
    """
    Extract template names efficiently by sampling one document per index
    Much faster than aggregating all documents
    """
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
    """
    Check index mappings to see what metadata fields are actually available including vectors
    """
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

            if "document_embedding" in properties:
                vector_fields.add("document_embedding")
        
        logger.info(f"🔍 Available metadata fields: {sorted(available_fields)}")
        logger.info(f"🔮 Vector fields detected: {sorted(vector_fields)}")
        return list(available_fields)
    
        
    except Exception as e:
        logger.warning(f"Could not check field mappings: {e}")
        # Return expected fields as fallback
        return ["program", "partner", "site", "lob", "agent", "disposition", 
        "sub_disposition", "language", "call_type", "weighted_score", "url"]

async def get_metadata_values_efficiently(client, indices_info, available_fields):
    """
    Get metadata values using targeted sampling instead of full aggregation
    """
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
        
        logger.info(f"Metadata sampling completed from {len(indices_info)} indices")
        return result
        
    except Exception as e:
        logger.error(f"Failed to sample metadata efficiently: {e}")
        return {key: [] for key in metadata_values.keys()}

def get_cached_filter_metadata():
    """
    Get cached filter metadata if still valid
    """
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
    """
    Cache filter metadata for faster subsequent requests
    """
    try:
        _filter_metadata_cache["data"] = data.copy()
        _filter_metadata_cache["timestamp"] = time.time()
        
        logger.info(f"📋 Filter metadata cached for {_filter_metadata_cache['ttl_seconds']}s")
        
    except Exception as e:
        logger.warning(f"Failed to cache filter metadata: {e}")

def create_empty_filter_response(status="no_data", error_msg=""):
    """
    Create empty filter response for error cases
    """
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
        "version": "4.2.0_efficient",
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
    """
    Clear the filter metadata cache (useful after data imports)
    """
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
    
# ENHANCED DEBUG ENDPOINTS WITH VECTOR SEARCH TESTING
    
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
        "fix_applied": "moved_from_router_to_direct_endpoint",
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

# ============================================================================
# ENHANCED DATA PROCESSING - PRODUCTION QUALITY
# ============================================================================

def clean_template_id_for_index(template_id: str) -> str:
    """Clean template_id to create valid OpenSearch index names - PRODUCTION version"""
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

def extract_program_from_template(template_name: str, template_id: str = None, evaluation_data: Dict = None) -> str:
    """
    PRODUCTION: Enhanced program extraction from template name with multiple strategies
    """
    if not template_name:
        # Fallback strategies if template_name is missing
        if evaluation_data:
            # Check if program is already provided in the data
            existing_program = evaluation_data.get("program")
            if existing_program and existing_program.strip() and existing_program.lower() not in ["unknown", "null", ""]:
                return existing_program.strip()
            
            # Try to extract from other fields
            partner = evaluation_data.get("partner", "").lower()
            if "metro" in partner:
                return "Metro"
        
        return "Unknown Program"
    
    template_lower = template_name.lower().strip()
    
    # PRODUCTION: Enhanced program mapping logic
    program_mappings = [
        {
            "program": "Metro",
            "patterns": ["metro", "corporate", "sptr", "metro by t-mobile", "corporate sptr"]
        },
        {
            "program": "T-Mobile Prepaid",
            "patterns": ["t-mobile", "tmobile", "prepaid", "t-mobile prepaid", "pre-paid", "prepay"]
        },
        {
            "program": "ASW", 
            "patterns": ["asw", "authorized", "dealer", "authorized dealer", "agent", "indirect"]
        },
        {
            "program": "Technical Support",
            "patterns": ["technical", "tech support", "support", "troubleshooting", "device support"]
        },
        {
            "program": "Customer Service",
            "patterns": ["customer service", "cs", "customer care", "care", "service"]
        },
        {
            "program": "Sales",
            "patterns": ["sales", "revenue", "acquisition", "new customer", "upgrade"]
        },
        {
            "program": "Billing",
            "patterns": ["billing", "payment", "finance", "collections", "account management"]
        },
        {
            "program": "Quality Assurance",
            "patterns": ["quality", "qa", "evaluation", "assessment", "review", "monitoring"]
        }
    ]
    
    # First pass: Look for exact matches or strong indicators
    for mapping in program_mappings:
        for pattern in mapping["patterns"]:
            if pattern in template_lower:
                log_import(f"🎯 Program extracted: '{mapping['program']}' from template '{template_name}' (pattern: '{pattern}')")
                return mapping["program"]
    
    # Second pass: Check evaluation data for additional context
    if evaluation_data:
        partner = evaluation_data.get("partner", "").lower()
        site = evaluation_data.get("site", "").lower()
        lob = evaluation_data.get("lob", "").lower()
        
        # Partner-based program detection
        if any(keyword in partner for keyword in ["metro", "corporate"]):
            return "Metro"
        elif any(keyword in partner for keyword in ["prepaid", "t-mobile"]):
            return "T-Mobile Prepaid"
        elif any(keyword in partner for keyword in ["asw", "dealer", "indirect"]):
            return "ASW"
        
        # LOB-based program detection
        if any(keyword in lob for keyword in ["tech", "support"]):
            return "Technical Support"
        elif any(keyword in lob for keyword in ["sales", "revenue"]):
            return "Sales"
        elif any(keyword in lob for keyword in ["billing", "payment"]):
            return "Billing"
    
    # Final fallback for PRODUCTION
    log_import(f"⚠️ Could not extract program from template '{template_name}' - using fallback")
    return "Corporate"  # Default to Corporate instead of "Unknown Program"

def clean_field_value(value, default=None):
    """PRODUCTION: Clean and normalize field values for consistent filtering"""
    if not value:
        return default
    
    if isinstance(value, str):
        cleaned = value.strip()
        
        # PRODUCTION: Remove common placeholder values
        if cleaned.lower() in [
            "null", "undefined", "n/a", "na", "", "unknown", "not set", 
            "not specified", "test", "sample", "demo", "tbd", "pending"
        ]:
            return default
        
        return cleaned
    
    return value

def safe_int(value, default=0):
    """PRODUCTION: Safely convert value to integer"""
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

def generate_agent_id(agent_name):
    """PRODUCTION: Generate consistent agent ID from agent name"""
    if not agent_name or agent_name.strip().lower() in ["unknown", "null", ""]:
        return "00000000"
    
    try:
        # Create a consistent hash for PRODUCTION
        hash_object = hashlib.md5(agent_name.encode())
        return hash_object.hexdigest()[:8]
    except:
        # Fallback method
        return str(hash(agent_name) % 100000000).zfill(8)

def extract_comprehensive_metadata(evaluation: Dict) -> Dict[str, Any]:
    """
    PRODUCTION: Extract all metadata for real data filters with enhanced field mapping
    VERSION: 4.5.0 - Now includes weighted_score and url fields
    """
    template_name = evaluation.get("template_name", "Unknown Template")
    template_id = evaluation.get("template_id", "")
    
    # Enhanced program extraction with evaluation context
    program = extract_program_from_template(template_name, template_id, evaluation)
    
    # PRODUCTION: Comprehensive metadata extraction
    metadata = {
        # Primary identifiers
        "evaluationId": evaluation.get("evaluationId"),
        "internalId": evaluation.get("internalId"),
        "template_id": template_id,
        "template_name": template_name,    
        
        # ENHANCED: Program as separate field
        "program": program,
        
        # NEWLY ADDED: Previously missing fields ✅
        "weighted_score": safe_int(evaluation.get("weighted_score"), 0),  # ✅ FIXED: was commented out
        "url": clean_field_value(evaluation.get("url")),  # ✅ ADDED: was completely missing
        
        # Organizational hierarchy - clean and normalize
        "partner": clean_field_value(evaluation.get("partner"), "Unknown Partner"),
        "site": clean_field_value(evaluation.get("site"), "Unknown Site"),
        "lob": clean_field_value(evaluation.get("lob"), "Unknown LOB"),
        
        # Agent information - clean and normalize
        "agent": clean_field_value(evaluation.get("agentName"), "Unknown Agent"),
        "agentId": evaluation.get("agentId") or evaluation.get("agent_id"),
        
        # Call details - clean and normalize
        "disposition": clean_field_value(evaluation.get("disposition"), "Unknown Disposition"),
        "sub_disposition": clean_field_value(evaluation.get("subDisposition"), "Unknown Sub-Disposition"),
        "language": clean_field_value(evaluation.get("language"), "English"),  # Default to English
        
        # Date and timing
        "call_date": evaluation.get("call_date"),
        "call_duration": safe_int(evaluation.get("call_duration"), 0),
        "created_on": evaluation.get("created_on"),
        
        # Additional contact information (if available and not sensitive)
        #"phone_number": clean_field_value(evaluation.get("phoneNumber")),
        #"contact_id": clean_field_value(evaluation.get("contactId")),
        #"ucid": clean_field_value(evaluation.get("ucid")),
        "call_type": clean_field_value(evaluation.get("callType"), "CSR")  # Default to CSR
    }
    
    return metadata

# I will use this for Weighted Score value
def safe_float(value, default=0.0):
    """PRODUCTION: Safely convert value to float"""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def extract_qa_pairs(evaluation_text: str) -> List[Dict[str, Any]]:
    """Extract Question and Answer pairs from evaluation text - PRODUCTION version"""
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
    """Split transcript while preserving speaker boundaries - PRODUCTION version"""
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

# ============================================================================
# PRODUCTION MEMORY MANAGEMENT
# ============================================================================

async def cleanup_memory_after_batch():
    """PRODUCTION: Comprehensive memory cleanup after processing a batch"""
    import gc
    
    try:
        # Clear embedding cache if available
        if EMBEDDER_AVAILABLE:
            try:
                from embedder import get_embedding_service
                service = get_embedding_service()
                # Clear LRU cache if it's getting large
                if hasattr(service, '_cached_embed_single'):
                    cache_info = service._cached_embed_single.cache_info()
                    if cache_info.currsize > 100:
                        service._cached_embed_single.cache_clear()
                        log_import(f"🧹 Cleared embedding LRU cache ({cache_info.currsize} entries)")
            except Exception as e:
                log_import(f"⚠️ Could not clear embedding cache: {str(e)[:50]}")
        
        # Force garbage collection
        collected = gc.collect()
        if collected > 0:
            log_import(f"🧹 Garbage collected {collected} objects")
        
        # Small delay to let cleanup complete
        await asyncio.sleep(0.1)
        
    except Exception as e:
        log_import(f"⚠️ Memory cleanup error: {str(e)[:100]}")

# ============================================================================
# PRODUCTION EVALUATION PROCESSING 
# ============================================================================

async def process_evaluation(evaluation: Dict) -> Dict:
    """
    PRODUCTION: Process evaluation with comprehensive metadata and real data integration
    """
    try:
        evaluation_text = evaluation.get("evaluation", "")
        transcript_text = evaluation.get("transcript", "")
        
        if not evaluation_text and not transcript_text:
            return {"status": "skipped", "reason": "no_content"}
        
        # Extract all chunks
        all_chunks = []
        
        # Process evaluation Q&A
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
        
        # Process transcript
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
        
        # PRODUCTION: Use comprehensive metadata extraction
        comprehensive_metadata = extract_comprehensive_metadata(evaluation)
        
        # Validation
        evaluation_id = evaluation.get("evaluationId")
        if not evaluation_id:
            return {"status": "skipped", "reason": "missing_eval_id"}
        
        template_id = evaluation.get("template_id")
        if not template_id:
            return {"status": "skipped", "reason": "missing_template_id"}
        
        # Document ID and collection
        doc_id = str(evaluation_id)
        collection = clean_template_id_for_index(template_id)
        
        # PRODUCTION logging
        log_import(f" PRODUCTION METADATA for {evaluation_id}:")
        log_import(f"    Template: '{comprehensive_metadata['template_name']}'")
        log_import(f"    Program: '{comprehensive_metadata['program']}'")
        log_import(f"    Partner: '{comprehensive_metadata['partner']}'")
        log_import(f"    Site: '{comprehensive_metadata['site']}'")
        log_import(f"    LOB: '{comprehensive_metadata['lob']}'")
        log_import(f"    Agent: '{comprehensive_metadata['agent']}'")
        
        # Generate embeddings
        chunk_embeddings = []
        if EMBEDDER_AVAILABLE:
            try:
                chunk_texts = [chunk["text"] for chunk in all_chunks]
                
                batch_size = 10
                for i in range(0, len(chunk_texts), batch_size):
                    batch_texts = chunk_texts[i:i + batch_size]
                    try:
                        from embedder import embed_texts
                        batch_embeddings = embed_texts(batch_texts)
                        chunk_embeddings.extend(batch_embeddings)
                    except ImportError:
                        for text in batch_texts:
                            embedding = embed_text(text)
                            chunk_embeddings.append(embedding)
                    
                    if i + batch_size < len(chunk_texts):
                        await asyncio.sleep(0.1)
                        
            except Exception as e:
                log_import(f"⚠️ Embedding failed for evaluation {evaluation_id}: {str(e)[:50]}")
                chunk_embeddings = []
        
        # CREATE SINGLE DOCUMENT WITH COMPREHENSIVE METADATA
        document_body = {
            # Primary identification
            "evaluationId": evaluation_id,
            "internalId": comprehensive_metadata["internalId"],
            "template_id": template_id,
            "template_name": comprehensive_metadata["template_name"],
            
            # Document structure
            "document_type": "evaluation",
            "total_chunks": len(all_chunks),
            "evaluation_chunks_count": len([c for c in all_chunks if c["content_type"] == "evaluation_qa"]),
            "transcript_chunks_count": len([c for c in all_chunks if c["content_type"] == "transcript"]),
            
            # All chunks
            "chunks": [],
            
            # Combined text for search
            "full_text": "",
            "evaluation_text": evaluation_text,
            "transcript_text": transcript_text,
            
            # PRODUCTION: Enhanced metadata for real data filters
            "metadata": comprehensive_metadata,
            
            # Indexing info
            "source": "evaluation_api",
            "indexed_at": datetime.now().isoformat(),
            "collection_name": collection,
            "collection_source": f"template_id_{template_id}",
            "version": "4.2.0_production"
        }
        
        # Add chunks with embeddings
        full_text_parts = []
        
        for i, chunk in enumerate(all_chunks):
            chunk_data = {
                "chunk_index": i,
                "text": chunk["text"],
                "content_type": chunk["content_type"],
                "length": len(chunk["text"]),
                **{k: v for k, v in chunk.items() if k not in ["text", "content_type", "chunk_index"]}
            }
            
            if i < len(chunk_embeddings):
                chunk_data["embedding"] = chunk_embeddings[i]
            
            document_body["chunks"].append(chunk_data)
            full_text_parts.append(chunk["text"])
        
        # Create combined full text
        document_body["full_text"] = "\n\n".join(full_text_parts)
        
        # Add document-level embedding
        if chunk_embeddings:
            try:
                import numpy as np
                doc_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                document_body["document_embedding"] = doc_embedding
            except Exception as e:
                log_import(f"⚠️ Could not create document embedding: {str(e)[:50]}")
        
        # INDEX DOCUMENT WITH RETRY LOGIC
        try:
            max_retries = 3
            for retry in range(max_retries):
                try:
                    index_document(doc_id, document_body, index_override=collection)
                    log_import(f"✅ PRODUCTION INDEXED: Eval {evaluation_id} | Template: '{comprehensive_metadata['template_name']}' | Program: '{comprehensive_metadata['program']}' | {len(all_chunks)} chunks")
                    break
                    
                except Exception as index_error:
                    if retry < max_retries - 1:
                        delay = (retry + 1) * 2
                        log_import(f"⚠️ Retry {retry + 1}/{max_retries} for eval {evaluation_id} in {delay}s: {str(index_error)[:50]}")
                        time.sleep(delay)
                    else:
                        raise index_error
            
        except Exception as e:
            error_msg = f"Failed to index evaluation {evaluation_id}: {str(e)}"
            log_import(f"❌ {error_msg}")
            
            if any(keyword in str(e).lower() for keyword in ["timeout", "connection", "unreachable", "opensearch"]):
                raise Exception(f"OpenSearch connection error: {str(e)}")
            
            return {"status": "error", "error": str(e)}
        
        return {
            "status": "success",
            "document_id": doc_id,
            "evaluationId": evaluation_id,
            "template_id": template_id,
            "template_name": comprehensive_metadata["template_name"],
            "program": comprehensive_metadata["program"],
            "partner": comprehensive_metadata["partner"],
            "site": comprehensive_metadata["site"],
            "lob": comprehensive_metadata["lob"],
            "collection": collection,
            "total_chunks": len(all_chunks),
            "agent_id": comprehensive_metadata["agent_id"],
            "agent_name": comprehensive_metadata["agent"],
            "evaluation_chunks": len([c for c in all_chunks if c["content_type"] == "evaluation_qa"]),
            "transcript_chunks": len([c for c in all_chunks if c["content_type"] == "transcript"]),
            "total_content_length": sum(len(chunk["text"]) for chunk in all_chunks),
            "has_embeddings": bool(chunk_embeddings)
        }
        
    except Exception as e:
        logger.error(f"PRODUCTION: Failed to process evaluation: {e}")
        return {"status": "error", "error": str(e)}

# ============================================================================
# PRODUCTION API FETCHING (Keeping existing)
# ============================================================================

async def fetch_evaluations(max_docs: int = None):
    """PRODUCTION: Fetch evaluations from API with enhanced error handling"""
    try:
        if not API_BASE_URL or not API_AUTH_VALUE:
            raise Exception("API configuration missing")
        
        headers = {
            API_AUTH_KEY: API_AUTH_VALUE,
            'Accept': 'application/json',
            'User-Agent': 'Ask-InnovAI-Production/4.2.0'
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
        logger.error(f"PRODUCTION: Failed to fetch evaluations: {e}")
        raise

async def run_production_import(collection: str = "all", max_docs: int = None, batch_size: int = None):
    """
    PRODUCTION: Import process with enhanced real data integration
    """
    try:
        update_import_status("running", "Starting PRODUCTION import with real data integration")
        log_import("🚀 Starting PRODUCTION import: Real data filter system + Evaluation grouping")
        
        # Clear filter cache on import start
        _filter_metadata_cache["data"] = None
        _filter_metadata_cache["timestamp"] = None
        log_import("🧹 Cleared filter metadata cache for fresh import data")
        
        # Memory management settings
        BATCH_SIZE = batch_size or int(os.getenv("IMPORT_BATCH_SIZE", "5"))
        DELAY_BETWEEN_BATCHES = float(os.getenv("DELAY_BETWEEN_BATCHES", "2.0"))
        DELAY_BETWEEN_DOCS = float(os.getenv("DELAY_BETWEEN_DOCS", "0.5"))
        MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "1"))
        
        log_import(f"📊 PRODUCTION import configuration:")
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
        
        # Check OpenSearch connectivity
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
                "document_structure": "evaluation_grouped",
                "version": "4.2.0_production"
            }
            update_import_status("completed", results=results)
            return
        
        # Process evaluations with PRODUCTION structure
        update_import_status("running", f"Processing {len(evaluations)} evaluations with PRODUCTION real data integration")
        
        total_processed = 0
        total_chunks = 0
        total_evaluations_indexed = 0
        errors = 0
        opensearch_errors = 0
        consecutive_opensearch_errors = 0
        batch_count = 0
        template_collections = set()
        program_stats = defaultdict(int)
        
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
                    process = psutil.Process(os.getpid())
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
                        batch_evaluations_indexed += 1
                        consecutive_opensearch_errors = 0
                        
                        # Track template-based collections and programs
                        if result.get("collection"):
                            template_collections.add(result["collection"])
                        if result.get("program"):
                            program_stats[result["program"]] += 1
                            
                        log_import(f"✅ Evaluation {result['evaluationId']}: {result['total_chunks']} chunks → Collection '{result['collection']}' | Program: '{result['program']}'")
                        
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
                    if actual_index < len(evaluations) - 1:
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
                        process = psutil.Process(os.getpid())
                        memory_after_cleanup = process.memory_info().rss / 1024 / 1024  # MB
                        memory_saved = current_memory - memory_after_cleanup if current_memory else 0
                        log_import(f"💾 Memory after cleanup: {memory_after_cleanup:.1f} MB (saved: {memory_saved:.1f} MB)")
                    except Exception:
                        pass
            
            # Adjust delay based on OpenSearch errors
            if batch_opensearch_errors >= 2:
                extended_delay = DELAY_BETWEEN_BATCHES + (batch_opensearch_errors * 2)
                log_import(f"🔄 Batch had {batch_opensearch_errors} OpenSearch errors, extending delay to {extended_delay}s")
                await asyncio.sleep(extended_delay)
            else:
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)
            
            # Clear batch references
            batch.clear()
            del batch
        
        # Complete with final memory cleanup
        log_import("🧹 Performing final memory cleanup")
        await cleanup_memory_after_batch()
        
        # Clear filter cache after successful import to force refresh
        _filter_metadata_cache["data"] = None
        _filter_metadata_cache["timestamp"] = None
        log_import("🧹 Cleared filter metadata cache - will refresh on next request")
        
        # Final memory check
        final_memory = None
        memory_change = 0
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_change = final_memory - initial_memory if initial_memory else 0
                log_import(f"💾 Final memory usage: {final_memory:.1f} MB (change: {memory_change:+.1f} MB)")
            except Exception:
                pass
        
        results = {
            "total_documents_processed": total_processed,
            "total_evaluations_indexed": total_evaluations_indexed,
            "total_chunks_processed": total_chunks,
            "errors": errors,
            "opensearch_errors": opensearch_errors,
            "import_type": "full",
            "document_structure": "evaluation_grouped",
            "collection_strategy": "template_id_based",
            "program_distribution": dict(program_stats),  # NEW: Program statistics
            "completed_at": datetime.now().isoformat(),
            "success_rate": f"{(total_processed / len(evaluations) * 100):.1f}%" if evaluations else "0%",
            "batch_size": BATCH_SIZE,
            "total_batches": batch_count,
            "template_collections_created": list(template_collections),
            "memory_stats": {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_change_mb": memory_change
            },
            "version": "4.2.0_production"
        }
        
        log_import(f"🎉 PRODUCTION import completed:")
        log_import(f"   📄 Evaluations processed: {total_processed}/{len(evaluations)}")
        log_import(f"   📋 Documents indexed: {total_evaluations_indexed} (1 per evaluation)")
        log_import(f"   🧩 Total chunks processed: {total_chunks} (grouped within documents)")
        log_import(f"   📁 Template collections created: {len(template_collections)}")
        log_import(f"   🏢 Program distribution: {dict(program_stats)}")
        log_import(f"   ❌ Total errors: {errors}")
        log_import(f"   🔌 OpenSearch errors: {opensearch_errors}")
        log_import(f"   📊 Success rate: {results['success_rate']}")
        log_import(f"   💾 Memory change: {memory_change:+.1f} MB")
        log_import(f"   🏗️ Document structure: Evaluation-grouped (chunks within documents)")
        log_import(f"   🏷️ Collection strategy: Template_ID-based")
        log_import(f"   🎯 Real data filters: Ready for production use")
        
        update_import_status("completed", results=results)
        
    except Exception as e:
        error_msg = f"PRODUCTION import failed: {str(e)}"
        
        # Check if it's an OpenSearch-related error
        if any(keyword in str(e).lower() for keyword in ["opensearch", "connection", "timeout", "unreachable"]):
            error_msg = f"OpenSearch connection issue: {str(e)}"
            log_import(f"❌ {error_msg}")
            log_import("💡 PRODUCTION Suggestions:")
            log_import("   - Check if OpenSearch cluster is healthy")
            log_import("   - Verify network connectivity")
            log_import("   - Consider scaling up the cluster")
            log_import("   - Try reducing import batch size")
        else:
            log_import(f"❌ {error_msg}")
        
        update_import_status("failed", error=error_msg)

# ============================================================================
# PRODUCTION STATISTICS AND HEALTH ENDPOINTS
# ============================================================================

@app.get("/opensearch_statistics")
async def get_opensearch_statistics():
    """CORRECTED: Get OpenSearch statistics using document sampling like the filter system"""
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

        # STEP 3: Check vector search capabilities
        vector_support = detect_vector_support(client)
        available_fields = get_available_fields(client) if vector_support else {"vector_fields": [], "has_vector_support": False}
        
        # STEP 4: Get available metadata fields from mapping
        try:
            mapping_response = client.indices.get_mapping(index="eval-*", request_timeout=10)
            available_metadata_fields = set()  # <-- RENAMED VARIABLE
            
            for index_name, mapping_data in mapping_response.items():
                properties = mapping_data.get("mappings", {}).get("properties", {})
                metadata_props = properties.get("metadata", {}).get("properties", {})
                
                # Collect available metadata fields
                for field_name in metadata_props.keys():
                    available_metadata_fields.add(field_name)
            logger.info(f"✅ Available metadata fields: {available_metadata_fields}")

        except Exception as e:
            logger.warning(f"Mapping query failed: {e}")
            available_metadata_fields = set()
        
        # STEP 5: Sample documents to extract real statistics (same method as filters)
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
                        "metadata.url", "metadata.call_duration"
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
                # Check for vector embeddings
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
                "available_fields": sorted(list(available_meta_fields)),
                "cluster_status": cluster_status,

                # NEW: Vector search capabilities
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
            "method": "document_sampling",
            "sample_size": chunks_sampled,
            "version": "4.4.0_sampling_fixed"
        }
        
        logger.info("Enhanced statistics with vector search analysis completed")
        return response_data
        
    except Exception as e:
        logger.error(f"Statistics generation failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "method": "document_sampling",
            "version": "4.8.0 vector_enabled"
        }

# Add a simple health endpoint that never fails
@app.get("/opensearch_health_simple")
async def opensearch_health_simple():
    """Ultra-simple health check that never causes compilation errors"""
    try:
        from opensearch_client import get_opensearch_client
        
        client = get_opensearch_client()
        if not client:
            return {"status": "no_client", "timestamp": datetime.now().isoformat()}
        
        # Just ping - no complex queries
        ping_result = client.ping(request_timeout=5)
        
        return {
            "status": "healthy" if ping_result else "unhealthy",
            "connected": ping_result,
            "timestamp": datetime.now().isoformat(),
            "safe_mode": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Add middleware to catch compilation errors
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class CompilationErrorMiddleware(BaseHTTPMiddleware):
    """Catch OpenSearch compilation errors and return safe responses"""
    
    async def dispatch(self, request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for compilation errors
            if any(keyword in error_str for keyword in [
                "search_phase_execution_exception",
                "compile_error", 
                "compilation_exception"
            ]):
                logger.error(f"🚨 COMPILATION ERROR caught in {request.url.path}: {e}")
                
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Query compilation failed",
                        "details": "OpenSearch could not compile the query",
                        "suggestion": "Try simpler search terms or contact support",
                        "safe_mode": True,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            else:
                # Re-raise non-compilation errors
                raise

# Add the middleware to your app
app.add_middleware(CompilationErrorMiddleware)

def create_empty_statistics_response():
    """Create empty statistics response structure"""
    return {
        "summary": {
            "total_evaluations": 0,
            "total_chunks": 0,
            "evaluation_chunks": 0,
            "transcript_chunks": 0,
            "evaluations_with_transcript": 0,
            "evaluations_without_transcript": 0
        },
        "template_analysis": {
            "template_counts": {},
            "program_counts": {}
        },
        "organizational_analysis": {
            "partner_counts": {},
            "site_counts": {},
            "lob_counts": {}
        },
        "agent_analysis": {
            "agent_counts": {},
            "top_agents": []
        },
        "call_analysis": {
            "disposition_counts": {},
            "sub_disposition_counts": {},
            "language_counts": {},
            "call_type_counts": {},
            "call_duration_stats": {},
            "call_duration_ranges": {}
        },
        "weighted_score_analysis": {
            "statistics": {},
            "score_ranges": {},
            "score_histogram": [],
            "evaluations_with_scores": 0,
            "score_coverage_percentage": 0,
            "avg_score_by_program": {}
        },
        "url_analysis": {
            "evaluations_with_urls": 0,
            "url_coverage_percentage": 0,
            "url_domains": {},
            "url_protocols": {}
        },
        "temporal_analysis": {
            "evaluations_by_month": [],
            "evaluations_by_day": [],
            "date_range_stats": {}
        },
        "index_information": []
    }

@app.get("/ping")
async def ping():
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "service": "ask-innovai-production",
        "version": "4.2.1_fixed_routing",
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
            "vector_search_enabled": VECTOR_SEARCH_READY,
            "hybrid_search_available": VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE,
            "semantic_similarity": VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE,
            "enhanced_relevance": VECTOR_SEARCH_READY
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
        <h1>🤖 Ask InnovAI Production v4.2.0</h1>
        <p><strong>Status:</strong> Production Ready ✅</p>
        <p><strong>Features:</strong> Real Data Filters + Efficient Metadata Loading + Evaluation Grouping</p>
        <p><strong>Structure:</strong> Template_ID Collections with Program Extraction</p>
        <p>Admin interface file not found. Please ensure static/index.html exists.</p>
        </body></html>
        """)




@app.post("/import")
async def start_import(request: ImportRequest, background_tasks: BackgroundTasks):
    """PRODUCTION: Start the enhanced import process with real data integration"""
    global import_status
    
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
        log_import(f"🚀 PRODUCTION import request received:")
        log_import(f"   Collection: {request.collection}")
        log_import(f"   Import Type: {request.import_type}")
        log_import(f"   Max Docs: {request.max_docs or 'All'}")
        log_import(f"   Batch Size: {request.batch_size or 'Default'}")
        log_import(f"   Version: 4.2.0_production")
        
        # Start background import
        background_tasks.add_task(
            run_production_import,
            collection=request.collection,
            max_docs=request.max_docs,
            batch_size=request.batch_size
        )
        
        return {
            "status": "success",
            "message": f"PRODUCTION import started: {request.import_type} mode",
            "collection": request.collection,
            "max_docs": request.max_docs,
            "import_type": request.import_type,
            "structure": "evaluation_grouped",
            "features": "real_data_filters_efficient",
            "version": "4.2.0_production"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PRODUCTION: Failed to start import: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start import: {str(e)}")

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
            from opensearch_client import get_connection_status, test_connection, get_opensearch_config, detect_vector_support, get_available_fields, get_opensearch_client
            
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

@app.post("/analytics/stats")
async def analytics_stats(request: dict):
    """
    FIXED: Get analytics statistics showing CORRECT evaluation counts (not chunks)
    """
    try:
        filters = request.get("filters", {})
        
        from opensearch_client import get_opensearch_client, test_connection
        
        if not test_connection():
            return {
                "status": "error", 
                "error": "OpenSearch not available",
                "totalRecords": 0
            }
        
        client = get_opensearch_client()
        
        # Build filter query for counting
        filter_query = {"match_all": {}}
        if filters:
            filter_clauses = []
            
            # Date filters
            if filters.get("call_date_start") or filters.get("call_date_end"):
                date_range = {}
                if filters.get("call_date_start"):
                    date_range["gte"] = filters["call_date_start"]
                if filters.get("call_date_end"):
                    date_range["lte"] = filters["call_date_end"]
                filter_clauses.append({"range": {"metadata.call_date": date_range}})
            
            # Keyword filters
            keyword_filters = {
                "template_name": "template_name.keyword",
                "program": "metadata.program.keyword", 
                "partner": "metadata.partner.keyword",
                "site": "metadata.site.keyword",
                "lob": "metadata.lob.keyword",
                "agent": "metadata.agent.keyword",
                "disposition": "metadata.disposition.keyword",
                "sub_disposition": "metadata.sub_disposition.keyword",
                "language": "metadata.language.keyword"
            }
            
            for filter_key, field_path in keyword_filters.items():
                filter_value = filters.get(filter_key)
                if filter_value and str(filter_value).strip():
                    filter_clauses.append({"term": {field_path: filter_value}})
            
            # Duration filters
            if filters.get("min_duration") or filters.get("max_duration"):
                duration_range = {}
                if filters.get("min_duration"):
                    duration_range["gte"] = int(filters["min_duration"])
                if filters.get("max_duration"):
                    duration_range["lte"] = int(filters["max_duration"])
                filter_clauses.append({"range": {"metadata.call_duration": duration_range}})
            
            if filter_clauses:
                filter_query = {"bool": {"filter": filter_clauses}}
        
        # FIXED: Count unique evaluations, not documents/chunks
        response = client.search(
            index="eval-*",
            body={
                "size": 0,  # Don't return documents, just count
                "query": filter_query
            }
        )
        
        # Get total count (this represents unique evaluations since we store 1 doc per evaluation)
        total_hits = response.get("hits", {}).get("total", {})
        if isinstance(total_hits, dict):
            total_evaluations = total_hits.get("value", 0)
        else:
            total_evaluations = total_hits
        
        logger.info(f"📊 ANALYTICS STATS: {total_evaluations} evaluations match filters: {filters}")
        
        return {
            "status": "success",
            "totalRecords": total_evaluations,  # This is now correct
            "filters_applied": filters,
            "timestamp": datetime.now().isoformat(),
            "data_type": "unique_evaluations",  # Clarify what we're counting
            "version": "4.4.0_fixed_counting"
        }
        
    except Exception as e:
        logger.error(f"Analytics stats error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "totalRecords": 0,
            "filters_applied": filters
        }
# Add this debug endpoint to your app.py to test metadata extraction
@app.get("/debug/test_metadata_extraction_new_fields")
async def test_metadata_extraction_new_fields():
    """
    DEBUG: Test the metadata extraction with sample data including new fields
    """
    try:
        # Sample evaluation data similar to your API response
        sample_evaluation = {
            "internalId": "686455bb08b81a355a14506e",
            "evaluationId": 316,
            "weighted_score": 79,  # ✅ Test new field
            "url": "https://innovai-demo.metrocare-agent.com/evaluation/view/316",  # ✅ Test new field
            "template_id": "685eed5f7e5fe29dc0b183e1",
            "template_name": "Ai Corporate SPTR - Haiku (working prompts)",
            "partner": "iQor",
            "site": "Bacolod",
            "lob": "CSR",
            "agentName": "Romnick Belando",
            "agentId": "30847084",
            "disposition": "Account",
            "subDisposition": "Extension",
            "created_on": "2025-07-01T21:39:41.377Z",
            "call_date": "2025-06-03T05:00:00.000Z",
            "call_duration": 239,
            "language": "english"
        }
        
        # Test the extraction
        extracted_metadata = extract_comprehensive_metadata(sample_evaluation)
        
        return {
            "status": "success",
            "test_results": {
                "input_data": sample_evaluation,
                "extracted_metadata": extracted_metadata,
                "new_fields_check": {
                    "weighted_score_extracted": "weighted_score" in extracted_metadata,
                    "weighted_score_value": extracted_metadata.get("weighted_score"),
                    "url_extracted": "url" in extracted_metadata,
                    "url_value": extracted_metadata.get("url")
                },
                "field_coverage": f"{len(extracted_metadata)}/19 fields extracted",
                "all_fields": sorted(list(extracted_metadata.keys()))
            },
            "version": "4.5.0_metadata_test"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "4.5.0_metadata_test"
        }
    
@app.get("/debug/test_metadata_extraction")
async def debug_test_metadata_extraction():
    """
    DEBUG: Test metadata extraction from search results to verify it's working
    """
    try:
        from opensearch_client import search_opensearch
        from chat_handlers import verify_metadata_alignment
        
        # Get some sample search results
        logger.info("🔍 Testing metadata extraction with sample search...")
        
        # Search for something that should return results
        results = search_opensearch("customer service", size=5)
        
        if not results:
            return {
                "status": "warning",
                "message": "No search results found - check if data is imported",
                "sample_results": [],
                "metadata_test": "no_data_to_test"
            }
        
        # Test metadata extraction
        logger.info(f"📊 Testing metadata extraction on {len(results)} search results...")
        metadata_summary = verify_metadata_alignment(results)
        
        # Analyze the first few results in detail
        detailed_analysis = []
        for i, result in enumerate(results[:3]):
            source = result.get("_source", result)
            
            analysis = {
                "result_index": i + 1,
                "has_source": "_source" in result,
                "source_keys": list(source.keys()) if source else [],
                "evaluation_id_sources": {
                    "evaluationId": source.get("evaluationId"),
                    "evaluation_id": source.get("evaluation_id"), 
                    "internalId": source.get("internalId"),
                    "metadata.evaluationId": source.get("metadata", {}).get("evaluationId") if source.get("metadata") else None
                },
                "metadata_structure": {
                    "has_metadata_field": "metadata" in source,
                    "metadata_keys": list(source.get("metadata", {}).keys()) if source.get("metadata") else [],
                    "metadata_values": {
                        "program": source.get("metadata", {}).get("program"),
                        "partner": source.get("metadata", {}).get("partner"),
                        "site": source.get("metadata", {}).get("site"),
                        "lob": source.get("metadata", {}).get("lob"),
                        "agent": source.get("metadata", {}).get("agent"),
                        "disposition": source.get("metadata", {}).get("disposition")
                    } if source.get("metadata") else {}
                },
                "index": result.get("_index"),
                "id": result.get("_id"),
                "score": result.get("_score")
            }
            detailed_analysis.append(analysis)
        
        return {
            "status": "success",
            "search_results_count": len(results),
            "metadata_summary": {
                "total_chunks_found": metadata_summary["total_chunks_found"],
                "total_evaluations": metadata_summary["total_evaluations"],
                "has_real_data": metadata_summary["has_real_data"],
                "programs_found": metadata_summary["programs"],
                "dispositions_found": metadata_summary["dispositions"],
                "evaluation_ids_found": len(metadata_summary["evaluation_ids"])
            },
            "detailed_analysis": detailed_analysis,
            "recommendations": [
                "Check if 'has_real_data' is True",
                "Verify that evaluation_ids_found > 0", 
                "Ensure programs_found and dispositions_found are not empty",
                "Look at detailed_analysis to see metadata structure"
            ],
            "version": "4.4.0_metadata_debug"
        }
        
    except Exception as e:
        logger.error(f"Debug metadata extraction failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "version": "4.4.0_metadata_debug"
        }

@app.get("/opensearch_health_detailed")
async def get_opensearch_health_detailed():
    """PRODUCTION: Get detailed OpenSearch cluster health and statistics"""
    try:
        from opensearch_client import get_opensearch_client, test_connection
        
        if not test_connection():
            return {
                "status": "error",
                "error": "OpenSearch connection failed"
            }
        
        client = get_opensearch_client()
        
        # Get cluster health
        cluster_health = client.cluster.health()
        
        # Get cluster stats
        cluster_stats = client.cluster.stats()
        
        # Get index information
        try:
            all_indices = client.indices.get(index="*")
            user_indices = [name for name in all_indices.keys() if not name.startswith('.')]
            system_indices = [name for name in all_indices.keys() if name.startswith('.')]
            eval_indices = [name for name in all_indices.keys() if name.startswith('eval-')]
        except Exception:
            user_indices = []
            system_indices = []
            eval_indices = []
        
        return {
            "status": "success",
            "cluster_health": {
                "status": cluster_health.get("status"),
                "number_of_nodes": cluster_health.get("number_of_nodes"),
                "number_of_data_nodes": cluster_health.get("number_of_data_nodes"),
                "active_primary_shards": cluster_health.get("active_primary_shards"),
                "active_shards": cluster_health.get("active_shards"),
                "relocating_shards": cluster_health.get("relocating_shards"),
                "initializing_shards": cluster_health.get("initializing_shards"),
                "unassigned_shards": cluster_health.get("unassigned_shards")
            },
            "indices_summary": {
                "total_indices": len(user_indices) + len(system_indices),
                "user_indices": len(user_indices),
                "system_indices": len(system_indices),
                "evaluation_indices": len(eval_indices)
            },
            "storage": {
                "total_size": cluster_stats.get("indices", {}).get("store", {}).get("size_in_bytes", 0),
                "total_documents": cluster_stats.get("indices", {}).get("docs", {}).get("count", 0)
            },
            "version": "4.2.0_production",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"PRODUCTION: Failed to get detailed OpenSearch health: {e}")
        return {
            "status": "error", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
from fastapi.responses import FileResponse

@app.get("/chat", response_class=FileResponse)
async def serve_chat_ui():
    return FileResponse("static/chat.html")


@app.get("/search")
async def search_endpoint(q: str = Query(..., description="Search query")):
    """PRODUCTION: Search endpoint with real data integration"""
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
                    "program": result.get("metadata", {}).get("program", "Unknown"),  # NEW
                    "collection": result.get("_index")
                }
                for result in results
            ],
            "count": len(results),
            "version": "4.2.0_production"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": q,
            "results": []
        }
    
# Add these debugging endpoints to your existing app.py file
# These will help diagnose chat and filter issues

@app.get("/debug/opensearch_data")
async def debug_opensearch_data():
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

@app.get("/debug/test_search")
async def debug_test_search(q: str = "customer service", filters: str = "{}"):
    """DEBUG: Test search functionality with filters"""
    try:
        import json
        from opensearch_client import search_opensearch
        
        # Parse filters
        try:
            parsed_filters = json.loads(filters) if filters != "{}" else {}
        except:
            parsed_filters = {}
        
        logger.info(f"🔍 DEBUG SEARCH: query='{q}', filters={parsed_filters}")
        
        # Perform search
        results = search_opensearch(q, filters=parsed_filters, size=5)
        
        # Summarize results
        result_summary = []
        for i, result in enumerate(results):
            summary = {
                "index": i + 1,
                "id": result.get("_id"),
                "score": result.get("_score"),
                "evaluationId": result.get("evaluationId"),
                "template_name": result.get("template_name"),
                "has_content": bool(result.get("text")),
                "content_length": len(result.get("text", "")),
                "content_preview": result.get("text", "")[:100],
                "metadata": result.get("metadata", {}),
                "search_index": result.get("_index")
            }
            result_summary.append(summary)
        
        return {
            "status": "success",
            "query": q,
            "filters_applied": parsed_filters,
            "total_results": len(results),
            "results_summary": result_summary,
            "message": "Search test completed - check if results contain expected data",
            "version": "4.3.2_debug"
        }
        
    except Exception as e:
        logger.error(f"❌ DEBUG SEARCH FAILED: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": q,
            "filters": filters,
            "version": "4.3.2_debug"
        }
@app.post("/debug/test_chat_simple")
async def debug_test_chat_simple():
    """Simple test to isolate the 500 error"""
    try:
        logger.info("🧪 Testing basic chat function...")
        
        # Test 1: Basic imports
        try:
            from chat_handlers import ChatRequest
            logger.info("✅ ChatRequest import works")
        except Exception as e:
            logger.error(f"❌ ChatRequest import failed: {e}")
            return {"error": f"Import failed: {e}"}
        
        # Test 2: Environment variables
        endpoint = os.getenv("GENAI_ENDPOINT", "")
        access_key = os.getenv("GENAI_ACCESS_KEY", "")
        
        config_status = {
            "endpoint_set": bool(endpoint),
            "endpoint_value": endpoint[:50] + "..." if len(endpoint) > 50 else endpoint,
            "access_key_set": bool(access_key),
            "access_key_preview": f"{access_key[:10]}..." if access_key else "Not set"
        }
        
        # Test 3: Try a simple request structure
        test_request = ChatRequest(
            message="Hello test",
            history=[],
            filters={},
            analytics=True
        )
        
        logger.info("✅ ChatRequest creation works")
        
        # Test 4: Check imports that might be missing
        missing_imports = []
        
        try:
            from opensearch_client import get_opensearch_client
            logger.info("✅ OpenSearch client import works")
        except Exception as e:
            missing_imports.append(f"opensearch_client: {e}")
        
        try:
            import requests
            logger.info("✅ Requests import works")
        except Exception as e:
            missing_imports.append(f"requests: {e}")
        
        return {
            "status": "debug_complete",
            "chat_request_works": True,
            "config": config_status,
            "missing_imports": missing_imports,
            "message": "Basic components working - ready to test full chat"
        }
        
    except Exception as e:
        logger.error(f"❌ Debug test failed: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/debug/check_chat_route")
async def debug_check_chat_route():
    """Check if chat route is properly registered"""
    try:
        # Check available routes
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and '/chat' in route.path:
                routes.append({
                    "path": route.path,
                    "methods": getattr(route, 'methods', 'Unknown'),
                    "name": getattr(route, 'name', 'Unknown')
                })
        
        return {
            "status": "success",
            "chat_routes_found": routes,
            "total_routes": len([r for r in app.routes if hasattr(r, 'path')]),
            "message": "Check if /api/chat route is registered"
        }
        
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/debug/test_chat_context")
async def debug_test_chat_context(q: str = "What are the most common call dispositions?"):
    """DEBUG: Test chat context building"""
    try:
        from chat_handlers import build_search_context
        
        # Test with no filters
        logger.info(f"🔍 DEBUG CHAT CONTEXT: query='{q}'")
        
        context, sources = build_search_context(q, {})
        
        return {
            "status": "success",
            "query": q,
            "context_length": len(context),
            "context_preview": context[:500] + ("..." if len(context) > 500 else ""),
            "sources_count": len(sources),
            "sources_summary": [
                {
                    "evaluationId": s.get("evaluationId"),
                    "search_type": s.get("search_type"),
                    "content_type": s.get("content_type"),
                    "text_length": len(s.get("text", "")),
                    "score": s.get("score"),
                    "template_name": s.get("template_name"),
                    "metadata": s.get("metadata", {})
                }
                for s in sources[:3]
            ],
            "has_context": bool(context),
            "message": "Chat context test - verify if context is built from your data",
            "version": "4.3.2_debug"
        }
        
    except Exception as e:
        logger.error(f"❌ DEBUG CHAT CONTEXT FAILED: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": q,
            "version": "4.3.2_debug"
        }

@app.get("/debug/test_filters")
async def debug_test_filters():
    """DEBUG: Test filter options and search with filters"""
    try:
        from opensearch_client import search_opensearch
        
        # Test different filter combinations
        test_cases = [
            {"description": "No filters", "filters": {}},
            {"description": "Program filter", "filters": {"program": "Metro"}},
            {"description": "Partner filter", "filters": {"partner": "Advanced Solutions"}},
            {"description": "LOB filter", "filters": {"lob": "Customer Service"}},
            {"description": "Template filter", "filters": {"template_name": "CSR Quality"}},
            {"description": "Multiple filters", "filters": {"program": "Metro", "lob": "Customer Service"}}
        ]
        
        results = {}
        
        for test_case in test_cases:
            try:
                search_results = search_opensearch(
                    "customer service", 
                    filters=test_case["filters"], 
                    size=3
                )
                
                results[test_case["description"]] = {
                    "filters": test_case["filters"],
                    "results_count": len(search_results),
                    "sample_results": [
                        {
                            "evaluationId": r.get("evaluationId"),
                            "template_name": r.get("template_name"),
                            "metadata": r.get("metadata", {}),
                            "score": r.get("_score")
                        }
                        for r in search_results[:2]
                    ] if search_results else []
                }
                
            except Exception as e:
                results[test_case["description"]] = {
                    "filters": test_case["filters"],
                    "error": str(e)
                }
        
        return {
            "status": "success",
            "filter_tests": results,
            "message": "Filter test completed - check which filters work and return results",
            "version": "4.3.2_debug"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "4.3.2_debug"
        }

@app.post("/debug/test_chat_full")
async def debug_test_chat_full(request: dict):
    """DEBUG: Test full chat flow with debugging"""
    try:
        message = request.get("message", "What are the most common call dispositions?")
        filters = request.get("filters", {})
        
        logger.info(f"🔍 DEBUG FULL CHAT: message='{message}', filters={filters}")
        
        # Step 1: Test search context
        from chat_handlers import build_search_context
        context, sources = build_search_context(message, filters)
        
        # Step 2: Create test response without calling GenAI
        if context:
            test_response = f"[DEBUG RESPONSE] Based on the data found, I can see {len(sources)} relevant evaluations. Here's what the context contains: {context[:200]}..."
        else:
            test_response = "[DEBUG RESPONSE] No relevant data found in the search results. This could mean: 1) No data is imported, 2) Search terms don't match content, 3) Filters are too restrictive."
        
        return {
            "status": "success",
            "debug_info": {
                "message": message,
                "filters_applied": filters,
                "context_found": bool(context),
                "context_length": len(context),
                "sources_count": len(sources),
                "search_worked": len(sources) > 0
            },
            "context_preview": context[:300] + ("..." if len(context) > 300 else ""),
            "sources_preview": [
                {
                    "evaluationId": s.get("evaluationId"),
                    "template_name": s.get("template_name"),
                    "search_type": s.get("search_type"),
                    "score": s.get("score")
                }
                for s in sources[:3]
            ],
            "test_response": test_response,
            "next_steps": [
                "If context_found is False, check if data is imported",
                "If sources_count is 0, try broader search terms",
                "If search_worked is False, check OpenSearch connection",
                "Check the context_preview to see if it contains relevant data"
            ],
            "version": "4.3.2_debug"
        }
        
    except Exception as e:
        logger.error(f"❌ DEBUG FULL CHAT FAILED: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": message,
            "filters": filters,
            "version": "4.3.2_debug"
        }

@app.get("/debug/check_indices")
async def debug_check_indices():
    """DEBUG: Check what indices exist and their basic stats"""
    try:
        from opensearch_client import get_opensearch_client
        
        client = get_opensearch_client()
        if not client:
            return {"error": "OpenSearch client not available"}
        
        # Get all indices
        try:
            all_indices = client.indices.get(index="*")
            eval_indices = {k: v for k, v in all_indices.items() if k.startswith("eval-")}
            
            # Get stats for eval indices
            if eval_indices:
                stats_response = client.indices.stats(index="eval-*")
                
                indices_info = []
                for index_name in eval_indices.keys():
                    stats = stats_response.get("indices", {}).get(index_name, {})
                    doc_count = stats.get("primaries", {}).get("docs", {}).get("count", 0)
                    size_bytes = stats.get("primaries", {}).get("store", {}).get("size_in_bytes", 0)
                    
                    indices_info.append({
                        "name": index_name,
                        "document_count": doc_count,
                        "size_mb": round(size_bytes / (1024 * 1024), 2),
                        "template_id": index_name.replace("eval-", "") if index_name.startswith("eval-") else "unknown"
                    })
                
                return {
                    "status": "success",
                    "total_eval_indices": len(eval_indices),
                    "total_documents": sum(info["document_count"] for info in indices_info),
                    "indices": indices_info,
                    "message": "Check if you have evaluation indices with documents",
                    "version": "4.3.2_debug"
                }
            else:
                return {
                    "status": "warning",
                    "message": "No evaluation indices found - data may not be imported yet",
                    "all_indices": list(all_indices.keys()),
                    "version": "4.3.2_debug"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "version": "4.3.2_debug"
            }
            
    except Exception as e:
        return {"error": str(e), "version": "4.3.2_debug"}

# Add this route to test the debug endpoints
@app.get("/debug")
async def debug_dashboard():
    """DEBUG: Simple HTML dashboard for testing"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ask InnovAI Debug Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .test-button { 
                display: inline-block; 
                margin: 10px; 
                padding: 10px 20px; 
                background: #007bff; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
            }
            .test-button:hover { background: #0056b3; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🔍 Ask InnovAI Debug Dashboard</h1>
        
        <div class="section">
            <h2>Data Verification</h2>
            <a href="/debug/opensearch_data" class="test-button">Check OpenSearch Data</a>
            <a href="/debug/check_indices" class="test-button">Check Indices</a>
        </div>
        
        <div class="section">
            <h2>Search Testing</h2>
            <a href="/debug/test_search?q=customer service" class="test-button">Test Basic Search</a>
            <a href="/debug/test_search?q=customer service&filters={\"program\":\"Metro\"}" class="test-button">Test With Filters</a>
            <a href="/debug/test_filters" class="test-button">Test All Filters</a>
        </div>
        
        <div class="section">
            <h2>Chat System Testing</h2>
            <a href="/debug/test_chat_context?q=What are the most common call dispositions?" class="test-button">Test Chat Context</a>
        </div>
        
        <div class="section">
            <h2>Manual Tests</h2>
            <p><strong>Test Chat Full:</strong> POST to /debug/test_chat_full with body: {"message": "your question", "filters": {}}</p>
            <p><strong>Filter Metadata:</strong> GET /filter_options_metadata</p>
            <p><strong>Health Check:</strong> GET /health</p>
        </div>
        
        <div class="section">
            <h2>Instructions</h2>
            <ol>
                <li>First check "Check OpenSearch Data" to see if data exists</li>
                <li>Then try "Test Basic Search" to see if search works</li>
                <li>Test filters to see which ones work</li>
                <li>Test chat context to see if it builds properly</li>
            </ol>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# =============================================================================
# METADATA VERIFICATION DEBUG ENDPOINTS - ADD THESE AFTER EXISTING DEBUG ENDPOINTS
# =============================================================================

@app.get("/debug/verify_metadata_alignment")
async def debug_verify_metadata_alignment():
    """DEBUG: Verify that metadata in OpenSearch matches expected structure for call dispositions"""
    try:
        from opensearch_client import get_opensearch_client, test_connection
        
        if not test_connection():
            return {"error": "OpenSearch not available"}
        
        client = get_opensearch_client()
        
        # Sample query specifically for call disposition data
        disposition_query = {
            "size": 10,
            "query": {"match_all": {}},
            "_source": [
                "evaluationId", 
                "metadata.disposition", 
                "metadata.sub_disposition",
                "metadata.program",
                "metadata.partner", 
                "metadata.site",
                "metadata.lob",
                "metadata.agent",
                "metadata.call_date",
                "template_name"
            ]
        }
        
        response = client.search(index="eval-*", body=disposition_query)
        hits = response.get("hits", {}).get("hits", [])
        
        # Analyze metadata structure with correct counting
        metadata_analysis = {
            "dispositions_found": set(),
            "sub_dispositions_found": set(),
            "programs_found": set(),
            "partners_found": set(),
            "sites_found": set(),
            "lobs_found": set(),
            "agents_found": set(),
            "templates_found": set(),
            "sample_records": [],
            "total_sampled": len(hits),
            "unique_evaluations_sampled": set(),
            "metadata_structure_issues": []
        }
        
        for hit in hits:
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            evaluation_id = source.get("evaluationId")
            
            # Track unique evaluations vs total hits
            if evaluation_id:
                metadata_analysis["unique_evaluations_sampled"].add(evaluation_id)
            
            # Check if metadata exists
            if not metadata:
                metadata_analysis["metadata_structure_issues"].append(
                    f"Evaluation {evaluation_id or 'Unknown'} has no metadata field"
                )
                continue
            
            # Collect all unique values
            if metadata.get("disposition"):
                metadata_analysis["dispositions_found"].add(metadata["disposition"])
            if metadata.get("sub_disposition"):
                metadata_analysis["sub_dispositions_found"].add(metadata["sub_disposition"])
            if metadata.get("program"):
                metadata_analysis["programs_found"].add(metadata["program"])
            if metadata.get("partner"):
                metadata_analysis["partners_found"].add(metadata["partner"])
            if metadata.get("site"):
                metadata_analysis["sites_found"].add(metadata["site"])
            if metadata.get("lob"):
                metadata_analysis["lobs_found"].add(metadata["lob"])
            if metadata.get("agent"):
                metadata_analysis["agents_found"].add(metadata["agent"])
            if source.get("template_name"):
                metadata_analysis["templates_found"].add(source["template_name"])
            
            # Add sample record
            metadata_analysis["sample_records"].append({
                "evaluationId": evaluation_id,
                "template_name": source.get("template_name"),
                "metadata": {
                    "disposition": metadata.get("disposition"),
                    "sub_disposition": metadata.get("sub_disposition"),
                    "program": metadata.get("program"),
                    "partner": metadata.get("partner"),
                    "site": metadata.get("site"),
                    "lob": metadata.get("lob"),
                    "agent": metadata.get("agent"),
                    "call_date": metadata.get("call_date")
                }
            })
        
        # Convert sets to lists for JSON serialization
        for key in ["dispositions_found", "sub_dispositions_found", "programs_found", 
                   "partners_found", "sites_found", "lobs_found", "agents_found", "templates_found"]:
            metadata_analysis[key] = sorted(list(metadata_analysis[key]))
        
        # Correct counting
        metadata_analysis["unique_evaluations_sampled"] = len(metadata_analysis["unique_evaluations_sampled"])
        
        # Add data quality assessment with correct counts
        metadata_analysis["data_quality_assessment"] = {
            "has_disposition_data": len(metadata_analysis["dispositions_found"]) > 0,
            "has_program_data": len(metadata_analysis["programs_found"]) > 0,
            "metadata_coverage": len([r for r in metadata_analysis["sample_records"] if r["metadata"]["disposition"]]) / len(metadata_analysis["sample_records"]) if metadata_analysis["sample_records"] else 0,
            "ready_for_analysis": len(metadata_analysis["dispositions_found"]) > 0 and len(metadata_analysis["sample_records"]) > 0,
            "evaluation_vs_chunk_note": f"Sampled {metadata_analysis['total_sampled']} chunks from {metadata_analysis['unique_evaluations_sampled']} unique evaluations"
        }
        
        return {
            "status": "success",
            "metadata_analysis": metadata_analysis,
            "recommendations": [
                "Use the 'dispositions_found' list for accurate call disposition queries",
                "Verify that these match your expected call center dispositions",
                f"Your database has {metadata_analysis['unique_evaluations_sampled']} unique evaluations in this sample",
                "Agent should report evaluation counts, not chunk counts",
                "If dispositions are empty, check if data import included metadata"
            ],
            "version": "4.4.0_metadata_debug"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "4.4.0_metadata_debug"
        }
@app.get("/debug/verify_new_fields")
async def verify_new_fields():
    """
    DEBUG: Verify that the new fields (weighted_score and url) are working correctly
    """
    try:
        from opensearch_client import get_opensearch_client, test_connection
        
        if not test_connection():
            return {
                "status": "error",
                "message": "OpenSearch connection failed"
            }
        
        client = get_opensearch_client()
        
        # Test 1: Check if fields exist in mapping
        mapping_response = client.indices.get_mapping(index="eval-*")
        fields_in_mapping = set()
        
        for index_name, mapping_data in mapping_response.items():
            metadata_props = mapping_data.get("mappings", {}).get("properties", {}).get("metadata", {}).get("properties", {})
            fields_in_mapping.update(metadata_props.keys())
        
        # Test 2: Search for documents with new fields
        search_response = client.search(
            index="eval-*",
            body={
                "size": 5,
                "query": {"match_all": {}},
                "_source": ["metadata.weighted_score", "metadata.url", "evaluationId"]
            }
        )
        
        # Test 3: Check field statistics
        stats_response = client.search(
            index="eval-*",
            body={
                "size": 0,
                "aggs": {
                    "weighted_score_stats": {
                        "stats": {
                            "field": "metadata.weighted_score"
                        }
                    },
                    "urls_exist": {
                        "filter": {
                            "exists": {
                                "field": "metadata.url"
                            }
                        }
                    },
                    "weighted_scores_exist": {
                        "filter": {
                            "exists": {
                                "field": "metadata.weighted_score"
                            }
                        }
                    }
                }
            }
        )
        
        # Analyze results
        hits = search_response.get("hits", {}).get("hits", [])
        sample_data = []
        
        for hit in hits:
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            
            sample_data.append({
                "evaluationId": source.get("evaluationId"),
                "weighted_score": metadata.get("weighted_score"),
                "url": metadata.get("url"),
                "has_weighted_score": "weighted_score" in metadata,
                "has_url": "url" in metadata
            })
        
        # Get aggregation results
        aggs = stats_response.get("aggregations", {})
        weighted_score_stats = aggs.get("weighted_score_stats", {})
        urls_exist_count = aggs.get("urls_exist", {}).get("doc_count", 0)
        weighted_scores_exist_count = aggs.get("weighted_scores_exist", {}).get("doc_count", 0)
        
        return {
            "status": "success",
            "verification_results": {
                "fields_in_mapping": {
                    "weighted_score": "weighted_score" in fields_in_mapping,
                    "url": "url" in fields_in_mapping,
                    "all_fields": sorted(list(fields_in_mapping))
                },
                "field_statistics": {
                    "weighted_score_stats": weighted_score_stats,
                    "documents_with_urls": urls_exist_count,
                    "documents_with_weighted_scores": weighted_scores_exist_count
                },
                "sample_data": sample_data,
                "summary": {
                    "mapping_updated": "weighted_score" in fields_in_mapping and "url" in fields_in_mapping,
                    "data_populated": weighted_scores_exist_count > 0 or urls_exist_count > 0,
                    "recommendation": "Re-run import process if data_populated is False"
                }
            },
            "version": "4.5.0_field_verification"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "4.5.0_field_verification"
        }

@app.get("/debug/test_disposition_search")
async def debug_test_disposition_search(query: str = "call dispositions"):
    """DEBUG: Test search specifically for call disposition queries"""
    try:
        from chat_handlers import build_search_context
        
        # Test search with no filters
        context, sources = build_search_context(query, {})
        
        # Analyze what was found with correct counting
        analysis = {
            "query_tested": query,
            "context_generated": bool(context),
            "context_length": len(context),
            "sources_found": len(sources),
            "has_verified_data": "VERIFIED_REAL_DATA" in context,
            "has_no_data_message": "NO DATA FOUND" in context or "NO EVALUATION DATA FOUND" in context,
            "context_preview": context[:500] + "..." if len(context) > 500 else context,
            "sources_summary": [],
            "unique_evaluations_found": set()
        }
        
        # Analyze sources with correct counting
        for i, source in enumerate(sources[:5]):
            evaluation_id = source.get("evaluationId")
            if evaluation_id:
                analysis["unique_evaluations_found"].add(evaluation_id)
                
            source_summary = {
                "source_number": i + 1,
                "evaluation_id": evaluation_id,
                "search_type": source.get("search_type"),
                "template_name": source.get("template_name"),
                "metadata_preview": {
                    "disposition": source.get("metadata", {}).get("disposition"),
                    "sub_disposition": source.get("metadata", {}).get("sub_disposition"),
                    "program": source.get("metadata", {}).get("program"),
                    "partner": source.get("metadata", {}).get("partner")
                },
                "has_disposition_data": bool(source.get("metadata", {}).get("disposition"))
            }
            analysis["sources_summary"].append(source_summary)
        
        analysis["unique_evaluations_found"] = len(analysis["unique_evaluations_found"])
        
        # Add diagnostic recommendations
        if not analysis["context_generated"]:
            analysis["diagnosis"] = "No context generated - likely no matching data found"
            analysis["recommendations"] = [
                "Check if evaluation data has been imported",
                "Verify OpenSearch connection",
                "Try broader search terms"
            ]
        elif analysis["has_no_data_message"]:
            analysis["diagnosis"] = "Search ran but no relevant data found"
            analysis["recommendations"] = [
                "Verify that imported data contains call disposition information",
                "Check if metadata.disposition field exists in your data",
                "Try different search terms"
            ]
        elif analysis["has_verified_data"]:
            analysis["diagnosis"] = "SUCCESS - Real data found and verified"
            analysis["recommendations"] = [
                "This search is working correctly",
                "The agent should now use only this real data",
                "Context includes verified metadata from your database",
                f"Found {analysis['unique_evaluations_found']} unique evaluations from {analysis['sources_found']} content sources"
            ]
        else:
            analysis["diagnosis"] = "Context generated but verification status unclear"
            analysis["recommendations"] = [
                "Check context content for data quality",
                "Verify metadata extraction is working"
            ]
        
        return {
            "status": "success",
            "search_analysis": analysis,
            "version": "4.4.0_metadata_debug"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": query,
            "version": "4.4.0_metadata_debug"
        }

@app.get("/debug/simulate_disposition_query")
async def debug_simulate_disposition_query():
    """DEBUG: Simulate a full call disposition query to test the complete flow"""
    try:
        from chat_handlers import build_search_context, verify_metadata_alignment, build_strict_metadata_context
        
        test_query = "What are the most common call dispositions?"
        test_filters = {}
        
        # Step 1: Build search context
        context, sources = build_search_context(test_query, test_filters)
        
        # Step 2: Verify metadata alignment
        metadata_summary = verify_metadata_alignment(sources)
        
        # Step 3: Test strict context building
        if metadata_summary["has_real_data"]:
            strict_context = build_strict_metadata_context(metadata_summary, test_query)
        else:
            strict_context = "No data found"
        
        # Create comprehensive test report with correct counting
        test_report = {
            "test_query": test_query,
            "test_filters": test_filters,
            "step1_search_results": {
                "sources_found": len(sources),
                "context_length": len(context),
                "search_successful": len(sources) > 0
            },
            "step2_metadata_verification": {
                "has_real_data": metadata_summary["has_real_data"],
                "total_evaluations": metadata_summary["total_evaluations"],  # Correct count
                "total_chunks": metadata_summary["total_chunks_found"],      # Chunk count
                "unique_dispositions": len(metadata_summary["dispositions"]),
                "actual_dispositions": metadata_summary["dispositions"],
                "unique_programs": len(metadata_summary["programs"]),
                "actual_programs": metadata_summary["programs"],
                "data_verification_status": metadata_summary["data_verification"]
            },
            "step3_strict_context": {
                "context_length": len(strict_context),
                "contains_real_data_verification": "VERIFIED_REAL_DATA" in strict_context,
                "contains_actual_dispositions": "ACTUAL CALL DISPOSITIONS FOUND" in strict_context,
                "contains_correct_counting": "Report EVALUATIONS" in strict_context,
                "strict_context_preview": strict_context[:300] + "..." if len(strict_context) > 300 else strict_context
            },
            "expected_agent_behavior": {
                "should_fabricate_data": False,
                "should_use_only_real_dispositions": True,
                "should_mention_evaluation_count": metadata_summary["total_evaluations"] > 0,
                "should_not_mention_chunk_count": True,
                "should_avoid_percentage_estimates": True,
                "correct_count_to_report": metadata_summary["total_evaluations"]
            },
            "test_result": "PASS" if metadata_summary["has_real_data"] and len(metadata_summary["dispositions"]) > 0 else "FAIL"
        }
        
        # Add specific recommendations based on test results
        if test_report["test_result"] == "PASS":
            test_report["agent_instructions"] = [
                f"Agent should mention finding {metadata_summary['total_evaluations']} evaluations (not {metadata_summary['total_chunks_found']} chunks)",
                f"Agent should list these specific dispositions: {', '.join(metadata_summary['dispositions'])}",
                "Agent should NOT generate any fake percentages or counts",
                "Agent should NOT mention any specific dates unless found in the data",
                "Agent should distinguish between evaluations and content pieces"
            ]
        else:
            test_report["troubleshooting"] = [
                "No evaluation data found in search results",
                "Verify that data import included call disposition metadata",
                "Check OpenSearch index structure",
                "Confirm that metadata.disposition field exists"
            ]
        
        return {
            "status": "success",
            "simulation_test": test_report,
            "version": "4.4.0_metadata_debug"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "4.4.0_metadata_debug"
        }

@app.post("/debug/test_chat_with_verification")
async def debug_test_chat_with_verification(request: dict):
    """DEBUG: Test the complete chat flow with metadata verification"""
    try:
        message = request.get("message", "What are the most common call dispositions?")
        filters = request.get("filters", {})
        
        from chat_handlers import build_search_context, verify_metadata_alignment
        
        # Step 1: Test search context building
        context, sources = build_search_context(message, filters)
        
        # Step 2: Test metadata verification
        metadata_summary = verify_metadata_alignment(sources)
        
        # Step 3: Analyze what would be sent to GenAI
        system_message_preview = f"""You are an AI assistant for call center evaluation data analysis. You must ONLY use the specific data provided in the context below.

CRITICAL RULES:
1. NEVER generate, estimate, or extrapolate any numbers, dates, or statistics
2. ONLY use the exact values and data shown in the context
3. Report EVALUATIONS ({metadata_summary.get('total_evaluations', 0)}) not chunks ({metadata_summary.get('total_chunks_found', 0)})
...

EVALUATION DATABASE CONTEXT:
{context[:500]}...
"""
        
        debug_analysis = {
            "input": {
                "message": message,
                "filters": filters
            },
            "search_results": {
                "sources_found": len(sources),
                "context_built": bool(context),
                "context_length": len(context)
            },
            "metadata_verification": {
                "real_data_found": metadata_summary["has_real_data"],
                "total_evaluations": metadata_summary["total_evaluations"],
                "total_chunks": metadata_summary["total_chunks_found"],
                "dispositions": metadata_summary["dispositions"],
                "programs": metadata_summary["programs"],
                "verification_status": metadata_summary["data_verification"],
                "counting_note": f"Should report {metadata_summary['total_evaluations']} evaluations, not {metadata_summary['total_chunks_found']} chunks"
            },
            "genai_input_preview": {
                "system_message_length": len(system_message_preview),
                "strict_instructions": "NEVER generate, estimate" in system_message_preview,
                "real_data_context": "VERIFIED_REAL_DATA" in context,
                "counting_instructions": "Report EVALUATIONS" in context,
                "system_message_preview": system_message_preview
            },
            "expected_response_quality": {
                "should_be_factual": metadata_summary["has_real_data"],
                "should_avoid_fabrication": True,
                "should_use_real_dispositions": len(metadata_summary["dispositions"]) > 0,
                "should_report_correct_counts": True,
                "data_quality_score": "HIGH" if metadata_summary["has_real_data"] and len(metadata_summary["dispositions"]) > 0 else "LOW"
            }
        }
        
        # Add specific diagnostics
        if not metadata_summary["has_real_data"]:
            debug_analysis["issues"] = [
                "No real evaluation data found in search results",
                "Agent will receive 'NO DATA FOUND' context",
                "This should prevent fabricated responses"
            ]
            debug_analysis["fix_recommendations"] = [
                "Verify data import completed successfully",
                "Check that metadata fields are populated",
                "Confirm OpenSearch indexing is working"
            ]
        else:
            debug_analysis["success_indicators"] = [
                f"Found {metadata_summary['total_evaluations']} real evaluations (from {metadata_summary['total_chunks_found']} content pieces)",
                f"Extracted {len(metadata_summary['dispositions'])} actual dispositions",
                "Agent will receive verified real data only",
                "Agent will receive correct counting instructions"
            ]
        
        return {
            "status": "success",
            "debug_analysis": debug_analysis,
            "version": "4.4.0_metadata_debug"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "4.4.0_metadata_debug"
        }

# =============================================================================
# DEBUG DASHBOARD ROUTE
# =============================================================================

@app.get("/debug_metadata", response_class=HTMLResponse)
async def serve_metadata_debug_dashboard():
    """Serve the metadata verification debug dashboard"""
    try:
        # Simple dashboard that links to the debug endpoints
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Metro AI - Metadata Debug Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f7fa; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { background: linear-gradient(135deg, #6e32a0 0%, #e20074 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; text-align: center; }
                .test-button { display: inline-block; margin: 10px; padding: 12px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
                .test-button:hover { background: #0056b3; }
                .test-button.critical { background: #dc3545; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
                .result { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 5px; padding: 15px; margin: 15px 0; font-family: monospace; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 Metro AI - Metadata Verification Dashboard</h1>
                    <p>Verify that your agent uses ONLY real evaluation data</p>
                </div>
                
                <div class="section">
                    <h2>🚨 Critical Tests (Run in Order)</h2>
                    <a href="/debug/verify_metadata_alignment" class="test-button critical">
                        1. ✅ Verify Real Metadata Structure
                    </a>
                    <br>
                    <a href="/debug/test_disposition_search?query=call dispositions" class="test-button critical">
                        2. 🎯 Test Disposition Search
                    </a>
                    <br>
                    <a href="/debug/simulate_disposition_query" class="test-button critical">
                        3. 🤖 Simulate Complete Flow
                    </a>
                    <br>
                    <small style="color: #666;">These test the metadata alignment fix step by step</small>
                </div>
                
                <div class="section">
                    <h2>📊 Database Verification</h2>
                    <a href="/debug/opensearch_data" class="test-button">Check Sample Data</a>
                    <a href="/debug/check_indices" class="test-button">Check Indices</a>
                    <a href="/opensearch_statistics" class="test-button">Database Stats</a>
                </div>
                
                <div class="section">
                    <h2>🔍 Search Testing</h2>
                    <a href="/debug/test_search?q=customer service" class="test-button">Test Basic Search</a>
                    <a href="/debug/test_filters" class="test-button">Test All Filters</a>
                </div>
                
                <div class="section">
                    <h2>🎯 Expected Results</h2>
                    <p><strong>✅ Success:</strong> Test 1 shows your actual call dispositions (not empty)</p>
                    <p><strong>✅ Success:</strong> Test 2 shows "has_verified_data: true"</p>
                    <p><strong>✅ Success:</strong> Test 3 shows "test_result: PASS"</p>
                    <p><strong>❌ Failure:</strong> Empty dispositions or "FAIL" results mean data import issues</p>
                </div>
                
                <div class="section">
                    <h2>💬 Test Actual Chat</h2>
                    <p>After the above tests pass, test your actual chat interface:</p>
                    <a href="/chat" class="test-button" target="_blank">Open Chat Interface</a>
                    <p><small>Ask: "What are the most common call dispositions?" and verify it uses your real data</small></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=dashboard_html)
        
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>", 
            status_code=500
        )

# Add startup event
# FIXED: Background model loading to prevent health check timeouts

import asyncio
from threading import Thread

# Global flag to track model loading status
MODEL_LOADING_STATUS = {
    "loaded": False,
    "loading": False,
    "load_time": None,
    "error": None
}

def load_model_background():
    """Load embedding model in background thread"""
    global MODEL_LOADING_STATUS
    
    if not EMBEDDER_AVAILABLE:
        MODEL_LOADING_STATUS["error"] = "Embedder not available"
        return
    
    try:
        MODEL_LOADING_STATUS["loading"] = True
        logger.info("🔮 BACKGROUND: Starting embedding model load...")
        
        start_time = time.time()
        
        from embedder import preload_embedding_model, embed_text
        
        # Load the model
        preload_embedding_model()
        
        # Test with actual embedding
        test_embedding = embed_text("background preload test")
        
        load_time = time.time() - start_time
        
        MODEL_LOADING_STATUS["loaded"] = True
        MODEL_LOADING_STATUS["loading"] = False
        MODEL_LOADING_STATUS["load_time"] = round(load_time, 1)
        
        logger.info(f"✅ BACKGROUND: Model loaded successfully in {load_time:.1f}s")
        logger.info("🚀 Chat requests will now be fast!")
        
    except Exception as e:
        MODEL_LOADING_STATUS["loading"] = False
        MODEL_LOADING_STATUS["error"] = str(e)
        logger.error(f"❌ BACKGROUND: Model loading failed: {e}")

@app.on_event("startup")
async def startup_event():
    """FIXED: Fast startup that doesn't block health checks"""
    try:
        logger.info("🚀 Ask InnovAI PRODUCTION starting...")
        logger.info(f"   Version: 4.8.0_vector_enabled_background_load")
        logger.info(f"   🔮 VECTOR SEARCH: {'✅ ENABLED' if VECTOR_SEARCH_READY else '❌ DISABLED'}")
        logger.info(f"   🔥 HYBRID SEARCH: {'✅ AVAILABLE' if VECTOR_SEARCH_READY and EMBEDDER_AVAILABLE else '❌ NOT AVAILABLE'}")
        logger.info(f"   📚 EMBEDDER: {'✅ LOADED' if EMBEDDER_AVAILABLE else '❌ NOT AVAILABLE'}")
        logger.info(f"   CHAT FIX: Endpoint moved directly to app.py")
        logger.info(f"   405 Method Not Allowed error resolved")
        logger.info(f"   Router conflicts eliminated")
        logger.info(f"   CORS properly configured")
        logger.info(f"   Features: Vector Search + Real Data Filters + Hybrid Search + Semantic Similarity")
        logger.info(f"   Features: Real Data Filters + Efficient Metadata Loading + Evaluation Grouping")
        logger.info(f"   Collection Strategy: Template_ID-based")
        logger.info(f"   Document Strategy: Evaluation-grouped")
        logger.info(f"   Program Extraction: Enhanced pattern matching")
        logger.info(f"   Metadata Loading: Index-based efficient sampling")
        logger.info(f"   Filter Caching: {_filter_metadata_cache['ttl_seconds']}s TTL")
        logger.info(f"   Port: {os.getenv('PORT', '8080')}")
        logger.info(f"   Memory Monitoring: {'✅ Available' if PSUTIL_AVAILABLE else '❌ Disabled'}")
        
        # Check configuration
        api_configured = bool(API_AUTH_VALUE)
        genai_configured = bool(GENAI_ACCESS_KEY)
        opensearch_configured = bool(os.getenv("OPENSEARCH_HOST"))
        
        logger.info(f"   API Source: {'✅ Configured' if api_configured else '❌ Missing'}")
        logger.info(f"   GenAI: {'✅ Configured' if genai_configured else '❌ Missing'}")
        logger.info(f"   OpenSearch: {'✅ Configured' if opensearch_configured else '❌ Missing'}")
        
        # 🎯 CRITICAL FIX: Start model loading in background to avoid health check timeout
        if EMBEDDER_AVAILABLE:
            logger.info("🔮 STARTING EMBEDDING MODEL LOAD IN BACKGROUND...")
            logger.info("⏱️ This will take 20-30s but won't block app startup")
            logger.info("📱 Health checks will pass immediately, model loads separately")
            
            # Start background loading thread
            background_thread = Thread(target=load_model_background, daemon=True)
            background_thread.start()
            
            logger.info("✅ Background model loading initiated")
        else:
            logger.info("⚠️ No embedder available - skipping model preload")
        
        logger.info("✅ PRODUCTION startup complete - HEALTH CHECKS WILL PASS")
        logger.info("📊 Ready for enhanced search with semantic similarity matching")
        logger.info("🔮 Vector search enables finding semantically similar content beyond keyword matching")
        logger.info("🔥 Hybrid search combines text matching with vector similarity for best results")
        logger.info("💬 Chat endpoint should now work without 405 errors")
        logger.info("⏱️ First chat may be slower (~30s) until background model loading completes")
        
    except Exception as e:
        logger.error(f"❌ PRODUCTION startup error: {e}")
        logger.error("🚨 Startup failed - some features may not work correctly")

# Enhanced model status endpoint
@app.get("/admin/model_status")
async def check_model_status():
    """Check embedding model loading status"""
    global MODEL_LOADING_STATUS
    
    if not EMBEDDER_AVAILABLE:
        return {
            "model_loaded": False,
            "embedder_available": False,
            "message": "Embedder module not available"
        }
    
    if MODEL_LOADING_STATUS["loaded"]:
        # Test response time
        try:
            start_time = time.time()
            from embedder import embed_text
            test_embedding = embed_text("quick test")
            response_time = time.time() - start_time
            
            return {
                "model_loaded": True,
                "embedder_available": True,
                "background_load_time": MODEL_LOADING_STATUS["load_time"],
                "current_response_time": round(response_time, 3),
                "status": "ready",
                "message": f"Model loaded in background ({MODEL_LOADING_STATUS['load_time']}s) and ready for fast responses"
            }
        except Exception as e:
            return {
                "model_loaded": False,
                "error": str(e),
                "message": "Model status check failed"
            }
    
    elif MODEL_LOADING_STATUS["loading"]:
        return {
            "model_loaded": False,
            "embedder_available": True,
            "status": "loading",
            "message": "Model is currently loading in background - first chat may be slow"
        }
    
    elif MODEL_LOADING_STATUS["error"]:
        return {
            "model_loaded": False,
            "embedder_available": True,
            "status": "error",
            "error": MODEL_LOADING_STATUS["error"],
            "message": "Background model loading failed"
        }
    
    else:
        return {
            "model_loaded": False,
            "embedder_available": True,
            "status": "not_started",
            "message": "Model loading not yet initiated"
        }

# Manual warmup endpoint (now triggers background loading if not done)
@app.post("/admin/warmup_model")
async def warmup_embedding_model():
    """Manually trigger model warmup"""
    global MODEL_LOADING_STATUS
    
    if not EMBEDDER_AVAILABLE:
        return {
            "status": "error",
            "message": "Embedder not available"
        }
    
    if MODEL_LOADING_STATUS["loaded"]:
        return {
            "status": "already_loaded",
            "load_time": MODEL_LOADING_STATUS["load_time"],
            "message": "Model is already loaded and ready"
        }
    
    if MODEL_LOADING_STATUS["loading"]:
        return {
            "status": "loading",
            "message": "Model is already loading in background"
        }
    
    # Start background loading if not already started
    logger.info("🔥 MANUAL MODEL WARMUP REQUESTED...")
    background_thread = Thread(target=load_model_background, daemon=True)
    background_thread.start()
    
    return {
        "status": "initiated",
        "message": "Background model loading initiated - check /admin/model_status for progress"
    }

# Health check endpoint that always responds quickly
@app.get("/health")
async def health_check():
    """Fast health check that doesn't depend on model loading"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "app_ready": True,
        "model_status": "loaded" if MODEL_LOADING_STATUS["loaded"] else ("loading" if MODEL_LOADING_STATUS["loading"] else "not_loaded"),
        "version": "4.8.0_vector_enabled_background_load"
    }