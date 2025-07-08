# opensearch_client.py - COMPLETE FIXED VERSION
# Version: 4.4.1 - Fixed timeout issues, disabled problematic vector search, enhanced error handling
# FIXES: Timeout configuration, SSL warnings, vector search stability

import os
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

try:
    from opensearchpy import OpenSearch
    from opensearchpy.exceptions import ConnectionError, RequestError
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    logging.warning("opensearch-py not installed. Run: pip install opensearch-py")

# Setup logging
logger = logging.getLogger(__name__)

# Global vector support detection - TEMPORARILY DISABLED
_vector_support_detected = False  # Disabled to prevent timeout errors
_vector_support_tested = True     # Skip testing to avoid issues

# Global client instance
_client = None

# =============================================================================
# FIXED CLIENT CREATION WITH PROPER TIMEOUT CONFIGURATION
# =============================================================================

def get_client():
    """
    FIXED: Get OpenSearch client with proper timeout configuration
    """
    if not OPENSEARCH_AVAILABLE:
        logger.error("OpenSearch library not available")
        return None
    
    try:
        # FIXED: Proper timeout parameter names and SSL configuration
        client = OpenSearch(
            hosts=[{
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": int(os.getenv("OPENSEARCH_PORT", "25060"))
            }],
            http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS")),
            use_ssl=True,
            verify_certs=False,
            # ✅ FIXED TIMEOUT CONFIGURATION:
            request_timeout=30,         # Fixed: was timeout=30
            connect_timeout=30,         # Fixed: was connection_timeout=30
            max_retries=3,
            retry_on_timeout=True,
            # ✅ SUPPRESS SSL WARNINGS:
            ssl_show_warn=False,        # Added to reduce log noise
            # ✅ ADDITIONAL STABILITY:
            pool_maxsize=20,
            http_compress=True
        )
        
        # Test basic connection (without vector operations)
        test_result = client.ping()
        if test_result:
            logger.info("✅ OpenSearch connection successful with fixed timeouts")
        else:
            logger.warning("⚠️ OpenSearch ping returned False")
        
        return client
        
    except Exception as e:
        logger.error(f"Failed to create OpenSearch client: {e}")
        return None

def get_opensearch_client():
    """Get or create global client instance"""
    global _client
    if _client is None:
        _client = get_client()
    return _client

def test_connection() -> bool:
    """Test connection with enhanced error handling"""
    client = get_opensearch_client()
    if not client:
        return False
    
    try:
        result = client.ping()
        if result:
            logger.info("✅ OpenSearch connection test successful")
            return True
        else:
            logger.error("❌ OpenSearch ping failed")
            return False
    except Exception as e:
        logger.error(f"❌ OpenSearch connection test failed: {e}")
        return False

def get_opensearch_config():
    """Get OpenSearch configuration for debugging"""
    return {
        "host": os.getenv("OPENSEARCH_HOST", "not_configured"),
        "port": os.getenv("OPENSEARCH_PORT", "25060"),
        "user": os.getenv("OPENSEARCH_USER", "not_configured"),
        "ssl": True,
        "verify_certs": False,
        "timeout_fixed": True,
        "ssl_warnings_suppressed": True
    }

def get_connection_status() -> Dict[str, Any]:
    """Get connection status with timeout fix information"""
    return {
        "connected": test_connection(),
        "host": os.getenv("OPENSEARCH_HOST"),
        "port": os.getenv("OPENSEARCH_PORT", "25060"),
        "user": os.getenv("OPENSEARCH_USER"),
        "last_test": datetime.now().isoformat(),
        "timeout_configuration": "fixed",
        "ssl_warnings": "suppressed"
    }

# =============================================================================
# ENHANCED SEARCH FUNCTIONS WITH FIXED TIMEOUT HANDLING
# =============================================================================

def search_opensearch(query: str, index_override: str = None, 
                     filters: Dict[str, Any] = None, size: int = 100) -> List[Dict]:
    """
    ENHANCED: Search evaluations with increased default size (was 10, now 100)
    """
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available for search")
        return []
    
    # Determine index pattern
    index_pattern = index_override or "eval-*"
    logger.info(f"🔍 ENHANCED SEARCHING: index='{index_pattern}', query='{query}', size={size}")
    logger.info(f"🏷️ FILTERS: {filters}")
    
    try:
        # STEP 1: Check if any indices exist
        try:
            indices_response = client.indices.get(index=index_pattern)
            available_indices = list(indices_response.keys())
            logger.info(f"📊 AVAILABLE INDICES: {len(available_indices)} ({available_indices[:3]}...)")
        except Exception as e:
            logger.error(f"❌ NO INDICES FOUND for pattern '{index_pattern}': {e}")
            return []
        
        # STEP 2: Build comprehensive search query (same as before, but with larger size)
        
        # Base text query with multiple fields
        text_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "full_text^3",           # Highest priority
                    "evaluation_text^2",     # Second priority  
                    "transcript_text^1.5",   # Third priority
                    "template_name^2",       # Template names important
                    "chunks.text^1.8",       # Chunk content
                    "chunks.question^1.5",   # QA questions
                    "chunks.answer^1.5",     # QA answers
                    "metadata.disposition^1.2",
                    "metadata.agent^1.1"
                ],
                "type": "best_fields",
                "fuzziness": "AUTO",
                "minimum_should_match": "75%"
            }
        }
        
        # Enhanced nested query for chunks
        nested_query = {
            "nested": {
                "path": "chunks",
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "chunks.text^2",
                                        "chunks.question^1.5",
                                        "chunks.answer^1.5"
                                    ],
                                    "fuzziness": "AUTO"
                                }
                            },
                            {
                                "match_phrase": {
                                    "chunks.text": {
                                        "query": query,
                                        "boost": 2
                                    }
                                }
                            }
                        ]
                    }
                },
                "score_mode": "max",
                "inner_hits": {
                    "size": 3,
                    "_source": ["text", "content_type"]
                }
            }
        }
        
        # Combine queries
        combined_query = {
            "bool": {
                "should": [text_query, nested_query],
                "minimum_should_match": 1
            }
        }
        
        # STEP 3: Apply filters (same logic as before)
        if filters and any(filters.values()):
            logger.info(f"🔧 APPLYING FILTERS: {filters}")
            filter_clauses = []
            
            # Date filters
            if filters.get("call_date_start") or filters.get("call_date_end"):
                date_range = {}
                if filters.get("call_date_start"):
                    date_range["gte"] = filters["call_date_start"]
                if filters.get("call_date_end"):
                    date_range["lte"] = filters["call_date_end"]
                
                filter_clauses.append({
                    "range": {"metadata.call_date": date_range}
                })
                logger.info(f"📅 Date filter applied: {date_range}")
            
            # Enhanced keyword filters with fallbacks
            keyword_filters = {
                "template_name": ["template_name.keyword", "template_name"],
                "template_id": ["template_id"],
                "program": ["metadata.program.keyword", "metadata.program"],
                "partner": ["metadata.partner.keyword", "metadata.partner"], 
                "site": ["metadata.site.keyword", "metadata.site"],
                "lob": ["metadata.lob.keyword", "metadata.lob"],
                "agent_name": ["metadata.agent.keyword", "metadata.agent"],
                "agent": ["metadata.agent.keyword", "metadata.agent"],
                "disposition": ["metadata.disposition.keyword", "metadata.disposition"],
                "sub_disposition": ["metadata.sub_disposition.keyword", "metadata.sub_disposition"],
                "language": ["metadata.language.keyword", "metadata.language"],
                "call_type": ["metadata.call_type.keyword", "metadata.call_type"],
                "phone_number": ["metadata.phone_number"],
                "contact_id": ["metadata.contact_id"],
                "ucid": ["metadata.ucid"]
            }
            
            for filter_key, field_paths in keyword_filters.items():
                filter_value = filters.get(filter_key)
                if filter_value and str(filter_value).strip():
                    # Try multiple field paths (with and without .keyword)
                    field_queries = []
                    for field_path in field_paths:
                        field_queries.append({"term": {field_path: filter_value}})
                    
                    if len(field_queries) == 1:
                        filter_clauses.append(field_queries[0])
                    else:
                        filter_clauses.append({
                            "bool": {
                                "should": field_queries
                            }
                        })
                    
                    logger.info(f"🏷️ Filter applied: {filter_key}='{filter_value}' -> {field_paths}")
            
            # Duration filters
            if filters.get("min_duration") or filters.get("max_duration"):
                duration_range = {}
                if filters.get("min_duration"):
                    duration_range["gte"] = int(filters["min_duration"])
                if filters.get("max_duration"):
                    duration_range["lte"] = int(filters["max_duration"])
                
                filter_clauses.append({
                    "range": {"metadata.call_duration": duration_range}
                })
                logger.info(f"⏱️ Duration filter applied: {duration_range}")
            
            # Apply filters to query
            if filter_clauses:
                combined_query["bool"]["filter"] = filter_clauses
                logger.info(f"✅ TOTAL FILTERS APPLIED: {len(filter_clauses)}")
        
        # STEP 4: Build final search body with INCREASED SIZE
        search_body = {
            "query": combined_query,
            "size": size,  # This is now much larger (100 instead of 10)
            "sort": [
                {"_score": {"order": "desc"}},
                {"metadata.call_date": {"order": "desc", "missing": "_last"}}
            ],
            "highlight": {
                "fields": {
                    "full_text": {
                        "fragment_size": 200,
                        "number_of_fragments": 2
                    },
                    "evaluation_text": {
                        "fragment_size": 200, 
                        "number_of_fragments": 1
                    },
                    "chunks.text": {
                        "fragment_size": 150,
                        "number_of_fragments": 1
                    }
                }
            },
            "_source": {
                "includes": [
                    "evaluationId", "internalId", "template_id", "template_name",
                    "full_text", "evaluation_text", "transcript_text",
                    "total_chunks", "chunks", "metadata", "indexed_at"
                ]
            }
        }
        
        # STEP 5: Execute search with detailed logging
        logger.info(f"🚀 EXECUTING ENHANCED SEARCH with size={size}...")
        
        response = client.search(
            index=index_pattern,
            body=search_body,
            timeout="45s"  # Increased timeout for larger results
        )
        
        # STEP 6: Process results
        hits = response.get("hits", {}).get("hits", [])
        total_hits = response.get("hits", {}).get("total", {})
        
        # Handle different total formats
        if isinstance(total_hits, dict):
            total_count = total_hits.get("value", 0)
        else:
            total_count = total_hits
        
        logger.info(f"✅ ENHANCED SEARCH COMPLETED: {len(hits)} hits returned, {total_count} total matches")
        
        # STEP 7: Build result objects (same as before)
        results = []
        for i, hit in enumerate(hits):
            try:
                source = hit.get("_source", {})
                
                result = {
                    "_id": hit.get("_id"),
                    "_score": hit.get("_score", 0),
                    "_index": hit.get("_index"),
                    "_source": source,
                    
                    # Key fields for easy access
                    "evaluationId": source.get("evaluationId"),
                    "internalId": source.get("internalId"),
                    "template_id": source.get("template_id"),
                    "template_name": source.get("template_name"),
                    "total_chunks": source.get("total_chunks", 0),
                    "metadata": source.get("metadata", {}),
                    
                    # Content for context building
                    "text": source.get("full_text", "")[:500],
                    "full_text": source.get("full_text", ""),
                    "evaluation_text": source.get("evaluation_text", ""),
                    "transcript_text": source.get("transcript_text", ""),
                    "chunks": source.get("chunks", []),
                    
                    # Highlighting
                    "highlight": hit.get("highlight", {}),
                    "inner_hits": hit.get("inner_hits", {})
                }
                
                results.append(result)
                
                if i < 10:  # Log first 10 for visibility
                    logger.info(f"📄 RESULT {i+1}: {result['evaluationId']} (score: {result['_score']:.3f})")
                
            except Exception as e:
                logger.error(f"❌ Failed to process hit {i}: {e}")
        
        logger.info(f"🎯 ENHANCED SEARCH SUMMARY: {len(results)} processed results")
        return results
        
    except Exception as e:
        logger.error(f"❌ ENHANCED SEARCH FAILED: {e}")
        logger.error(f"🔍 Query: '{query}'")
        logger.error(f"🏷️ Filters: {filters}")
        logger.error(f"📍 Index: '{index_pattern}'")
        return []

def search_vector(query_vector: List[float], index_override: str = None, 
                 size: int = 10) -> List[Dict]:
    """
    DISABLED: Vector search temporarily disabled to prevent timeout issues
    """
    logger.warning("🔮 Vector search temporarily disabled due to timeout configuration issues")
    logger.info("💡 Using text search only until vector search is stabilized")
    return []

# =============================================================================
# DEBUG FUNCTIONS WITH TIMEOUT FIXES
# =============================================================================

def debug_search_simple(query: str = "test") -> Dict[str, Any]:
    """
    DEBUG: Simple search test with fixed timeouts
    """
    client = get_opensearch_client()
    if not client:
        return {"error": "No client available"}
    
    try:
        # Simple match_all query first
        response = client.search(
            index="eval-*",
            body={
                "query": {"match_all": {}},
                "size": 3,
                "_source": ["evaluationId", "template_name", "metadata.program"]
            },
            request_timeout=30  # ✅ FIXED timeout parameter
        )
        
        hits = response.get("hits", {}).get("hits", [])
        total = response.get("hits", {}).get("total", {})
        
        return {
            "status": "success",
            "total_documents": total.get("value", 0) if isinstance(total, dict) else total,
            "sample_results": [
                {
                    "id": hit.get("_id"),
                    "evaluationId": hit.get("_source", {}).get("evaluationId"),
                    "template": hit.get("_source", {}).get("template_name"),
                    "program": hit.get("_source", {}).get("metadata", {}).get("program")
                }
                for hit in hits
            ],
            "timeout_fix": "applied"
        }
        
    except Exception as e:
        return {"error": str(e), "timeout_fix": "applied_but_failed"}

# =============================================================================
# VECTOR SUPPORT FUNCTIONS - TEMPORARILY DISABLED
# =============================================================================

def detect_vector_support(client) -> bool:
    """
    DISABLED: Vector support detection temporarily disabled to prevent timeout errors
    """
    global _vector_support_detected, _vector_support_tested
    
    # Force disable vector support to prevent timeout issues
    _vector_support_detected = False
    _vector_support_tested = True
    
    logger.info("🚫 Vector support temporarily disabled to prevent timeout errors")
    logger.info("💡 Will re-enable after timeout configuration is fully stabilized")
    
    return False

def get_vector_field_mapping(dimension: int = 384) -> Dict[str, Any]:
    """Vector field mapping disabled"""
    logger.warning("🚫 Vector field mapping disabled (vector support temporarily off)")
    return None

# =============================================================================
# INDEX MANAGEMENT WITH TIMEOUT FIXES
# =============================================================================

def ensure_evaluation_index_exists(client, index_name: str):
    """Create index with evaluation grouping mapping (vector fields disabled)"""
    if client.indices.exists(index=index_name):
        return
    
    logger.info(f"🏗️ Creating index {index_name} (vector support disabled)")
    
    chunk_properties = {
        "chunk_index": {"type": "integer"},
        "text": {"type": "text", "analyzer": "evaluation_analyzer"},
        "content_type": {"type": "keyword"},
        "length": {"type": "integer"},
        "section": {"type": "keyword"},
        "question": {"type": "text", "analyzer": "evaluation_analyzer"},
        "answer": {"type": "text", "analyzer": "evaluation_analyzer"},
        "qa_pair_index": {"type": "integer"},
        "speakers": {"type": "keyword"},
        "timestamps": {"type": "keyword"},
        "speaker_count": {"type": "integer"},
        "transcript_chunk_index": {"type": "integer"}
        # Note: embedding fields omitted (vector support disabled)
    }
    
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "evaluation_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "evaluationId": {"type": "keyword"},
                "internalId": {"type": "keyword"},
                "template_id": {"type": "keyword"},
                "template_name": {"type": "text", "analyzer": "evaluation_analyzer", 
                                "fields": {"keyword": {"type": "keyword"}}},
                
                "document_type": {"type": "keyword"},
                "total_chunks": {"type": "integer"},
                "evaluation_chunks_count": {"type": "integer"},
                "transcript_chunks_count": {"type": "integer"},
                
                "full_text": {"type": "text", "analyzer": "evaluation_analyzer"},
                "evaluation_text": {"type": "text", "analyzer": "evaluation_analyzer"},
                "transcript_text": {"type": "text", "analyzer": "evaluation_analyzer"},
                
                "chunks": {
                    "type": "nested",
                    "properties": chunk_properties
                },
                
                "metadata": {
                    "properties": {
                        "evaluationId": {"type": "keyword"},
                        "internalId": {"type": "keyword"},
                        "template_id": {"type": "keyword"},
                        "template_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "program": {"type": "keyword"},
                        "partner": {"type": "keyword"},
                        "site": {"type": "keyword"},
                        "lob": {"type": "keyword"},
                        "agent": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "agent_id": {"type": "keyword"},
                        "disposition": {"type": "keyword"},
                        "sub_disposition": {"type": "keyword"},
                        "language": {"type": "keyword"},
                        "call_date": {"type": "date"},
                        "call_duration": {"type": "integer"},
                        "created_on": {"type": "date"},
                        "phone_number": {"type": "keyword"},
                        "contact_id": {"type": "keyword"},
                        "ucid": {"type": "keyword"},
                        "call_type": {"type": "keyword"}
                    }
                },
                
                "source": {"type": "keyword"},
                "indexed_at": {"type": "date"},
                "collection_name": {"type": "keyword"},
                "collection_source": {"type": "keyword"},
                "_structure_version": {"type": "keyword"},
                "_document_type": {"type": "keyword"},
                "version": {"type": "keyword"}
            }
        }
    }
    
    # Note: No document-level embedding field (vector support disabled)
    
    try:
        client.indices.create(
            index=index_name, 
            body=mapping, 
            request_timeout=60  # ✅ FIXED timeout parameter
        )
        logger.info(f"✅ Created evaluation index (TEXT ONLY): {index_name}")
    except Exception as e:
        logger.error(f"❌ Failed to create index {index_name}: {e}")
        raise

def index_document(doc_id: str, document: Dict[str, Any], index_override: str = None) -> bool:
    """Index evaluation document with fixed timeout and no vector fields"""
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available")
        return False
    
    index_name = index_override or "evaluations-grouped"
    
    try:
        ensure_evaluation_index_exists(client, index_name)
        
        # Remove any vector fields from document (since vector support disabled)
        clean_document = remove_vector_fields(document)
        
        clean_document["_indexed_at"] = datetime.now().isoformat()
        clean_document["_structure_version"] = "4.4.1_timeout_fixed"
        clean_document["_document_type"] = "evaluation_grouped_text_only"
        
        response = client.index(
            index=index_name,
            id=doc_id,
            body=clean_document,
            refresh=True,
            request_timeout=60  # ✅ FIXED timeout parameter
        )
        
        logger.info(f"✅ Indexed evaluation {doc_id} in {index_name} (TEXT ONLY)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to index evaluation {doc_id}: {e}")
        return False

def remove_vector_fields(document: Dict[str, Any]) -> Dict[str, Any]:
    """Remove all vector/embedding fields from document"""
    import copy
    clean_doc = copy.deepcopy(document)
    
    vector_fields_to_remove = [
        "document_embedding",
        "embedding", 
        "vector",
        "embeddings"
    ]
    
    for field in vector_fields_to_remove:
        if field in clean_doc:
            del clean_doc[field]
            logger.debug(f"🧹 Removed vector field: {field}")
    
    if "chunks" in clean_doc and isinstance(clean_doc["chunks"], list):
        for chunk in clean_doc["chunks"]:
            if isinstance(chunk, dict):
                for field in vector_fields_to_remove:
                    if field in chunk:
                        del chunk[field]
                        logger.debug(f"🧹 Removed vector field from chunk: {field}")
    
    return clean_doc

def get_evaluation_by_id(evaluation_id: str) -> Optional[Dict]:
    """Get a specific evaluation document by ID with fixed timeout"""
    client = get_opensearch_client()
    if not client:
        return None
    
    try:
        response = client.search(
            index="eval-*",
            body={
                "query": {"term": {"evaluationId": evaluation_id}},
                "size": 1
            },
            request_timeout=10  # ✅ FIXED timeout parameter
        )
        
        hits = response.get("hits", {}).get("hits", [])
        return hits[0] if hits else None
        
    except Exception as e:
        logger.error(f"❌ Failed to get evaluation {evaluation_id}: {e}")
        return None

# =============================================================================
# HEALTH CHECK WITH TIMEOUT FIX STATUS
# =============================================================================

def health_check() -> Dict[str, Any]:
    """Health check with timeout fix and vector status information"""
    try:
        client = get_opensearch_client()
        
        if not client:
            return {
                "status": "not_configured",
                "message": "Could not create OpenSearch client",
                "provider": "unknown",
                "timeout_fix": "applied_but_client_creation_failed"
            }
        
        if test_connection():
            info = client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "cluster_name": info.get("cluster_name", "Unknown"),
                "version": info.get("version", {}).get("number", "Unknown"),
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": os.getenv("OPENSEARCH_PORT", "25060"),
                "user": os.getenv("OPENSEARCH_USER"),
                "structure_version": "4.4.1_timeout_fixed",
                "document_structure": "evaluation_grouped_text_only",
                "auth_method": "fixed_timeout_configuration",
                "vector_support": False,
                "vector_status": "temporarily_disabled_for_stability",
                "search_type": "text_only",
                "timeout_configuration": "fixed",
                "ssl_warnings": "suppressed",
                "debug_features": "enabled"
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Connection test failed",
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": os.getenv("OPENSEARCH_PORT", "25060"),
                "user": os.getenv("OPENSEARCH_USER"),
                "timeout_configuration": "fixed_but_connection_failed",
                "vector_support": None
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timeout_configuration": "fixed_but_error_occurred",
            "vector_support": None
        }

def get_opensearch_manager():
    """Get manager for compatibility"""
    class SimpleManager:
        def test_connection(self):
            return test_connection()
        
        def get_opensearch_config(self):
            return get_opensearch_config()
        
        def get_connection_status(self):
            return get_connection_status()
    
    return SimpleManager()

def get_opensearch_stats() -> Dict[str, Any]:
    """Get OpenSearch statistics with timeout fix information"""
    client = get_opensearch_client()
    
    return {
        "connected": test_connection(),
        "structure_version": "4.4.1_timeout_fixed",
        "auth_method": "fixed_timeout_configuration",
        "client_type": "enhanced_opensearch_2x_text_only",
        "vector_support": False,
        "vector_status": "temporarily_disabled_for_stability",
        "timeout_configuration": "fixed",
        "ssl_warnings": "suppressed",
        "debug_features": "enabled"
    }

# =============================================================================
# MAIN EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing FIXED OpenSearch Client v4.4.1")
    print("✅ Expected: Fixed timeout configuration, vector search disabled")
    
    health = health_check()
    print(f"\n🏥 Health check: {health['status']}")
    
    if health["status"] == "healthy":
        print(f"   Cluster: {health['cluster_name']}")
        print(f"   Version: {health['version']}")
        print(f"   Vector Support: {health['vector_support']} (temporarily disabled)")
        print(f"   Timeout Fix: {health['timeout_configuration']}")
        print(f"   SSL Warnings: {health['ssl_warnings']}")
        print(f"   Debug Features: {health['debug_features']}")
        
        # Test simple search
        print("\n🔍 Testing simple search...")
        debug_result = debug_search_simple()
        print(f"   Status: {debug_result.get('status', 'failed')}")
        print(f"   Total docs: {debug_result.get('total_documents', 0)}")
        
        if debug_result.get("sample_results"):
            print("   Sample results:")
            for i, result in enumerate(debug_result["sample_results"][:3]):
                print(f"     {i+1}. {result['evaluationId']} - {result['template']}")
        
        print("\n✅ FIXED client with timeout configuration ready!")
        print("🔍 Text search should work, vector search temporarily disabled")
        
    else:
        print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
        print(f"   Timeout fix status: {health.get('timeout_configuration', 'unknown')}")
    
    print("\n🏁 Testing complete!")
else:
    logger.info("🔌 FIXED OpenSearch client v4.4.1 loaded")
    logger.info("   ✅ Timeout configuration fixed")
    logger.info("   🚫 Vector search temporarily disabled for stability")
    logger.info("   🔍 Text search fully functional")
    logger.info("   🛡️ SSL warnings suppressed")