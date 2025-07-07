# opensearch_client.py - FIXED: Enhanced search with proper debugging and filter application
# Version: 4.3.2 - DEBUGGING ENABLED + PROPER SEARCH FUNCTIONALITY

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

# Global vector support detection
_vector_support_detected = None
_vector_support_tested = False

# Global client instance
_client = None

def get_client():
    """
    PRODUCTION: Get OpenSearch client using exact working setup
    """
    if not OPENSEARCH_AVAILABLE:
        logger.error("OpenSearch library not available")
        return None
    
    try:
        client = OpenSearch(
            hosts=[{
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": int(os.getenv("OPENSEARCH_PORT", "25060"))
            }],
            http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS")),
            use_ssl=True,
            verify_certs=False,
            timeout=30,
            connection_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create OpenSearch client: {e}")
        return None

def get_opensearch_client():
    """
    PRODUCTION: Get or create global client instance
    """
    global _client
    if _client is None:
        _client = get_client()
    return _client

def test_connection() -> bool:
    """
    PRODUCTION: Test connection using simple approach
    """
    client = get_opensearch_client()
    if not client:
        return False
    
    try:
        result = client.ping()
        if result:
            logger.info("✅ OpenSearch connection successful")
            return True
        else:
            logger.error("❌ OpenSearch ping failed")
            return False
    except Exception as e:
        logger.error(f"❌ OpenSearch connection failed: {e}")
        return False

def get_opensearch_config():
    """Get OpenSearch configuration for debugging"""
    return {
        "host": os.getenv("OPENSEARCH_HOST", "not_configured"),
        "port": os.getenv("OPENSEARCH_PORT", "25060"),
        "user": os.getenv("OPENSEARCH_USER", "not_configured"),
        "ssl": True,
        "verify_certs": False
    }

def get_connection_status() -> Dict[str, Any]:
    """
    PRODUCTION: Get simple connection status
    """
    return {
        "connected": test_connection(),
        "host": os.getenv("OPENSEARCH_HOST"),
        "port": os.getenv("OPENSEARCH_PORT", "25060"),
        "user": os.getenv("OPENSEARCH_USER"),
        "last_test": datetime.now().isoformat()
    }

# =============================================================================
# ENHANCED SEARCH FUNCTIONS WITH DEBUGGING
# =============================================================================

def search_opensearch(query: str, index_override: str = None, 
                     filters: Dict[str, Any] = None, size: int = 10) -> List[Dict]:
    """
    ENHANCED: Search evaluations with comprehensive debugging and proper filter support
    """
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available for search")
        return []
    
    # Determine index pattern
    index_pattern = index_override or "eval-*"
    logger.info(f"🔍 SEARCHING: index='{index_pattern}', query='{query}', size={size}")
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
        
        # STEP 2: Build comprehensive search query
        
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
        
        # STEP 3: Apply filters with enhanced field mapping
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
                # Frontend filter key -> OpenSearch field path
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
        
        # STEP 4: Build final search body
        search_body = {
            "query": combined_query,
            "size": size,
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
        logger.info(f"🚀 EXECUTING SEARCH...")
        logger.debug(f"📋 SEARCH BODY: {json.dumps(search_body, indent=2)}")
        
        response = client.search(
            index=index_pattern,
            body=search_body,
            timeout="30s"
        )
        
        # STEP 6: Process results
        hits = response.get("hits", {}).get("hits", [])
        total_hits = response.get("hits", {}).get("total", {})
        
        # Handle different total formats
        if isinstance(total_hits, dict):
            total_count = total_hits.get("value", 0)
        else:
            total_count = total_hits
        
        logger.info(f"✅ SEARCH COMPLETED: {len(hits)} hits returned, {total_count} total matches")
        
        # STEP 7: Build result objects
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
                
                logger.info(f"📄 RESULT {i+1}: {result['evaluationId']} (score: {result['_score']:.3f})")
                
            except Exception as e:
                logger.error(f"❌ Failed to process hit {i}: {e}")
        
        logger.info(f"🎯 SEARCH SUMMARY: {len(results)} processed results")
        return results
        
    except Exception as e:
        logger.error(f"❌ SEARCH FAILED: {e}")
        logger.error(f"🔍 Query: '{query}'")
        logger.error(f"🏷️ Filters: {filters}")
        logger.error(f"📍 Index: '{index_pattern}'")
        return []

def search_vector(query_vector: List[float], index_override: str = None, 
                 size: int = 10) -> List[Dict]:
    """
    ENHANCED: Vector search with proper debugging
    """
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available for vector search")
        return []
    
    if not detect_vector_support(client):
        logger.warning("❌ Vector search not available - cluster doesn't support vectors")
        return []
    
    index_pattern = index_override or "eval-*"
    logger.info(f"🔮 VECTOR SEARCH: index='{index_pattern}', vector_dim={len(query_vector)}, k={size}")
    
    try:
        # OpenSearch 2.x k-NN query syntax
        search_body = {
            "query": {
                "knn": {
                    "document_embedding": {
                        "vector": query_vector,
                        "k": size
                    }
                }
            },
            "size": size,
            "_source": {
                "includes": [
                    "evaluationId", "internalId", "template_id", "template_name",
                    "full_text", "evaluation_text", "transcript_text",
                    "total_chunks", "chunks", "metadata"
                ]
            }
        }
        
        logger.info(f"🚀 EXECUTING VECTOR SEARCH...")
        
        response = client.search(
            index=index_pattern,
            body=search_body,
            timeout="30s"
        )
        
        hits = response.get("hits", {}).get("hits", [])
        
        results = []
        for i, hit in enumerate(hits):
            try:
                source = hit.get("_source", {})
                result = {
                    "_id": hit.get("_id"),
                    "_score": hit.get("_score", 0),
                    "_index": hit.get("_index"),
                    "_source": source,
                    "evaluationId": source.get("evaluationId"),
                    "template_id": source.get("template_id"),
                    "template_name": source.get("template_name"),
                    "total_chunks": source.get("total_chunks", 0),
                    "metadata": source.get("metadata", {}),
                    "text": source.get("full_text", "")[:500],
                    "search_type": "vector"
                }
                results.append(result)
                
                logger.info(f"🔮 VECTOR RESULT {i+1}: {result['evaluationId']} (score: {result['_score']:.3f})")
                
            except Exception as e:
                logger.error(f"❌ Failed to process vector hit {i}: {e}")
        
        logger.info(f"✅ VECTOR SEARCH COMPLETED: {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"❌ VECTOR SEARCH FAILED: {e}")
        return []

# =============================================================================
# DEBUG FUNCTIONS
# =============================================================================

def debug_search_simple(query: str = "test") -> Dict[str, Any]:
    """
    DEBUG: Simple search test to verify basic functionality
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
            }
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
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# VECTOR SUPPORT FUNCTIONS (KEEPING EXISTING)
# =============================================================================

def detect_vector_support(client) -> bool:
    """
    PRODUCTION: Conservative vector support detection for OpenSearch 2.x
    """
    global _vector_support_detected, _vector_support_tested
    
    if _vector_support_tested:
        return _vector_support_detected
    
    try:
        logger.info("🔍 Testing OpenSearch 2.x vector support...")
        
        test_index = f"vector-test-{int(time.time())}"
        
        test_mapping = {
            "mappings": {
                "properties": {
                    "test_vector": {
                        "type": "knn_vector",
                        "dimension": 384
                    },
                    "test_text": {
                        "type": "text"
                    }
                }
            }
        }
        
        try:
            client.indices.create(index=test_index, body=test_mapping, timeout="30s")
            client.indices.delete(index=test_index, timeout="30s")
            
            logger.info("✅ OpenSearch 2.x vector support confirmed")
            _vector_support_detected = True
            _vector_support_tested = True
            return True
            
        except Exception as config_error:
            logger.warning(f"❌ Vector configuration failed: {config_error}")
            
            try:
                if client.indices.exists(index=test_index):
                    client.indices.delete(index=test_index)
            except:
                pass
        
        logger.warning("❌ OpenSearch 2.x vector support not available")
        _vector_support_detected = False
        _vector_support_tested = True
        return False
    
    except Exception as e:
        logger.warning(f"⚠️ Vector support detection failed: {e}")
        _vector_support_detected = False
        _vector_support_tested = True
        return False

# =============================================================================
# EXISTING FUNCTIONS (keeping for compatibility)
# =============================================================================

def get_vector_field_mapping(dimension: int = 384) -> Dict[str, Any]:
    """Get simple OpenSearch 2.x compatible vector field mapping"""
    client = get_opensearch_client()
    if not client or not detect_vector_support(client):
        return None
    
    return {
        "type": "knn_vector",
        "dimension": dimension
    }

def ensure_evaluation_index_exists(client, index_name: str):
    """Create index with evaluation grouping mapping and optional vector support"""
    if client.indices.exists(index=index_name):
        return
    
    has_vectors = detect_vector_support(client)
    vector_field = get_vector_field_mapping() if has_vectors else None
    
    logger.info(f"🏗️ Creating index {index_name} with vector support: {has_vectors}")
    
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
    }
    
    if vector_field:
        chunk_properties["embedding"] = vector_field
        logger.info("✅ Added vector field to chunk mapping")
    
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
    
    if vector_field:
        mapping["mappings"]["properties"]["document_embedding"] = vector_field
        logger.info("✅ Added document-level vector field")
    
    try:
        client.indices.create(index=index_name, body=mapping, timeout="60")
        vector_status = "WITH VECTORS" if has_vectors else "TEXT ONLY"
        logger.info(f"✅ Created evaluation index ({vector_status}): {index_name}")
    except Exception as e:
        logger.error(f"❌ Failed to create index {index_name}: {e}")
        raise

def index_document(doc_id: str, document: Dict[str, Any], index_override: str = None) -> bool:
    """Index evaluation document with grouped chunks and conditional vector support"""
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available")
        return False
    
    index_name = index_override or "evaluations-grouped"
    
    try:
        ensure_evaluation_index_exists(client, index_name)
        
        has_vectors = detect_vector_support(client)
        
        if not has_vectors:
            clean_document = remove_vector_fields(document)
            logger.debug("🧹 Removed vector fields (not supported)")
        else:
            clean_document = document
            logger.debug("🔗 Keeping vector fields (supported)")
        
        clean_document["_indexed_at"] = datetime.now().isoformat()
        clean_document["_structure_version"] = "4.3.2"
        clean_document["_document_type"] = "evaluation_grouped_enhanced"
        
        response = client.index(
            index=index_name,
            id=doc_id,
            body=clean_document,
            refresh=True,
            timeout="60s"
        )
        
        vector_status = "WITH VECTORS" if has_vectors else "TEXT ONLY"
        logger.info(f"✅ Indexed evaluation {doc_id} in {index_name} ({vector_status})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to index evaluation {doc_id}: {e}")
        return False

def remove_vector_fields(document: Dict[str, Any]) -> Dict[str, Any]:
    """Remove all vector/embedding fields from document for non-vector clusters"""
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
    """Get a specific evaluation document by ID"""
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
            timeout="10s"
        )
        
        hits = response.get("hits", {}).get("hits", [])
        return hits[0] if hits else None
        
    except Exception as e:
        logger.error(f"❌ Failed to get evaluation {evaluation_id}: {e}")
        return None

def health_check() -> Dict[str, Any]:
    """Health check with vector support detection"""
    try:
        client = get_opensearch_client()
        
        if not client:
            return {
                "status": "not_configured",
                "message": "Could not create OpenSearch client",
                "provider": "unknown"
            }
        
        if test_connection():
            has_vectors = detect_vector_support(client)
            info = client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "cluster_name": info.get("cluster_name", "Unknown"),
                "version": info.get("version", {}).get("number", "Unknown"),
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": os.getenv("OPENSEARCH_PORT", "25060"),
                "user": os.getenv("OPENSEARCH_USER"),
                "structure_version": "4.3.2",
                "document_structure": "evaluation_grouped_enhanced",
                "auth_method": "simple_proven_setup",
                "vector_support": has_vectors,
                "vector_type": "simple_knn_vector" if has_vectors else None,
                "search_type": "hybrid" if has_vectors else "text_only",
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
                "auth_method": "simple_proven_setup",
                "vector_support": None
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "auth_method": "simple_proven_setup",
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
    """Get simple OpenSearch statistics with vector support info"""
    client = get_opensearch_client()
    has_vectors = detect_vector_support(client) if client else False
    
    return {
        "connected": test_connection(),
        "structure_version": "4.3.2",
        "auth_method": "simple_proven_setup",
        "client_type": "enhanced_opensearch_2x",
        "vector_support": has_vectors,
        "vector_type": "simple_knn_vector" if has_vectors else None,
        "debug_features": "enabled"
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing ENHANCED OpenSearch Client v4.3.2")
    print("Expected: Enhanced search with proper debugging")
    
    health = health_check()
    print(f"\n🏥 Health check: {health['status']}")
    
    if health["status"] == "healthy":
        print(f"   Cluster: {health['cluster_name']}")
        print(f"   Version: {health['version']}")
        print(f"   Vector Support: {health['vector_support']}")
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
        
        print("\n✅ ENHANCED client with debugging ready!")
        
    else:
        print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
    
    print("\n🏁 Testing complete!")
else:
    logger.info("🔌 ENHANCED OpenSearch client v4.3.2 loaded")
    logger.info("   Features: Enhanced search + comprehensive debugging + proper filters")