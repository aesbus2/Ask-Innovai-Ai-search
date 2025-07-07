# opensearch_client.py - PRODUCTION: OpenSearch 2.x Compatible with Simplified Vector Support
# Version: 4.2.0 - Production-ready with conservative vector field mapping

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
            connection_timeout=30,  # Add explicit connection timeout
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

def search_opensearch(query: str, index_override: str = None, 
                     filters: Dict[str, Any] = None, size: int = 10) -> List[Dict]:
    """
    FIXED: Search with proper timeout format
    """
    client = get_opensearch_client()
    if not client:
        logger.warning("❌ OpenSearch client not available")
        return []
    
    index_pattern = index_override or "eval-*"
    
    try:
        # ... your existing search logic ...
        
        response = client.search(
            index=index_pattern,
            body=search_body,
            timeout=30  # FIX: Use integer instead of "30s"
        )
        
        # ... rest of function ...
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return []

# =============================================================================
# PRODUCTION OPENSEARCH 2.X VECTOR SUPPORT - SIMPLIFIED AND CONSERVATIVE
# =============================================================================

def detect_vector_support(client) -> bool:
    """
    PRODUCTION: Conservative vector support detection for OpenSearch 2.x
    Uses simple test approach to avoid complex configurations
    """
    global _vector_support_detected, _vector_support_tested
    
    if _vector_support_tested:
        return _vector_support_detected
    
    try:
        logger.info("🔍 Testing OpenSearch 2.x vector support...")
        
        # Simple test index name
        test_index = f"vector-test-{int(time.time())}"
        
        # PRODUCTION: Use simple OpenSearch 2.x vector field configuration
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
            # Try to create index with simple vector configuration
            client.indices.create(index=test_index, body=test_mapping, timeout="30s")
            
            # If successful, clean up and confirm support
            client.indices.delete(index=test_index, timeout="30s")
            
            logger.info("✅ OpenSearch 2.x vector support confirmed (simple knn_vector)")
            _vector_support_detected = True
            _vector_support_tested = True
            
            return True
            
        except Exception as config_error:
            logger.warning(f"❌ Simple vector configuration failed: {config_error}")
            
            # Clean up any partially created index
            try:
                if client.indices.exists(index=test_index):
                    client.indices.delete(index=test_index)
            except:
                pass
        
        # If simple config failed, vectors not supported
        logger.warning("❌ OpenSearch 2.x vector support not available")
        logger.info("💡 This cluster may not have k-NN plugin enabled")
        
        _vector_support_detected = False
        _vector_support_tested = True
        return False
    
    except Exception as e:
        logger.warning(f"⚠️ Vector support detection failed: {e}")
        _vector_support_detected = False
        _vector_support_tested = True
        return False

def get_vector_field_mapping(dimension: int = 384) -> Dict[str, Any]:
    """
    PRODUCTION: Get simple OpenSearch 2.x compatible vector field mapping
    """
    client = get_opensearch_client()
    if not client or not detect_vector_support(client):
        return None
    
    # PRODUCTION: Simple OpenSearch 2.x vector field configuration
    return {
        "type": "knn_vector",
        "dimension": dimension
    }

# =============================================================================
# PRODUCTION ENHANCED INDEX MANAGEMENT - OPENSEARCH 2.X OPTIMIZED
# =============================================================================

def ensure_evaluation_index_exists(client, index_name: str):
    """
    PRODUCTION: Create index with evaluation grouping mapping and optional vector support
    """
    if client.indices.exists(index=index_name):
        return
    
    # Detect vector support
    has_vectors = detect_vector_support(client)
    vector_field = get_vector_field_mapping() if has_vectors else None
    
    logger.info(f"🏗️ Creating index {index_name} with vector support: {has_vectors}")
    
    # Build mapping with conditional vector fields
    chunk_properties = {
        "chunk_index": {"type": "integer"},
        "text": {"type": "text", "analyzer": "evaluation_analyzer"},
        "content_type": {"type": "keyword"},
        "length": {"type": "integer"},
        
        # QA-specific fields
        "section": {"type": "keyword"},
        "question": {"type": "text", "analyzer": "evaluation_analyzer"},
        "answer": {"type": "text", "analyzer": "evaluation_analyzer"},
        "qa_pair_index": {"type": "integer"},
        
        # Transcript-specific fields
        "speakers": {"type": "keyword"},
        "timestamps": {"type": "keyword"},
        "speaker_count": {"type": "integer"},
        "transcript_chunk_index": {"type": "integer"}
    }
    
    # Add vector field to chunks if supported
    if vector_field:
        chunk_properties["embedding"] = vector_field
        logger.info("✅ Added vector field to chunk mapping")
    
    # PRODUCTION: Enhanced mapping with conditional vector support
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
                # Primary identifiers
                "evaluationId": {"type": "keyword"},
                "internalId": {"type": "keyword"},
                "template_id": {"type": "keyword"},
                "template_name": {"type": "text", "analyzer": "evaluation_analyzer", 
                                "fields": {"keyword": {"type": "keyword"}}},
                
                # Document structure
                "document_type": {"type": "keyword"},
                "total_chunks": {"type": "integer"},
                "evaluation_chunks_count": {"type": "integer"},
                "transcript_chunks_count": {"type": "integer"},
                
                # Full content for search
                "full_text": {"type": "text", "analyzer": "evaluation_analyzer"},
                "evaluation_text": {"type": "text", "analyzer": "evaluation_analyzer"},
                "transcript_text": {"type": "text", "analyzer": "evaluation_analyzer"},
                
                # Chunks array with nested mapping
                "chunks": {
                    "type": "nested",
                    "properties": chunk_properties
                },
                
                # PRODUCTION: Enhanced metadata mapping
                "metadata": {
                    "properties": {
                        "evaluationId": {"type": "keyword"},
                        "internalId": {"type": "keyword"},
                        #"weight_score": {"type": "float"},  # Enhanced: Weight score field from QA Form
                        "template_id": {"type": "keyword"},
                        "template_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "program": {"type": "keyword"},  # Enhanced: Program field
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
                
                # System fields
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
    
    # Add document-level vector field if supported
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
    """
    PRODUCTION: Index evaluation document with grouped chunks and conditional vector support
    """
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available")
        return False
    
    index_name = index_override or "evaluations-grouped"
    
    try:
        # Ensure index exists with vector detection
        ensure_evaluation_index_exists(client, index_name)
        
        # Check if we should clean vector fields
        has_vectors = detect_vector_support(client)
        
        # Prepare document for indexing
        if not has_vectors:
            # Remove vector fields if not supported
            clean_document = remove_vector_fields(document)
            logger.debug("🧹 Removed vector fields (not supported)")
        else:
            clean_document = document
            logger.debug("🔗 Keeping vector fields (supported)")
        
        # Add system metadata
        clean_document["_indexed_at"] = datetime.now().isoformat()
        clean_document["_structure_version"] = "4.2.0"
        clean_document["_document_type"] = "evaluation_grouped_production"
        
        # Index the document
        response = client.index(
            index=index_name,
            id=doc_id,
            body=clean_document,
            refresh=True,
            timeout="60s"
        )
        
        vector_status = "WITH VECTORS" if has_vectors else "TEXT ONLY"
        logger.info(f"✅ Indexed evaluation {doc_id} in {index_name} ({vector_status})")
        logger.info(f"   📄 Chunks: {clean_document.get('total_chunks', 0)}")
        logger.info(f"   📋 Template: {clean_document.get('template_name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to index evaluation {doc_id}: {e}")
        return False

def remove_vector_fields(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    PRODUCTION: Remove all vector/embedding fields from document for non-vector clusters
    """
    import copy
    clean_doc = copy.deepcopy(document)
    
    # Remove document-level embedding fields
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
    
    # Remove embedding fields from chunks
    if "chunks" in clean_doc and isinstance(clean_doc["chunks"], list):
        for chunk in clean_doc["chunks"]:
            if isinstance(chunk, dict):
                for field in vector_fields_to_remove:
                    if field in chunk:
                        del chunk[field]
                        logger.debug(f"🧹 Removed vector field from chunk: {field}")
    
    return clean_doc

# =============================================================================
# PRODUCTION SEARCH FUNCTIONS - OPENSEARCH 2.X OPTIMIZED
# =============================================================================

def search_opensearch(query: str, index_override: str = None, 
                     filters: Dict[str, Any] = None, size: int = 10) -> List[Dict]:
    """
    PRODUCTION: Search evaluations with enhanced filter support and error handling
    """
    client = get_opensearch_client()
    if not client:
        logger.warning("❌ OpenSearch client not available")
        return []
    
    index_pattern = index_override or "eval-*"
    
    try:
        # Build search query for evaluation documents
        text_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "full_text^2",
                    "evaluation_text^1.5",
                    "transcript_text",
                    "template_name^1.2",
                    "metadata.disposition^1.1",
                    "metadata.agent^1.1"
                ],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
        
        # Also search within chunks using nested query
        nested_query = {
            "nested": {
                "path": "chunks",
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "chunks.text^1.5",
                            "chunks.question",
                            "chunks.answer"
                        ],
                        "fuzziness": "AUTO"
                    }
                },
                "score_mode": "max"
            }
        }
        
        # Combine queries
        combined_query = {
            "bool": {
                "should": [text_query, nested_query],
                "minimum_should_match": 1
            }
        }
        
        # PRODUCTION: Apply filters if provided
        if filters:
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
            
            # PRODUCTION: Enhanced keyword filters with proper field mapping
            keyword_filters = {
                "template_name": "template_name.keyword",
                "template_id": "template_id",
                "program": "metadata.program",
                "partner": "metadata.partner",
                "site": "metadata.site",
                "lob": "metadata.lob",
                "agent_name": "metadata.agent.keyword",
                "disposition": "metadata.disposition",
                "sub_disposition": "metadata.sub_disposition",
                "language": "metadata.language",
                "call_type": "metadata.call_type",
                "phone_number": "metadata.phone_number",
                "contact_id": "metadata.contact_id",
                "ucid": "metadata.ucid"
            }
            
            for filter_key, field_path in keyword_filters.items():
                if filters.get(filter_key):
                    filter_clauses.append({
                        "term": {field_path: filters[filter_key]}
                    })
            
            # Duration filters
            if filters.get("min_duration") or filters.get("max_duration"):
                duration_range = {}
                if filters.get("min_duration"):
                    duration_range["gte"] = filters["min_duration"]
                if filters.get("max_duration"):
                    duration_range["lte"] = filters["max_duration"]
                
                filter_clauses.append({
                    "range": {"metadata.call_duration": duration_range}
                })
            
            if filter_clauses:
                combined_query["bool"]["filter"] = filter_clauses
        
        # Build search body
        search_body = {
            "query": combined_query,
            "size": size,
            "sort": [
                {"_score": {"order": "desc"}},
                {"metadata.call_date": {"order": "desc"}}
            ],
            "highlight": {
                "fields": {
                    "full_text": {"fragment_size": 150, "number_of_fragments": 2},
                    "evaluation_text": {"fragment_size": 150, "number_of_fragments": 1}
                }
            }
        }
        
        response = client.search(
            index=index_pattern,
            body=search_body,
            timeout="30s"
        )
        
        hits = response.get("hits", {}).get("hits", [])
        
        # Process results to return evaluation-level information
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            result = {
                "_id": hit.get("_id"),
                "_score": hit.get("_score", 0),
                "_index": hit.get("_index"),
                "_source": source,
                
                # Evaluation-level fields for easy access
                "evaluationId": source.get("evaluationId"),
                "template_id": source.get("template_id"),
                "template_name": source.get("template_name"),
                "total_chunks": source.get("total_chunks", 0),
                "metadata": source.get("metadata", {}),
                
                # For backward compatibility
                "text": source.get("full_text", "")[:500]
            }
            results.append(result)
        
        has_vectors = detect_vector_support(client)
        search_type = "HYBRID (text + vectors)" if has_vectors else "TEXT ONLY"
        logger.info(f"🔍 Found {len(results)} evaluations ({search_type})")
        return results
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return []

def search_vector(query_vector: List[float], index_override: str = None, 
                 size: int = 10) -> List[Dict]:
    """
    PRODUCTION: Vector search with OpenSearch 2.x k-NN query syntax
    """
    client = get_opensearch_client()
    if not client:
        logger.warning("❌ OpenSearch client not available")
        return []
    
    if not detect_vector_support(client):
        logger.warning("❌ Vector search not available - cluster doesn't support vectors")
        return []
    
    index_pattern = index_override or "eval-*"
    
    try:
        # PRODUCTION: OpenSearch 2.x k-NN query syntax (simple)
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
            "_source": True
        }
        
        logger.debug(f"🔍 Executing OpenSearch 2.x k-NN vector search (k={size})")
        
        response = client.search(
            index=index_pattern,
            body=search_body,
            timeout="30s"
        )
        
        hits = response.get("hits", {}).get("hits", [])
        
        results = []
        for hit in hits:
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
                "text": source.get("full_text", "")[:500]
            }
            results.append(result)
        
        logger.info(f"🔍 k-NN vector search found {len(results)} evaluations")
        return results
        
    except Exception as e:
        logger.error(f"❌ k-NN vector search failed: {e}")
        logger.debug(f"Query was: {search_body}")
        return []

# =============================================================================
# PRODUCTION UTILITY FUNCTIONS
# =============================================================================

def get_evaluation_by_id(evaluation_id: str) -> Optional[Dict]:
    """
    PRODUCTION: Get a specific evaluation document by ID
    """
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
    """
    PRODUCTION: Health check with vector support detection
    """
    try:
        client = get_opensearch_client()
        
        if not client:
            return {
                "status": "not_configured",
                "message": "Could not create OpenSearch client",
                "provider": "unknown"
            }
        
        # Test connection
        if test_connection():
            # Detect vector support
            has_vectors = detect_vector_support(client)
            
            # Get cluster info
            info = client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "cluster_name": info.get("cluster_name", "Unknown"),
                "version": info.get("version", {}).get("number", "Unknown"),
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": os.getenv("OPENSEARCH_PORT", "25060"),
                "user": os.getenv("OPENSEARCH_USER"),
                "structure_version": "4.2.0",
                "document_structure": "evaluation_grouped_production",
                "auth_method": "simple_proven_setup",
                "vector_support": has_vectors,
                "vector_type": "simple_knn_vector" if has_vectors else None,
                "search_type": "hybrid" if has_vectors else "text_only"
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

# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

def get_opensearch_manager():
    """
    PRODUCTION: Get manager for compatibility
    """
    class SimpleManager:
        def test_connection(self):
            return test_connection()
        
        def get_opensearch_config(self):
            return get_opensearch_config()
        
        def get_connection_status(self):
            return get_connection_status()
    
    return SimpleManager()

def get_opensearch_stats() -> Dict[str, Any]:
    """
    PRODUCTION: Get simple OpenSearch statistics with vector support info
    """
    client = get_opensearch_client()
    has_vectors = detect_vector_support(client) if client else False
    
    return {
        "connected": test_connection(),
        "structure_version": "4.2.0",
        "auth_method": "simple_proven_setup",
        "client_type": "production_opensearch_2x",
        "vector_support": has_vectors,
        "vector_type": "simple_knn_vector" if has_vectors else None
    }

if __name__ == "__main__":
    # PRODUCTION test
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing PRODUCTION OpenSearch Client v4.2.0")
    print("Expected: Production-ready with OpenSearch 2.x simple vector support")
    
    # Health check
    health = health_check()
    print(f"\n🏥 Health check: {health['status']}")
    
    if health["status"] == "healthy":
        print(f"   Cluster: {health['cluster_name']}")
        print(f"   Version: {health['version']}")
        print(f"   User: {health['user']}")
        print(f"   Document Structure: {health['document_structure']}")
        print(f"   Vector Support: {health['vector_support']}")
        print(f"   Vector Type: {health.get('vector_type', 'None')}")
        print(f"   Search Type: {health['search_type']}")
        
        if health['vector_support']:
            print("\n✅ PRODUCTION client with SIMPLE VECTOR SUPPORT!")
        else:
            print("\n✅ PRODUCTION client with TEXT-ONLY support!")
        
    else:
        print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
        print("\n🔧 Check your environment variables:")
        print("   OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_USER, OPENSEARCH_PASS")
    
    print("\n🏁 Testing complete!")
else:
    logger.info("🔌 PRODUCTION OpenSearch client v4.2.0 loaded")
    logger.info("   Features: Simple OpenSearch 2.x vector support + evaluation grouping")
    logger.info("   Compatible: OpenSearch 2.x with conservative vector detection")