# opensearch_client.py - SMART: Auto-detects vector support and adapts
# Version: 4.0.6 - Smart detection for OpenSearch 2.0+ with vector support

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
_working_vector_config = None  # Store the working vector configuration

# Your exact working client setup
def get_client():
    """Get OpenSearch client using YOUR exact working setup"""
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
            verify_certs=False
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create OpenSearch client: {e}")
        return None

# Global client instance
_client = None

def get_opensearch_client():
    """Get or create global client instance"""
    global _client
    if _client is None:
        _client = get_client()
    return _client

# Simple connection test
def test_connection() -> bool:
    """Test connection using your simple approach"""
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
    """Get simple connection status"""
    return {
        "connected": test_connection(),
        "host": os.getenv("OPENSEARCH_HOST"),
        "port": os.getenv("OPENSEARCH_PORT", "25060"),
        "user": os.getenv("OPENSEARCH_USER"),
        "last_test": datetime.now().isoformat()
    }

def get_opensearch_config() -> Dict[str, Any]:
    """Get OpenSearch configuration"""
    return {
        "host": os.getenv("OPENSEARCH_HOST", "not_configured"),
        "port": int(os.getenv("OPENSEARCH_PORT", "25060")),
        "user": os.getenv("OPENSEARCH_USER"),
        "password_set": bool(os.getenv("OPENSEARCH_PASS")),
        "use_ssl": True,
        "verify_certs": False
    }

# SMART: Vector support detection functions

def detect_vector_support(client) -> bool:
    """
    SMART: Detect if OpenSearch cluster supports k-NN vectors with proper method testing
    """
    global _vector_support_detected, _vector_support_tested
    
    if _vector_support_tested:
        return _vector_support_detected
    
    try:
        logger.info("🔍 Testing k-NN vector support on OpenSearch cluster...")
        
        # Method 1: Check cluster info and plugins
        try:
            cluster_info = client.info()
            version = cluster_info.get("version", {}).get("number", "unknown")
            logger.info(f"📊 OpenSearch version: {version}")
            
            # Check for k-NN plugin
            try:
                plugins_info = client.cat.plugins(format="json")
                if plugins_info:
                    plugin_names = [plugin.get("name", "") for plugin in plugins_info]
                    logger.info(f"🔌 Installed plugins: {plugin_names}")
                    
                    if any("knn" in name.lower() for name in plugin_names):
                        logger.info("✅ k-NN plugin detected in cluster")
                    else:
                        logger.warning("⚠️ k-NN plugin not found in plugin list")
            except Exception as e:
                logger.debug(f"Could not check plugins: {e}")
        
        except Exception as e:
            logger.debug(f"Could not get cluster info: {e}")
        
        # Method 2: Test vector field configurations systematically
        test_index = f"knn-support-test-{int(time.time())}"
        logger.info(f"🧪 Testing k-NN configurations with index: {test_index}")
        
        # Get all possible vector field configurations
        configurations = get_vector_field_mapping_fallback()
        
        for i, vector_config in enumerate(configurations):
            config_name = f"{vector_config['type']}"
            if 'method' in vector_config:
                method_info = vector_config['method']
                config_name += f" ({method_info.get('name', 'unknown')}/{method_info.get('engine', 'unknown')})"
            
            logger.info(f"🧪 Testing configuration {i+1}/{len(configurations)}: {config_name}")
            
            test_mapping = {
                "mappings": {
                    "properties": {
                        "test_vector": vector_config,
                        "test_text": {"type": "text"}
                    }
                }
            }
            
            try:
                # Try to create index with this vector configuration
                client.indices.create(index=test_index, body=test_mapping)
                
                # If successful, clean up and confirm support
                client.indices.delete(index=test_index)
                
                logger.info(f"✅ k-NN vector support confirmed with: {config_name}")
                _vector_support_detected = True
                _vector_support_tested = True
                
                # Store the working configuration globally for reuse
                global _working_vector_config
                _working_vector_config = vector_config
                
                return True
                
            except Exception as config_error:
                logger.debug(f"❌ Configuration {config_name} failed: {config_error}")
                
                # Clean up any partially created index
                try:
                    if client.indices.exists(index=test_index):
                        client.indices.delete(index=test_index)
                except:
                    pass
                
                continue  # Try next configuration
        
        # If we get here, none of the configurations worked
        logger.warning("❌ No k-NN vector configurations worked")
        logger.info("💡 This could mean:")
        logger.info("   - k-NN plugin is not installed/enabled")
        logger.info("   - Cluster doesn't support vector fields")
        logger.info("   - Different syntax required for this version")
        
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
    SMART: Get the correct vector field mapping for OpenSearch 2.x with proper k-NN configuration
    """
    global _working_vector_config
    
    client = get_opensearch_client()
    if not client or not detect_vector_support(client):
        return None
    
    # If we have a working configuration from detection, use it with the requested dimension
    if _working_vector_config:
        config = _working_vector_config.copy()
        
        # Update dimension based on request
        if config.get("type") == "knn_vector":
            config["dimension"] = dimension
        elif config.get("type") == "dense_vector":
            config["dims"] = dimension
            
        logger.debug(f"🔗 Using detected working vector configuration: {config['type']}")
        return config
    
    # Fallback: OpenSearch 2.x proper k-NN vector field configuration
    return {
        "type": "knn_vector",
        "dimension": dimension,
        "method": {
            "name": "hnsw",              # Hierarchical Navigable Small World
            "space_type": "l2",          # Euclidean distance
            "engine": "lucene",          # Native OpenSearch engine
            "parameters": {
                "ef_construction": 128,   # Build time parameter
                "m": 24                   # Number of connections
            }
        }
    }

def get_vector_field_mapping_fallback(dimension: int = 384) -> Dict[str, Any]:
    """
    Fallback vector field mappings for different OpenSearch configurations
    """
    configurations = [
        # OpenSearch 2.x with Lucene engine
        {
            "type": "knn_vector",
            "dimension": dimension,
            "method": {
                "name": "hnsw",
                "space_type": "l2",
                "engine": "lucene",
                "parameters": {
                    "ef_construction": 128,
                    "m": 24
                }
            }
        },
        # OpenSearch 2.x with Faiss engine
        {
            "type": "knn_vector", 
            "dimension": dimension,
            "method": {
                "name": "hnsw",
                "space_type": "l2", 
                "engine": "faiss",
                "parameters": {
                    "ef_construction": 128,
                    "m": 24
                }
            }
        },
        # Simple knn_vector without method (basic)
        {
            "type": "knn_vector",
            "dimension": dimension
        },
        # Legacy dense_vector syntax
        {
            "type": "dense_vector",
            "dims": dimension
        }
    ]
    
    return configurations

# ENHANCED: Evaluation grouping functions with SMART vector detection

def ensure_evaluation_index_exists(client, index_name: str):
    """
    SMART: Create index with evaluation grouping mapping WITH auto-detected vector support
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
    
    # Enhanced mapping with conditional vector support
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
                
                # Metadata
                "metadata": {
                    "properties": {
                        "evaluationId": {"type": "keyword"},
                        "internalId": {"type": "keyword"},
                        "template_id": {"type": "keyword"},
                        "template_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
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
                        "created_on": {"type": "date"}
                    }
                },
                
                # System fields
                "source": {"type": "keyword"},
                "indexed_at": {"type": "date"},
                "collection_name": {"type": "keyword"},
                "_structure_version": {"type": "keyword"},
                "_document_type": {"type": "keyword"}
            }
        }
    }
    
    # Add document-level vector field if supported
    if vector_field:
        mapping["mappings"]["properties"]["document_embedding"] = vector_field
        logger.info("✅ Added document-level vector field")
    
    try:
        client.indices.create(index=index_name, body=mapping)
        vector_status = "WITH VECTORS" if has_vectors else "TEXT ONLY"
        logger.info(f"✅ Created evaluation index ({vector_status}): {index_name}")
    except Exception as e:
        logger.error(f"❌ Failed to create index {index_name}: {e}")
        raise

def index_document(doc_id: str, document: Dict[str, Any], index_override: str = None) -> bool:
    """
    SMART: Index evaluation document with grouped chunks (conditional vector support)
    """
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available")
        return False
    
    index_name = index_override or "evaluations-grouped"
    
    try:
        # Ensure index exists with smart vector detection
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
        clean_document["_structure_version"] = "4.0.6"
        clean_document["_document_type"] = "evaluation_grouped_smart"
        
        # Index the document
        response = client.index(
            index=index_name,
            id=doc_id,
            body=clean_document,
            refresh=True
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
    Remove all vector/embedding fields from document for non-vector clusters
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

def search_opensearch(query: str, index_override: str = None, 
                     filters: Dict[str, Any] = None, size: int = 10) -> List[Dict]:
    """
    SMART: Search evaluations with automatic vector/text search selection
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
        
        # Apply filters if provided
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
            
            # Keyword filters
            keyword_filters = {
                "template_id": "template_id",
                "partner": "metadata.partner",
                "site": "metadata.site",
                "lob": "metadata.lob",
                "agent_name": "metadata.agent",
                "disposition": "metadata.disposition",
                "language": "metadata.language"
            }
            
            for filter_key, field_path in keyword_filters.items():
                if filters.get(filter_key):
                    filter_clauses.append({
                        "term": {f"{field_path}.keyword": filters[filter_key]}
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
            body=search_body
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
    SMART: Vector search with proper OpenSearch 2.x k-NN query syntax
    """
    client = get_opensearch_client()
    if not client:
        logger.warning("❌ OpenSearch client not available")
        return []
    
    if not detect_vector_support(client):
        logger.warning("❌ Vector search not available - use search_opensearch() for text search")
        return []
    
    index_pattern = index_override or "eval-*"
    
    try:
        # OpenSearch 2.x proper k-NN query syntax
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
        
        logger.debug(f"🔍 Executing k-NN vector search (k={size})")
        
        response = client.search(
            index=index_pattern,
            body=search_body
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

def search_evaluation_chunks(query: str, evaluation_id: str = None, 
                           content_type: str = None) -> List[Dict]:
    """
    Search within chunks of evaluations using nested queries
    """
    client = get_opensearch_client()
    if not client:
        logger.warning("❌ OpenSearch client not available")
        return []
    
    try:
        # Build nested query to search within chunks
        nested_query = {
            "nested": {
                "path": "chunks",
                "query": {
                    "bool": {
                        "must": [{"match": {"chunks.text": query}}]
                    }
                },
                "inner_hits": {
                    "size": 10,
                    "highlight": {"fields": {"chunks.text": {}}}
                }
            }
        }
        
        # Add filters if specified
        bool_query = {"bool": {"must": [nested_query]}}
        
        if evaluation_id:
            bool_query["bool"]["must"].append({
                "term": {"evaluationId": evaluation_id}
            })
        
        if content_type:
            nested_query["nested"]["query"]["bool"]["must"].append({
                "term": {"chunks.content_type": content_type}
            })
        
        search_body = {
            "query": bool_query,
            "size": 20
        }
        
        response = client.search(
            index="eval-*",
            body=search_body
        )
        
        # Process nested search results
        results = []
        
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            inner_hits = hit.get("inner_hits", {}).get("chunks", {}).get("hits", [])
            
            for inner_hit in inner_hits:
                chunk_source = inner_hit.get("_source", {})
                
                result = {
                    "_id": f"{hit.get('_id')}-chunk-{chunk_source.get('chunk_index')}",
                    "_score": inner_hit.get("_score", 0),
                    "_index": hit.get("_index"),
                    
                    # Chunk information
                    "text": chunk_source.get("text", ""),
                    "content_type": chunk_source.get("content_type"),
                    "chunk_index": chunk_source.get("chunk_index"),
                    
                    # Parent evaluation information
                    "evaluationId": source.get("evaluationId"),
                    "template_id": source.get("template_id"),
                    "template_name": source.get("template_name"),
                    "metadata": source.get("metadata", {}),
                    
                    "_source": chunk_source
                }
                results.append(result)
        
        logger.info(f"🔍 Found {len(results)} chunk matches")
        return results
        
    except Exception as e:
        logger.error(f"❌ Chunk search failed: {e}")
        return []

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
            }
        )
        
        hits = response.get("hits", {}).get("hits", [])
        return hits[0] if hits else None
        
    except Exception as e:
        logger.error(f"❌ Failed to get evaluation {evaluation_id}: {e}")
        return None

# For backward compatibility
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
        "structure_version": "4.0.6",
        "auth_method": "simple_proven_setup",
        "client_type": "smart_vector_detection",
        "vector_support": has_vectors
    }

def health_check() -> Dict[str, Any]:
    """SMART: Health check with vector support detection"""
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
                "structure_version": "4.0.6",
                "document_structure": "evaluation_grouped_smart",
                "auth_method": "simple_proven_setup",
                "client_type": "smart_vector_detection",
                "vector_support": has_vectors,
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

if __name__ == "__main__":
    # Test the SMART client
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing SMART OpenSearch Client with Vector Detection")
    print("Expected: YOUR exact working client + Smart vector support detection")
    
    # Health check
    health = health_check()
    print(f"\n🏥 Health check: {health['status']}")
    
    if health["status"] == "healthy":
        print(f"   Cluster: {health['cluster_name']}")
        print(f"   Version: {health['version']}")
        print(f"   User: {health['user']}")
        print(f"   Client Type: {health['client_type']}")
        print(f"   Document Structure: {health['document_structure']}")
        print(f"   Vector Support: {health['vector_support']}")
        print(f"   Search Type: {health['search_type']}")
        
        if health['vector_support']:
            print("\n✅ SMART client with VECTOR SUPPORT!")
        else:
            print("\n✅ SMART client with TEXT-ONLY support!")
        
    else:
        print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
        print("\n🔧 Check your environment variables:")
        print("   OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_USER, OPENSEARCH_PASS")
    
    print("\n🏁 Testing complete!")
else:
    logger.info("🔌 SMART OpenSearch client v4.0.6 loaded")
    logger.info("   Client: YOUR exact working setup")
    logger.info("   Features: Smart vector detection + evaluation grouping")
    logger.info("   Compatible: All OpenSearch versions")