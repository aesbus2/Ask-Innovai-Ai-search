# opensearch_client.py - Simple Merged: Your Working Client + Enhanced Features
# Version: 4.0.4 - Your exact working setup + evaluation grouping

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

# ENHANCED: Evaluation grouping functions

def ensure_evaluation_index_exists(client, index_name: str):
    """
    ENHANCED: Create index with evaluation grouping mapping
    """
    if client.indices.exists(index=index_name):
        return
    
    # Enhanced mapping for evaluation-grouped documents
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
                    "properties": {
                        "chunk_index": {"type": "integer"},
                        "text": {"type": "text", "analyzer": "evaluation_analyzer"},
                        "content_type": {"type": "keyword"},
                        "length": {"type": "integer"},
                                                    # "embedding": {"type": "dense_vector", "dims": 384},  # Commented out - not supported in older OpenSearch
                        
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
                },
                
                # Document-level embedding
                "document_embedding": {"type": "dense_vector", "dims": 384},
                
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
    
    try:
        client.indices.create(index=index_name, body=mapping)
        logger.info(f"✅ Created evaluation index: {index_name}")
    except Exception as e:
        logger.error(f"❌ Failed to create index {index_name}: {e}")
        raise

def index_document(doc_id: str, document: Dict[str, Any], index_override: str = None) -> bool:
    """
    ENHANCED: Index evaluation document with grouped chunks
    """
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available")
        return False
    
    index_name = index_override or "evaluations-grouped"
    
    try:
        # Ensure index exists with evaluation mapping
        ensure_evaluation_index_exists(client, index_name)
        
        # Add system metadata
        document["_indexed_at"] = datetime.now().isoformat()
        document["_structure_version"] = "4.0.4"
        document["_document_type"] = "evaluation_grouped"
        
        # Index the document
        response = client.index(
            index=index_name,
            id=doc_id,
            body=document,
            refresh=True
        )
        
        logger.info(f"✅ Indexed evaluation {doc_id} in {index_name}")
        logger.info(f"   📄 Chunks: {document.get('total_chunks', 0)}")
        logger.info(f"   📋 Template: {document.get('template_name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to index evaluation {doc_id}: {e}")
        return False

def search_opensearch(query: str, index_override: str = None, 
                     filters: Dict[str, Any] = None, size: int = 10) -> List[Dict]:
    """
    ENHANCED: Search evaluations (returns evaluation documents, not chunks)
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
        
        logger.info(f"🔍 Found {len(results)} evaluations")
        return results
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return []

def search_vector(query_vector: List[float], index_override: str = None, 
                 size: int = 10) -> List[Dict]:
    """
    ENHANCED: Vector search for evaluation documents
    NOTE: Disabled for older OpenSearch versions that don't support dense_vector
    """
    logger.warning("Vector search not available - OpenSearch version doesn't support dense_vector fields")
    return []

def search_evaluation_chunks(query: str, evaluation_id: str = None, 
                           content_type: str = None) -> List[Dict]:
    """
    NEW: Search within chunks of evaluations using nested queries
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
    """Get simple OpenSearch statistics"""
    return {
        "connected": test_connection(),
        "structure_version": "4.0.4",
        "auth_method": "simple_proven_setup",
        "client_type": "your_exact_working_client"
    }

def health_check() -> Dict[str, Any]:
    """Simple health check using your exact working client"""
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
                "structure_version": "4.0.4",
                "document_structure": "evaluation_grouped",
                "auth_method": "simple_proven_setup",
                "client_type": "your_exact_working_client"
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Connection test failed",
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": os.getenv("OPENSEARCH_PORT", "25060"),
                "user": os.getenv("OPENSEARCH_USER"),
                "auth_method": "simple_proven_setup"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "auth_method": "simple_proven_setup"
        }

if __name__ == "__main__":
    # Test the simple merged client
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing SIMPLE MERGED OpenSearch Client")
    print("Expected: YOUR exact working client + Enhanced evaluation grouping")
    
    # Health check
    health = health_check()
    print(f"\n🏥 Health check: {health['status']}")
    
    if health["status"] == "healthy":
        print(f"   Cluster: {health['cluster_name']}")
        print(f"   Version: {health['version']}")
        print(f"   User: {health['user']}")
        print(f"   Client Type: {health['client_type']}")
        print(f"   Document Structure: {health['document_structure']}")
        
        print("\n✅ YOUR exact working client + Enhanced features working!")
        
    else:
        print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
        print("\n🔧 Check your environment variables:")
        print("   OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_USER, OPENSEARCH_PASS")
    
    print("\n🏁 Testing complete!")
else:
    logger.info("🔌 SIMPLE MERGED OpenSearch client v4.0.4 loaded")
    logger.info("   Client: YOUR exact working setup")
    logger.info("   Features: Enhanced evaluation grouping")