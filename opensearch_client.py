# Filename: opensearch_client.py (Enhanced)
# Version: 2.0.0

from opensearchpy import OpenSearch
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default index name (can be overridden per call)
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "innovai-opensearch-storage")

# Create OpenSearch client instance
client = OpenSearch(
    hosts=[os.getenv("OPENSEARCH_HOST", "http://localhost:9200")],
    http_auth=(
        os.getenv("OPENSEARCH_USER", "admin"),
        os.getenv("OPENSEARCH_PASS", "admin")
    ),
    use_ssl=os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true",
    verify_certs=os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true",
    timeout=30,
    max_retries=3,
    retry_on_timeout=True
)

def ensure_index_exists(index_name: str = None):
    """Ensure the OpenSearch index exists with proper mapping for vector search"""
    index = index_name or OPENSEARCH_INDEX
    
    try:
        if not client.indices.exists(index=index):
            logger.info(f"Creating OpenSearch index: {index}")
            
            # Index mapping with vector field for embeddings
            mapping = {
                "mappings": {
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "text": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "offset": {"type": "integer"},
                        "length": {"type": "integer"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 384,  # all-MiniLM-L6-v2 dimension
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "evaluation_id": {"type": "keyword"},
                                "template": {"type": "keyword"},
                                "program": {"type": "keyword"},
                                "site": {"type": "keyword"},
                                "lob": {"type": "keyword"},
                                "agent": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword"}}
                                },
                                "disposition": {
                                    "type": "text",
                                    "fields": {"keyword": {"type": "keyword"}}
                                },
                                "sub_disposition": {"type": "keyword"},
                                "language": {"type": "keyword"},
                                "call_date": {"type": "date"},
                                "call_duration": {"type": "integer"},
                                "created_on": {"type": "date"},
                                "internal_id": {"type": "keyword"}
                            }
                        },
                        "source": {"type": "keyword"},
                        "indexed_at": {"type": "date"}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                }
            }
            
            client.indices.create(index=index, body=mapping)
            logger.info(f"Successfully created index: {index}")
        else:
            logger.debug(f"Index {index} already exists")
            
    except Exception as e:
        logger.error(f"Failed to ensure index exists: {e}")
        raise

def index_document(doc_id: str, body: Dict[str, Any], index_override: str = None) -> Dict[str, Any]:
    """
    Index a document into OpenSearch with automatic index creation
    """
    index = index_override or OPENSEARCH_INDEX
    
    try:
        # Ensure index exists
        ensure_index_exists(index)
        
        # Index the document
        result = client.index(
            index=index,
            id=doc_id,
            body=body,
            refresh=True  # Make immediately searchable
        )
        
        logger.debug(f"Indexed document {doc_id} to {index}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to index document {doc_id}: {e}")
        raise

def search_opensearch(query: str, index_override: str = None, size: int = 10) -> List[Dict[str, Any]]:
    """
    Perform a basic full-text search on the specified index
    """
    index = index_override or OPENSEARCH_INDEX
    
    try:
        body = {
            "size": size,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["text^2", "metadata.agent", "metadata.disposition"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        },
                        {
                            "nested": {
                                "path": "metadata",
                                "query": {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["metadata.*"]
                                    }
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "highlight": {
                "fields": {
                    "text": {
                        "fragment_size": 150,
                        "number_of_fragments": 2
                    }
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"indexed_at": {"order": "desc"}}
            ]
        }
        
        response = client.search(index=index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        
        logger.debug(f"Text search for '{query}' returned {len(hits)} results")
        return hits
        
    except Exception as e:
        logger.error(f"Text search failed for query '{query}': {e}")
        return []

def search_vector(embedding: List[float], k: int = 5, filters: Dict[str, Any] = None, 
                 index_override: str = None) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search using k-NN
    """
    index = index_override or OPENSEARCH_INDEX
    
    try:
        # Build filter query
        must_clauses = []
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    must_clauses.append({"terms": {field: value}})
                else:
                    must_clauses.append({"term": {field: value}})
        
        # Build k-NN query
        knn_query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": k
                    }
                }
            }
        }
        
        # Add filters if present
        if must_clauses:
            knn_query["query"] = {
                "bool": {
                    "must": [knn_query["query"]],
                    "filter": must_clauses
                }
            }
        
        response = client.search(index=index, body=knn_query)
        hits = response.get("hits", {}).get("hits", [])
        
        logger.debug(f"Vector search returned {len(hits)} results")
        return hits
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        # Fallback to script_score if k-NN is not available
        return search_vector_fallback(embedding, k, filters, index)

def search_vector_fallback(embedding: List[float], k: int = 5, filters: Dict[str, Any] = None,
                          index_override: str = None) -> List[Dict[str, Any]]:
    """
    Fallback vector search using script_score when k-NN is not available
    """
    index = index_override or OPENSEARCH_INDEX
    
    try:
        must_clauses = []
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    must_clauses.append({"terms": {field: value}})
                else:
                    must_clauses.append({"term": {field: value}})
        
        body = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": must_clauses if must_clauses else [{"match_all": {}}]
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": embedding
                        }
                    }
                }
            }
        }
        
        response = client.search(index=index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        
        logger.debug(f"Vector fallback search returned {len(hits)} results")
        return hits
        
    except Exception as e:
        logger.error(f"Vector fallback search failed: {e}")
        return []

def search_documents_advanced(query: str, k: int = 5, filters: Dict[str, Any] = None, 
                            index_override: str = None) -> List[Dict[str, Any]]:
    """
    Advanced search combining text and filters
    """
    index = index_override or OPENSEARCH_INDEX
    
    try:
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["text^2", "metadata.agent", "metadata.disposition"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
        ]
        
        # Add filters
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    must_clauses.append({"terms": {field: value}})
                else:
                    must_clauses.append({"term": {field: value}})
        
        body = {
            "size": k,
            "query": {
                "bool": {
                    "must": must_clauses
                }
            },
            "highlight": {
                "fields": {
                    "text": {
                        "fragment_size": 150,
                        "number_of_fragments": 1
                    }
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"indexed_at": {"order": "desc"}}
            ]
        }
        
        response = client.search(index=index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        
        logger.debug(f"Advanced search for '{query}' returned {len(hits)} results")
        return hits
        
    except Exception as e:
        logger.error(f"Advanced search failed: {e}")
        return []

def search_vector_with_fallback(query: str, embedding: List[float], k: int = 5, 
                               filters: Dict[str, Any] = None, index_override: str = None) -> List[Dict[str, Any]]:
    """
    Perform vector search with fallback to keyword search if no results
    """
    # Try vector search first
    vector_results = search_vector(embedding, k, filters, index_override)
    
    if vector_results:
        return vector_results
    
    # Fallback to keyword search
    logger.info(f"Vector search returned no results, falling back to keyword search")
    return search_documents_advanced(query, k, filters, index_override)

def delete_document_chunks(document_id: str, index_override: str = None) -> Dict[str, Any]:
    """
    Delete all chunks for a specific document
    """
    index = index_override or OPENSEARCH_INDEX
    
    try:
        body = {
            "query": {
                "term": {
                    "document_id": document_id
                }
            }
        }
        
        response = client.delete_by_query(index=index, body=body, refresh=True)
        deleted_count = response.get("deleted", 0)
        
        logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to delete chunks for document {document_id}: {e}")
        raise

def count_documents_by_program(index_override: str = None) -> Dict[str, int]:
    """
    Count documents grouped by program
    """
    index = index_override or OPENSEARCH_INDEX
    
    try:
        body = {
            "size": 0,
            "aggs": {
                "programs": {
                    "terms": {
                        "field": "metadata.program",
                        "size": 100
                    }
                }
            }
        }
        
        response = client.search(index=index, body=body)
        buckets = response.get("aggregations", {}).get("programs", {}).get("buckets", [])
        
        result = {}
        for bucket in buckets:
            program = bucket.get("key", "unknown")
            count = bucket.get("doc_count", 0)
            result[program] = count
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to count documents by program: {e}")
        return {}

def cleanup_old_documents(cutoff_date: str, index_override: str = None) -> Dict[str, Any]:
    """
    Delete documents older than cutoff date
    """
    index = index_override or OPENSEARCH_INDEX
    
    try:
        body = {
            "query": {
                "range": {
                    "indexed_at": {
                        "lt": cutoff_date
                    }
                }
            }
        }
        
        response = client.delete_by_query(index=index, body=body, refresh=True)
        logger.info(f"Cleanup deleted {response.get('deleted', 0)} old documents")
        return response
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise

def get_index_stats(index_override: str = None) -> Dict[str, Any]:
    """
    Get statistics about the index
    """
    index = index_override or OPENSEARCH_INDEX
    
    try:
        # Get index stats
        stats_response = client.indices.stats(index=index)
        
        # Get document count
        count_response = client.count(index=index)
        
        # Get sample documents to analyze structure
        sample_response = client.search(
            index=index,
            body={
                "size": 100,
                "_source": ["metadata", "indexed_at"],
                "sort": [{"indexed_at": {"order": "desc"}}]
            }
        )
        
        # Analyze collections and dates
        collections = set()
        oldest_date = None
        newest_date = None
        
        for hit in sample_response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            
            if metadata.get("template"):
                collections.add(metadata["template"])
            
            indexed_at = source.get("indexed_at")
            if indexed_at:
                if not oldest_date or indexed_at < oldest_date:
                    oldest_date = indexed_at
                if not newest_date or indexed_at > newest_date:
                    newest_date = indexed_at
        
        return {
            "total_documents": count_response.get("count", 0),
            "index_size_bytes": stats_response.get("_all", {}).get("total", {}).get("store", {}).get("size_in_bytes", 0),
            "collections": list(collections),
            "oldest_document": oldest_date,
            "newest_document": newest_date
        }
        
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        return {}

def health_check() -> Dict[str, Any]:
    """
    Check OpenSearch health and connectivity
    """
    try:
        # Check cluster health
        health_response = client.cluster.health()
        
        # Check if our index exists
        index_exists = client.indices.exists(index=OPENSEARCH_INDEX)
        
        # Get basic stats if index exists
        doc_count = 0
        if index_exists:
            count_response = client.count(index=OPENSEARCH_INDEX)
            doc_count = count_response.get("count", 0)
        
        return {
            "status": "healthy",
            "cluster_status": health_response.get("status", "unknown"),
            "index_exists": index_exists,
            "document_count": doc_count,
            "index_name": OPENSEARCH_INDEX
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "index_name": OPENSEARCH_INDEX
        }

# Initialize index on module import
try:
    ensure_index_exists()
except Exception as e:
    logger.warning(f"Could not initialize index on startup: {e}")