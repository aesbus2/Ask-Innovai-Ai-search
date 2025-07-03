# opensearch_client.py - Simplified without Environment Variable
# Version: 1.6.0

from opensearchpy import OpenSearch
import os
import logging
import time

# Setup logging
logger = logging.getLogger(__name__)

# Default index name for generic functions (your data will use template_name)
DEFAULT_INDEX = "innovai-search-default"

# Create OpenSearch client instance with better timeout settings
client = OpenSearch(
    hosts=[os.getenv("OPENSEARCH_HOST", "http://localhost:9200")],
    http_auth=(
        os.getenv("OPENSEARCH_USER", "admin"),
        os.getenv("OPENSEARCH_PASS", "admin")
    ),
    use_ssl=False,
    verify_certs=False,
    timeout=30,  # Request timeout
    max_retries=3,  # Retry attempts
    retry_on_timeout=True
)

def test_connection():
    """Test OpenSearch connection"""
    try:
        result = client.ping()
        logger.info(f"OpenSearch connection test: {'SUCCESS' if result else 'FAILED'}")
        return result
    except Exception as e:
        logger.error(f"OpenSearch connection test failed: {e}")
        raise

def index_document(doc_id, body, index_override=None, retry_count=0):
    """
    Index a document into OpenSearch with retry logic.
    
    Args:
        doc_id: Document ID
        body: Document content
        index_override: Specific index name (REQUIRED for your data)
        retry_count: Internal retry counter
    """
    if not index_override:
        raise ValueError("index_override is required. Specify the target index name.")
    
    max_retries = 3
    
    try:
        result = client.index(
            index=index_override,
            id=doc_id,
            body=body,
            timeout='30s'
        )
        logger.debug(f"Document indexed successfully: {doc_id} -> {index_override}")
        return result
        
    except Exception as e:
        error_msg = str(e)
        
        # Check if it's a timeout or connection error
        if any(keyword in error_msg.lower() for keyword in ['timeout', 'connection', 'unreachable']):
            if retry_count < max_retries:
                logger.warning(f"OpenSearch timeout, retrying ({retry_count + 1}/{max_retries}): {doc_id}")
                time.sleep(2 ** retry_count)  # Exponential backoff
                return index_document(doc_id, body, index_override, retry_count + 1)
            else:
                logger.error(f"OpenSearch connection failed after {max_retries} retries: {doc_id}")
                raise Exception(f"OpenSearch connection timeout after {max_retries} retries: {error_msg}")
        else:
            logger.error(f"OpenSearch indexing error: {doc_id} - {error_msg}")
            raise Exception(f"OpenSearch indexing error: {error_msg}")

def search_opensearch(query, index_override=None):
    """
    Perform a basic full-text search.
    
    Args:
        query: Search query
        index_override: Specific index name (use your template_name)
    """
    index = index_override or DEFAULT_INDEX
    
    try:
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text"]
                }
            }
        }
        
        response = client.search(
            index=index, 
            body=body,
            timeout='10s'
        )
        
        logger.debug(f"Search completed: {query} -> {index}")
        return response.get("hits", {}).get("hits", [])
        
    except Exception as e:
        logger.error(f"OpenSearch search error: {e}")
        # Return empty results instead of crashing
        return []

def search_documents_advanced(query, k=5, filters=None, index_override=None):
    """
    Perform full-text search with optional metadata filters and top-k limit.
    
    Args:
        query: Search query
        k: Number of results
        filters: Optional filters dict
        index_override: Specific index name (use your template_name)
    """
    index = index_override or DEFAULT_INDEX
    
    try:
        must_clauses = [{"multi_match": {"query": query, "fields": ["text"]}}]

        if filters:
            for field, value in filters.items():
                must_clauses.append({"term": {field: value}})

        body = {
            "size": k,
            "query": {
                "bool": {
                    "must": must_clauses
                }
            }
        }

        response = client.search(
            index=index, 
            body=body,
            timeout='10s'
        )
        
        logger.debug(f"Advanced search completed: {query} -> {index}")
        return response.get("hits", {}).get("hits", [])
        
    except Exception as e:
        logger.error(f"OpenSearch advanced search error: {e}")
        return []

def search_vector(embedding, k=5, filters=None, index_override=None):
    """
    Perform a vector similarity search with optional metadata filters.
    
    Args:
        embedding: Vector embedding
        k: Number of results
        filters: Optional filters dict
        index_override: Specific index name (use your template_name)
    """
    index = index_override or DEFAULT_INDEX
    
    try:
        must = []
        if filters:
            for field, value in filters.items():
                must.append({"term": {field: value}})

        body = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": must
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

        response = client.search(
            index=index, 
            body=body,
            timeout='10s'
        )
        
        logger.debug(f"Vector search completed: {len(embedding)} dims -> {index}")
        return response.get("hits", {}).get("hits", [])
        
    except Exception as e:
        logger.error(f"OpenSearch vector search error: {e}")
        return []

def search_vector_with_fallback(query, embedding, k=5, filters=None, index_override=None):
    """
    Perform vector search with fallback to keyword search if no results are found.
    
    Args:
        query: Search query
        embedding: Vector embedding
        k: Number of results
        filters: Optional filters dict
        index_override: Specific index name (use your template_name)
    """
    try:
        vector_results = search_vector(embedding, k, filters, index_override)
        if vector_results:
            return vector_results
        return search_documents_advanced(query, k, filters, index_override)
    except Exception as e:
        logger.error(f"OpenSearch fallback search error: {e}")
        return []

def get_index_health(index_override=None):
    """
    Get health information for an OpenSearch index.
    
    Args:
        index_override: Specific index name (use your template_name)
    """
    index = index_override or DEFAULT_INDEX
    
    try:
        # Check if index exists
        exists = client.indices.exists(index=index)
        
        if not exists:
            return {
                "status": "missing",
                "message": f"Index '{index}' does not exist"
            }
        
        # Get index stats
        stats = client.indices.stats(index=index)
        
        return {
            "status": "healthy",
            "index_name": index,
            "document_count": stats["_all"]["primaries"]["docs"]["count"],
            "index_size": stats["_all"]["primaries"]["store"]["size_in_bytes"]
        }
        
    except Exception as e:
        logger.error(f"OpenSearch health check error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# Test connection on import
try:
    if test_connection():
        logger.info("✅ OpenSearch client initialized successfully")
        logger.info(f"   Default index for generic functions: {DEFAULT_INDEX}")
        logger.info(f"   Your data will use template_name as index (e.g., 'Ai Corporate SPTR - TEST')")
    else:
        logger.warning("⚠️ OpenSearch connection test failed")
except Exception as e:
    logger.error(f"❌ OpenSearch initialization error: {e}")

# Helper function to get main data index name
def get_main_data_index():
    """
    Returns the expected main data index name based on your template.
    This is just for reference - actual indexing uses template_name from data.
    """
    return "Ai Corporate SPTR - TEST"