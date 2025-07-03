# opensearch_client.py - Fixed with Proper Port Handling
# Version: 1.8.0

from opensearchpy import OpenSearch
import os
import logging
import time

# Setup logging
logger = logging.getLogger(__name__)

# Default index name for generic functions (your data will use template_name)
DEFAULT_INDEX = "innovai-search-default"

# Build OpenSearch URL properly with separate host and port
def build_opensearch_url():
    """Build OpenSearch URL from host and port environment variables"""
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    
    # TEMPORARY HARDCODED PORT - Until environment variables are updated
    # Remove this hardcoding once OPENSEARCH_PORT=25060 is set in environment
    port = int(os.getenv("OPENSEARCH_PORT", "25060"))
    
    # Log the port being used
    logger.info(f"🔧 Using OpenSearch port: {port}")
    if port == 25060:
        logger.info("   (Using correct port 25060 for Digital Ocean)")
    
    # Remove protocol if included in host
    if host.startswith("http://"):
        host = host.replace("http://", "")
        protocol = "http"
    elif host.startswith("https://"):
        host = host.replace("https://", "")
        protocol = "https"
    else:
        # Default to http for local/development, https for cloud
        protocol = "https" if "cloud" in host.lower() or "digitalocean" in host.lower() else "http"
    
    url = f"{protocol}://{host}:{port}"
    logger.info(f"🔌 OpenSearch URL: {url}")
    return url

# Create OpenSearch client instance with proper URL construction
opensearch_url = build_opensearch_url()

client = OpenSearch(
    hosts=[opensearch_url],
    http_auth=(
        os.getenv("OPENSEARCH_USER", "admin"),
        os.getenv("OPENSEARCH_PASS", "admin")
    ),
    use_ssl=opensearch_url.startswith("https"),
    verify_certs=False,  # Often needed for cloud services
    timeout=15,
    max_retries=2,
    retry_on_timeout=True
)

# Connection status tracking
_connection_status = {
    "tested": False,
    "connected": False,
    "last_error": None,
    "last_test": None
}

def test_connection():
    """Test OpenSearch connection with caching"""
    global _connection_status
    
    # Don't test too frequently
    if _connection_status["tested"]:
        return _connection_status["connected"]
    
    try:
        result = client.ping()
        _connection_status.update({
            "tested": True,
            "connected": result,
            "last_error": None,
            "last_test": time.time()
        })
        
        if result:
            logger.info("✅ OpenSearch connection successful")
        else:
            logger.warning("⚠️ OpenSearch ping returned False")
            
        return result
        
    except Exception as e:
        _connection_status.update({
            "tested": True,
            "connected": False,
            "last_error": str(e),
            "last_test": time.time()
        })
        
        logger.warning(f"⚠️ OpenSearch connection failed: {e}")
        return False

def get_connection_status():
    """Get current connection status"""
    return _connection_status.copy()

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
    
    # Check connection first
    if not _connection_status["connected"] and not test_connection():
        raise Exception(f"OpenSearch not available: {_connection_status['last_error']}")
    
    max_retries = 3
    
    try:
        result = client.index(
            index=index_override,
            id=doc_id,
            body=body,
            timeout='15s'  # Reduced timeout
        )
        logger.debug(f"Document indexed successfully: {doc_id} -> {index_override}")
        return result
        
    except Exception as e:
        error_msg = str(e)
        
        # Check if it's a timeout or connection error
        if any(keyword in error_msg.lower() for keyword in ['timeout', 'connection', 'unreachable']):
            # Mark connection as failed
            _connection_status["connected"] = False
            _connection_status["last_error"] = error_msg
            
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
    
    # Check connection first
    if not _connection_status["connected"] and not test_connection():
        logger.warning("OpenSearch not available for search")
        return []
    
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
        # Mark connection as potentially failed
        _connection_status["connected"] = False
        _connection_status["last_error"] = str(e)
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
    
    # Check connection first
    if not _connection_status["connected"] and not test_connection():
        logger.warning("OpenSearch not available for advanced search")
        return []
    
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
        _connection_status["connected"] = False
        _connection_status["last_error"] = str(e)
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
    
    # Check connection first
    if not _connection_status["connected"] and not test_connection():
        logger.warning("OpenSearch not available for vector search")
        return []
    
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
        _connection_status["connected"] = False
        _connection_status["last_error"] = str(e)
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
    
    # Check connection first
    if not _connection_status["connected"] and not test_connection():
        return {
            "status": "connection_failed",
            "message": f"OpenSearch not available: {_connection_status['last_error']}"
        }
    
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
        _connection_status["connected"] = False
        _connection_status["last_error"] = str(e)
        return {
            "status": "error",
            "message": str(e)
        }

# NON-BLOCKING STARTUP - App can start even if OpenSearch is down
try:
    opensearch_host = os.getenv("OPENSEARCH_HOST", "not_configured")
    opensearch_port = os.getenv("OPENSEARCH_PORT", "25060")
    opensearch_user = os.getenv("OPENSEARCH_USER", "admin")
    
    logger.info(f"🔌 OpenSearch client initialized (non-blocking)")
    logger.info(f"   Host: {opensearch_host}")
    logger.info(f"   Port: {opensearch_port}")
    logger.info(f"   User: {opensearch_user}")
    logger.info(f"   Full URL: {opensearch_url}")
    logger.info(f"   Default index: {DEFAULT_INDEX}")
    logger.info(f"   Data will use template_name as index (e.g., 'Ai Corporate SPTR - TEST')")
    logger.info(f"   Connection will be tested on first use")
    
    # Optional: Test connection in background (non-blocking)
    # This gives us early feedback without blocking startup
    import threading
    def background_test():
        try:
            time.sleep(2)  # Give app time to start
            test_connection()
        except:
            pass  # Don't crash if background test fails
    
    threading.Thread(target=background_test, daemon=True).start()
    
except Exception as e:
    logger.error(f"❌ OpenSearch initialization error: {e}")

# Helper functions for environment info
def get_opensearch_config():
    """Get OpenSearch configuration details"""
    return {
        "host": os.getenv("OPENSEARCH_HOST", "not_configured"),
        "port": int(os.getenv("OPENSEARCH_PORT", "25060")),
        "user": os.getenv("OPENSEARCH_USER", "admin"),
        "url": opensearch_url,
        "ssl_enabled": opensearch_url.startswith("https"),
        "default_index": DEFAULT_INDEX,
        "main_data_index": "Ai Corporate SPTR - TEST"
    }

def get_main_data_index():
    """
    Returns the expected main data index name based on your template.
    This is just for reference - actual indexing uses template_name from data.
    """
    return "Ai Corporate SPTR - TEST"