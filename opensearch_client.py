# opensearch_client.py - Based on Working Version Pattern
# Version: 2.1.2 - Simplified approach based on working client
# Combines working connection pattern with necessary fixes

from opensearchpy import OpenSearch, exceptions
import os
import logging
import time
import random
from typing import Optional, Dict, List, Any, Union
import threading
import json

# Setup logging
logger = logging.getLogger(__name__)

# Default index name for generic functions
DEFAULT_INDEX = "innovai-search-default"

# Connection settings - simplified approach
TIMEOUT_SECONDS = 60  # Increased timeout
MAX_RETRIES = 3

# SSL Configuration - Allow override but default to True (based on working version)
def get_ssl_setting() -> bool:
    """Get SSL setting with manual override capability"""
    ssl_override = os.getenv("OPENSEARCH_SSL", "").lower()
    
    if ssl_override in ["true", "1", "yes", "on"]:
        return True
    elif ssl_override in ["false", "0", "no", "off"]:
        return False
    else:
        # Default to True (based on working version)
        return True

# Create OpenSearch client using working pattern
def create_opensearch_client():
    """Create OpenSearch client using the working pattern"""
    try:
        use_ssl = get_ssl_setting()
        
        # Log SSL setting
        ssl_override = os.getenv("OPENSEARCH_SSL", "").lower()
        if ssl_override:
            logger.info(f"🔒 SSL manually set to: {use_ssl} (OPENSEARCH_SSL={ssl_override})")
        else:
            logger.info(f"🔒 SSL defaulted to: {use_ssl} (based on working version, override with OPENSEARCH_SSL=true/false)")
        
        client = OpenSearch(
            hosts=[{
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": int(os.getenv("OPENSEARCH_PORT", "25060"))
            }],
            http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS")),
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=TIMEOUT_SECONDS,
            request_timeout=TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
            retry_on_timeout=True
        )
        
        # Log connection details
        host = os.getenv("OPENSEARCH_HOST", "unknown")
        port = os.getenv("OPENSEARCH_PORT", "25060")
        protocol = "https" if use_ssl else "http"
        url = f"{protocol}://{host}:{port}"
        
        logger.info(f"✅ OpenSearch client created: {url}")
        logger.info(f"   SSL/TLS: {'Enabled' if use_ssl else 'Disabled'}")
        logger.info(f"   Timeout: {TIMEOUT_SECONDS}s")
        logger.info(f"   Max retries: {MAX_RETRIES}")
        
        return client
        
    except Exception as e:
        logger.error(f"❌ Failed to create OpenSearch client: {e}")
        return None

# Create the global client
client = create_opensearch_client()

class OpenSearchManager:
    """Simplified OpenSearch manager based on working pattern"""
    
    def __init__(self):
        self.client = client
        self.connection_lock = threading.Lock()
        
        # Basic connection status tracking
        self._connection_status = {
            "tested": False,
            "connected": False,
            "last_error": None,
            "last_test": None,
            "last_successful_operation": None,
            "total_operations": 0,
            "failed_operations": 0
        }
        
        # Performance metrics
        self._performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0
        }
    
    def test_connection(self, force_test: bool = False) -> bool:
        """Test OpenSearch connection"""
        if not self.client:
            self._handle_connection_failure("Client not initialized")
            return False
        
        current_time = time.time()
        
        # Use cached result if recent and not forced
        if not force_test and self._connection_status["tested"]:
            if current_time - self._connection_status["last_test"] < 300:  # 5 minutes
                return self._connection_status["connected"]
        
        with self.connection_lock:
            try:
                start_time = time.time()
                
                # Perform ping test
                result = self.client.ping()
                
                response_time = time.time() - start_time
                
                if result:
                    self._handle_connection_success(response_time)
                    return True
                else:
                    self._handle_connection_failure("Ping returned False")
                    return False
                    
            except exceptions.ConnectionError as e:
                self._handle_connection_failure(f"Connection error: {str(e)}")
                return False
            except exceptions.RequestsHTTPError as e:
                self._handle_connection_failure(f"HTTP error: {str(e)}")
                return False
            except Exception as e:
                self._handle_connection_failure(f"Unexpected error: {str(e)}")
                return False
    
    def _handle_connection_success(self, response_time: float):
        """Handle successful connection"""
        current_time = time.time()
        
        self._connection_status.update({
            "tested": True,
            "connected": True,
            "last_error": None,
            "last_test": current_time,
            "last_successful_operation": current_time
        })
        
        # Update performance stats
        self._performance_stats["total_response_time"] += response_time
        self._performance_stats["total_requests"] += 1
        self._performance_stats["successful_requests"] += 1
        
        if self._performance_stats["total_requests"] > 0:
            self._performance_stats["avg_response_time"] = (
                self._performance_stats["total_response_time"] / 
                self._performance_stats["total_requests"]
            )
        
        logger.debug(f"✅ OpenSearch connection successful ({response_time:.2f}s)")
    
    def _handle_connection_failure(self, error_msg: str):
        """Handle connection failures"""
        current_time = time.time()
        
        self._connection_status.update({
            "tested": True,
            "connected": False,
            "last_error": error_msg,
            "last_test": current_time
        })
        
        # Update performance stats
        self._performance_stats["total_requests"] += 1
        self._performance_stats["failed_requests"] += 1
        
        logger.warning(f"⚠️ OpenSearch connection failed: {error_msg}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status"""
        return self._connection_status.copy()
    
    def index_document(self, doc_id: str, body: Dict[str, Any], index_name: str) -> bool:
        """Index document with basic retry logic"""
        if not index_name:
            raise ValueError("index_name is required")
        
        if not self.test_connection():
            raise Exception(f"OpenSearch not available: {self._connection_status.get('last_error', 'Unknown error')}")
        
        start_time = time.time()
        
        try:
            result = self.client.index(
                index=index_name,
                id=doc_id,
                body=body,
                timeout=TIMEOUT_SECONDS,  # Fixed: numeric timeout
                refresh=False
            )
            
            # Update success metrics
            response_time = time.time() - start_time
            self._connection_status["total_operations"] += 1
            self._connection_status["last_successful_operation"] = time.time()
            
            logger.debug(f"✅ Document indexed: {doc_id} -> {index_name} ({response_time:.2f}s)")
            return True
            
        except Exception as e:
            error_msg = str(e)
            
            # Update failure metrics
            self._connection_status["failed_operations"] += 1
            self._connection_status["total_operations"] += 1
            
            logger.error(f"❌ Failed to index document {doc_id}: {error_msg}")
            raise Exception(f"OpenSearch indexing failed: {error_msg}")
    
    def search(self, query: str, index_name: str = None, size: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Perform search"""
        index = index_name or DEFAULT_INDEX
        
        if not self.test_connection():
            logger.warning("OpenSearch not available for search")
            return []
        
        start_time = time.time()
        
        try:
            body = {
                "size": size,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text", "content", "title"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "highlight": {
                    "fields": {
                        "text": {},
                        "content": {}
                    }
                }
            }
            
            response = self.client.search(
                index=index,
                body=body,
                timeout=TIMEOUT_SECONDS,  # Fixed: numeric timeout
                **kwargs
            )
            
            hits = response.get("hits", {}).get("hits", [])
            response_time = time.time() - start_time
            
            # Update success metrics
            self._connection_status["total_operations"] += 1
            self._connection_status["last_successful_operation"] = time.time()
            
            logger.debug(f"✅ Search completed: {query} -> {index} ({len(hits)} results, {response_time:.2f}s)")
            return hits
            
        except Exception as e:
            # Update failure metrics
            self._connection_status["failed_operations"] += 1
            self._connection_status["total_operations"] += 1
            
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def search_vector(self, embedding: List[float], index_name: str = None, size: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        index = index_name or DEFAULT_INDEX
        
        if not self.test_connection():
            logger.warning("OpenSearch not available for vector search")
            return []
        
        try:
            must_clauses = []
            if filters:
                for field, value in filters.items():
                    must_clauses.append({"term": {field: value}})
            
            body = {
                "size": size,
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "must": must_clauses
                            } if must_clauses else {"match_all": {}}
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
            
            response = self.client.search(
                index=index,
                body=body,
                timeout=TIMEOUT_SECONDS  # Fixed: numeric timeout
            )
            
            hits = response.get("hits", {}).get("hits", [])
            logger.debug(f"✅ Vector search completed: {len(embedding)} dims -> {index} ({len(hits)} results)")
            return hits
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return []
    
    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """Get index information"""
        if not self.test_connection():
            return {"status": "connection_failed", "error": self._connection_status.get("last_error")}
        
        try:
            # Check if index exists
            exists = self.client.indices.exists(index=index_name)
            
            if not exists:
                return {"status": "missing", "message": f"Index '{index_name}' does not exist"}
            
            # Get index stats
            stats = self.client.indices.stats(index=index_name)
            
            return {
                "status": "healthy",
                "index_name": index_name,
                "document_count": stats["_all"]["primaries"]["docs"]["count"],
                "index_size": stats["_all"]["primaries"]["store"]["size_in_bytes"],
                "exists": True
            }
            
        except Exception as e:
            logger.error(f"❌ Index info failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_config(self) -> Dict[str, Any]:
        """Get OpenSearch configuration"""
        host = os.getenv("OPENSEARCH_HOST", "unknown")
        port = int(os.getenv("OPENSEARCH_PORT", "25060"))
        user = os.getenv("OPENSEARCH_USER", "admin")
        use_ssl = get_ssl_setting()
        
        # Clean host
        clean_host = host.replace("http://", "").replace("https://", "")
        protocol = "https" if use_ssl else "http"
        url = f"{protocol}://{clean_host}:{port}"
        
        return {
            "host": clean_host,
            "port": port,
            "user": user,
            "protocol": protocol,
            "url": url,
            "use_ssl": use_ssl,
            "ssl_enabled": use_ssl
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        total_ops = self._connection_status["total_operations"]
        failed_ops = self._connection_status["failed_operations"]
        
        success_rate = 0.0
        if total_ops > 0:
            success_rate = ((total_ops - failed_ops) / total_ops) * 100
        
        return {
            **self._connection_status,
            **self._performance_stats,
            "success_rate": f"{success_rate:.1f}%",
            "successful_operations": total_ops - failed_ops
        }

# Global OpenSearch manager instance
_opensearch_manager = None
_manager_lock = threading.Lock()

def get_opensearch_manager() -> OpenSearchManager:
    """Get or create the global OpenSearch manager"""
    global _opensearch_manager
    
    if _opensearch_manager is None:
        with _manager_lock:
            if _opensearch_manager is None:
                _opensearch_manager = OpenSearchManager()
    
    return _opensearch_manager

# Backward compatibility functions
def test_connection() -> bool:
    """Test OpenSearch connection"""
    manager = get_opensearch_manager()
    return manager.test_connection()

def get_connection_status() -> Dict[str, Any]:
    """Get connection status"""
    manager = get_opensearch_manager()
    return manager.get_connection_status()

def index_document(doc_id: str, body: Dict[str, Any], index_override: str = None) -> bool:
    """Index a document"""
    if not index_override:
        raise ValueError("index_override is required")
    
    manager = get_opensearch_manager()
    return manager.index_document(doc_id, body, index_override)

def search_opensearch(query: str, index_override: str = None, size: int = 10) -> List[Dict[str, Any]]:
    """Search OpenSearch"""
    manager = get_opensearch_manager()
    return manager.search(query, index_override, size)

def search_documents_advanced(query: str, k: int = 5, filters: Dict[str, Any] = None, index_override: str = None) -> List[Dict[str, Any]]:
    """Advanced search with filters"""
    manager = get_opensearch_manager()
    return manager.search(query, index_override, k)

def search_vector(embedding: List[float], k: int = 5, filters: Dict[str, Any] = None, index_override: str = None) -> List[Dict[str, Any]]:
    """Vector similarity search"""
    manager = get_opensearch_manager()
    return manager.search_vector(embedding, index_override, k, filters)

def search_vector_with_fallback(query: str, embedding: List[float], k: int = 5, filters: Dict[str, Any] = None, index_override: str = None) -> List[Dict[str, Any]]:
    """Vector search with fallback to keyword search"""
    manager = get_opensearch_manager()
    
    # Try vector search first
    results = manager.search_vector(embedding, index_override, k, filters)
    if results:
        return results
    
    # Fallback to text search
    return manager.search(query, index_override, k)

def get_opensearch_config() -> Dict[str, Any]:
    """Get OpenSearch configuration"""
    manager = get_opensearch_manager()
    return manager.get_config()

def get_index_health(index_override: str = None) -> Dict[str, Any]:
    """Get index health"""
    index_name = index_override or DEFAULT_INDEX
    manager = get_opensearch_manager()
    return manager.get_index_info(index_name)

def get_opensearch_stats() -> Dict[str, Any]:
    """Get OpenSearch statistics"""
    manager = get_opensearch_manager()
    return manager.get_stats()

def get_main_data_index() -> str:
    """Get the main data index name"""
    return "Ai Corporate SPTR - TEST"

# Global client access
def get_opensearch_client():
    """Get the OpenSearch client instance"""
    return client

# Initialize on import
try:
    if client:
        config = get_opensearch_manager().get_config()
        logger.info("🔌 OpenSearch client v2.1.2 initialized (working pattern)")
        logger.info(f"   Target: {config['url']}")
        logger.info(f"   SSL/TLS: {'Enabled' if config['ssl_enabled'] else 'Disabled'}")
        logger.info(f"   Connection will be tested on first use")
    else:
        logger.error("❌ OpenSearch client failed to initialize")
        
except Exception as e:
    logger.error(f"❌ OpenSearch client initialization error: {e}")