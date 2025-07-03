# opensearch_client.py - Enhanced Production Ready Version
# Version: 2.1.0 - Production Ready with Advanced Error Handling new

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

# Connection pool settings
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 30
TIMEOUT_SECONDS = 30
CONNECTION_POOL_SIZE = 25
CIRCUIT_BREAKER_THRESHOLD = 10

class OpenSearchManager:
    """Enhanced OpenSearch client with advanced error handling and connection management"""
    
    def __init__(self):
        self.client = None
        self.connection_lock = threading.Lock()
        self.last_connection_test = 0
        self.connection_test_interval = 300  # 5 minutes
        self.consecutive_failures = 0
        self.max_consecutive_failures = CIRCUIT_BREAKER_THRESHOLD
        
        # Connection status tracking
        self._connection_status = {
            "tested": False,
            "connected": False,
            "last_error": None,
            "last_test": None,
            "last_successful_operation": None,
            "total_operations": 0,
            "failed_operations": 0,
            "consecutive_failures": 0,
            "circuit_breaker_active": False,
            "circuit_breaker_reset_time": None
        }
        
        # Performance metrics
        self._performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0,
            "retry_count": 0,
            "circuit_breaker_trips": 0
        }
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenSearch client with robust production configuration"""
        try:
            # Get configuration
            config = self._get_config()
            
            # Validate configuration
            if not self._validate_config(config):
                logger.error("❌ Invalid OpenSearch configuration")
                return
            
            # Create client with production-ready settings
            self.client = OpenSearch(
                hosts=[config["url"]],
                http_auth=(config["user"], config["password"]),
                use_ssl=config["use_ssl"],
                verify_certs=False,  # Common for cloud services
                ssl_assert_hostname=False,
                ssl_show_warn=False,
                
                # Timeout settings
                timeout=TIMEOUT_SECONDS,
                request_timeout=TIMEOUT_SECONDS,
                
                # Retry settings
                max_retries=MAX_RETRIES,
                retry_on_timeout=True,
                retry_on_status={429, 502, 503, 504},  # Retry on these HTTP status codes
                
                # Connection pool settings
                maxsize=CONNECTION_POOL_SIZE,
                block=True,  # Block if no connections available
                
                # Performance settings
                http_compress=True,
                connections_per_node=10,
                
                # Circuit breaker settings
                sniff_on_start=False,
                sniff_on_connection_fail=False,
                sniffer_timeout=10,
                
                # Headers
                headers={"User-Agent": "InnovAI-OpenSearch-Client/2.1.0"}
            )
            
            logger.info(f"✅ OpenSearch client initialized: {config['url']}")
            logger.info(f"   Connection pool size: {CONNECTION_POOL_SIZE}")
            logger.info(f"   Timeout: {TIMEOUT_SECONDS}s")
            logger.info(f"   Max retries: {MAX_RETRIES}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenSearch client: {e}")
            self.client = None
    
    def _get_config(self) -> Dict[str, Any]:
        """Get OpenSearch configuration from environment with proper defaults"""
        host = os.getenv("OPENSEARCH_HOST", "localhost")
        port = int(os.getenv("OPENSEARCH_PORT", "25060"))
        user = os.getenv("OPENSEARCH_USER", "admin")
        password = os.getenv("OPENSEARCH_PASS", "admin")
        
        # Clean host - remove any protocol prefixes
        clean_host = host.replace("http://", "").replace("https://", "")
        
        # Determine protocol intelligently
        if host.startswith("https://"):
            protocol = "https"
        elif host.startswith("http://"):
            protocol = "http"
        else:
            # Auto-detect based on host patterns and port
            cloud_patterns = ["cloud", "digitalocean", "aws", "azure", "elasticsearch", "opensearch"]
            is_cloud = any(pattern in clean_host.lower() for pattern in cloud_patterns)
            is_ssl_port = port in [443, 9243, 25060]  # Common SSL ports
            
            protocol = "https" if (is_cloud or is_ssl_port) else "http"
        
        url = f"{protocol}://{clean_host}:{port}"
        
        return {
            "host": clean_host,
            "port": port,
            "user": user,
            "password": password,
            "protocol": protocol,
            "url": url,
            "use_ssl": protocol == "https"
        }
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate OpenSearch configuration"""
        required_fields = ["host", "port", "user", "password"]
        
        for field in required_fields:
            if not config.get(field):
                logger.error(f"❌ Missing required config: {field}")
                return False
        
        # Check for common misconfigurations
        if config["host"] in ["localhost", "127.0.0.1"] and config["port"] == 25060:
            logger.warning("⚠️ Using localhost with port 25060 - this might be a cloud config issue")
        
        if config["password"] == "admin" and "digitalocean" in config["host"].lower():
            logger.warning("⚠️ Using default password 'admin' with Digital Ocean - check your credentials")
        
        return True
    
    def _is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active"""
        if not self._connection_status["circuit_breaker_active"]:
            return False
        
        # Check if enough time has passed to reset circuit breaker
        reset_time = self._connection_status.get("circuit_breaker_reset_time")
        if reset_time and time.time() > reset_time:
            self._reset_circuit_breaker()
            return False
        
        return True
    
    def _activate_circuit_breaker(self):
        """Activate circuit breaker"""
        self._connection_status["circuit_breaker_active"] = True
        self._connection_status["circuit_breaker_reset_time"] = time.time() + 600  # 10 minutes
        self._performance_stats["circuit_breaker_trips"] += 1
        
        logger.error("🔴 Circuit breaker activated - too many consecutive failures")
        logger.info("⏰ Circuit breaker will reset in 10 minutes")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker"""
        self._connection_status["circuit_breaker_active"] = False
        self._connection_status["circuit_breaker_reset_time"] = None
        self.consecutive_failures = 0
        
        logger.info("🟢 Circuit breaker reset - attempting reconnection")
    
    def test_connection(self, force_test: bool = False) -> bool:
        """Test OpenSearch connection with caching and circuit breaker"""
        current_time = time.time()
        
        # Check circuit breaker
        if self._is_circuit_breaker_active():
            logger.debug("⚠️ Circuit breaker active - skipping connection test")
            return False
        
        # Use cached result if recent and not forced
        if not force_test and self._connection_status["tested"]:
            if current_time - self._connection_status["last_test"] < self.connection_test_interval:
                return self._connection_status["connected"]
        
        with self.connection_lock:
            # Double-check pattern
            if not force_test and self._connection_status["tested"]:
                if current_time - self._connection_status["last_test"] < self.connection_test_interval:
                    return self._connection_status["connected"]
            
            if not self.client:
                self._initialize_client()
                if not self.client:
                    self._handle_connection_failure("Client initialization failed")
                    return False
            
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
            "last_successful_operation": current_time,
            "consecutive_failures": 0
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
        
        self.consecutive_failures = 0
        
        logger.debug(f"✅ OpenSearch connection successful ({response_time:.2f}s)")
    
    def _handle_connection_failure(self, error_msg: str):
        """Handle connection failures with proper logging and status updates"""
        self.consecutive_failures += 1
        current_time = time.time()
        
        self._connection_status.update({
            "tested": True,
            "connected": False,
            "last_error": error_msg,
            "last_test": current_time,
            "consecutive_failures": self.consecutive_failures
        })
        
        # Update performance stats
        self._performance_stats["total_requests"] += 1
        self._performance_stats["failed_requests"] += 1
        
        if self.consecutive_failures <= 3:
            logger.warning(f"⚠️ OpenSearch connection failed ({self.consecutive_failures}): {error_msg}")
        elif self.consecutive_failures >= self.max_consecutive_failures:
            self._activate_circuit_breaker()
        
        # Try to reinitialize client after multiple failures
        if self.consecutive_failures == 5:
            logger.info("🔄 Attempting to reinitialize OpenSearch client")
            self.client = None
            self._initialize_client()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        status = self._connection_status.copy()
        status["performance_stats"] = self._performance_stats.copy()
        return status
    
    def _is_retryable_error(self, error_msg: str, exception: Exception = None) -> bool:
        """Check if an error is retryable based on error message and exception type"""
        # Check exception type first
        if isinstance(exception, (
            exceptions.ConnectionTimeout,
            exceptions.ConnectionError,
            exceptions.RequestsHTTPError
        )):
            return True
        
        # Check error message patterns
        retryable_patterns = [
            "timeout",
            "connection",
            "unreachable",
            "503",
            "502",
            "504",
            "429",
            "circuit_breaking_exception",
            "too_many_requests",
            "service_unavailable",
            "bad_gateway",
            "gateway_timeout",
            "temporary failure"
        ]
        
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)
    
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        base_delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
        max_delay = min(base_delay, MAX_RETRY_DELAY)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.5)
        delay = max_delay * (1 + jitter)
        
        return min(delay, MAX_RETRY_DELAY)
    
    def index_document(self, doc_id: str, body: Dict[str, Any], index_name: str, retry_count: int = 0) -> bool:
        """Index document with exponential backoff retry logic"""
        if not index_name:
            raise ValueError("index_name is required")
        
        # Check circuit breaker
        if self._is_circuit_breaker_active():
            raise Exception("Circuit breaker active - OpenSearch temporarily unavailable")
        
        # Check connection
        if not self.test_connection():
            raise Exception(f"OpenSearch not available: {self._connection_status.get('last_error', 'Unknown error')}")
        
        start_time = time.time()
        
        try:
            result = self.client.index(
                index=index_name,
                id=doc_id,
                body=body,
                timeout=f'{TIMEOUT_SECONDS}s',
                refresh=False  # Don't force refresh for better performance
            )
            
            # Update success metrics
            response_time = time.time() - start_time
            self._update_operation_success(response_time)
            
            logger.debug(f"✅ Document indexed: {doc_id} -> {index_name} ({response_time:.2f}s)")
            return True
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            # Update failure metrics
            self._update_operation_failure()
            
            # Check if this is a retryable error
            if self._is_retryable_error(error_msg, e) and retry_count < MAX_RETRIES:
                delay = self._calculate_retry_delay(retry_count)
                self._performance_stats["retry_count"] += 1
                
                logger.warning(f"⚠️ Retryable error for {doc_id}, retrying in {delay:.1f}s ({retry_count + 1}/{MAX_RETRIES}): {error_msg}")
                
                time.sleep(delay)
                return self.index_document(doc_id, body, index_name, retry_count + 1)
            else:
                # Mark connection as potentially failed
                self._connection_status["connected"] = False
                self._connection_status["last_error"] = error_msg
                
                logger.error(f"❌ Failed to index document {doc_id} after {retry_count + 1} attempts: {error_msg}")
                raise Exception(f"OpenSearch indexing failed: {error_msg}")
    
    def _update_operation_success(self, response_time: float):
        """Update metrics for successful operation"""
        self._connection_status["total_operations"] += 1
        self._connection_status["last_successful_operation"] = time.time()
        
        # Update performance stats
        self._performance_stats["total_response_time"] += response_time
        self._performance_stats["total_requests"] += 1
        self._performance_stats["successful_requests"] += 1
        
        if self._performance_stats["total_requests"] > 0:
            self._performance_stats["avg_response_time"] = (
                self._performance_stats["total_response_time"] / 
                self._performance_stats["total_requests"]
            )
    
    def _update_operation_failure(self):
        """Update metrics for failed operation"""
        self._connection_status["failed_operations"] += 1
        self._connection_status["total_operations"] += 1
        
        # Update performance stats
        self._performance_stats["total_requests"] += 1
        self._performance_stats["failed_requests"] += 1
    
    def search(self, query: str, index_name: str = None, size: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Perform search with error handling"""
        index = index_name or DEFAULT_INDEX
        
        # Check circuit breaker
        if self._is_circuit_breaker_active():
            logger.warning("Circuit breaker active - search unavailable")
            return []
        
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
                timeout=f'{TIMEOUT_SECONDS}s',
                **kwargs
            )
            
            hits = response.get("hits", {}).get("hits", [])
            response_time = time.time() - start_time
            
            self._update_operation_success(response_time)
            
            logger.debug(f"✅ Search completed: {query} -> {index} ({len(hits)} results, {response_time:.2f}s)")
            return hits
            
        except Exception as e:
            response_time = time.time() - start_time
            
            self._update_operation_failure()
            
            logger.error(f"❌ Search failed: {e}")
            self._connection_status["connected"] = False
            self._connection_status["last_error"] = str(e)
            return []
    
    def search_vector(self, embedding: List[float], index_name: str = None, size: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        index = index_name or DEFAULT_INDEX
        
        # Check circuit breaker
        if self._is_circuit_breaker_active():
            logger.warning("Circuit breaker active - vector search unavailable")
            return []
        
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
                timeout=f'{TIMEOUT_SECONDS}s'
            )
            
            hits = response.get("hits", {}).get("hits", [])
            logger.debug(f"✅ Vector search completed: {len(embedding)} dims -> {index} ({len(hits)} results)")
            return hits
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            self._connection_status["connected"] = False
            self._connection_status["last_error"] = str(e)
            return []
    
    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """Get index information with error handling"""
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
        return self._get_config()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics"""
        total_ops = self._connection_status["total_operations"]
        failed_ops = self._connection_status["failed_operations"]
        
        success_rate = 0.0
        if total_ops > 0:
            success_rate = ((total_ops - failed_ops) / total_ops) * 100
        
        return {
            **self._connection_status,
            **self._performance_stats,
            "success_rate": f"{success_rate:.1f}%",
            "successful_operations": total_ops - failed_ops,
            "circuit_breaker_active": self._is_circuit_breaker_active()
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self._performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0,
            "retry_count": 0,
            "circuit_breaker_trips": 0
        }
        
        self._connection_status.update({
            "total_operations": 0,
            "failed_operations": 0,
            "consecutive_failures": 0
        })
        
        logger.info("📊 OpenSearch statistics reset")

# Global OpenSearch manager instance
_opensearch_manager = None
_manager_lock = threading.Lock()

def get_opensearch_manager() -> OpenSearchManager:
    """Get or create the global OpenSearch manager (thread-safe)"""
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

def reset_opensearch_stats():
    """Reset OpenSearch statistics"""
    manager = get_opensearch_manager()
    manager.reset_stats()

def get_main_data_index() -> str:
    """Get the main data index name"""
    return "Ai Corporate SPTR - TEST"

# Initialize on import (non-blocking)
try:
    config = get_opensearch_manager().get_config()
    logger.info("🔌 Enhanced OpenSearch client v2.1.0 initialized")
    logger.info(f"   Target: {config['url']}")
    logger.info(f"   Features: Connection pooling, Circuit breaker, Advanced retry logic")
    logger.info(f"   Connection will be tested on first use")
except Exception as e:
    logger.error(f"❌ OpenSearch client initialization warning: {e}")