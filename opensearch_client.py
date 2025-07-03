# opensearch_client.py - Fixed with Better Error Handling & Retry Logic
# Version: 2.0.0 - Production Ready with Connection Pooling

from opensearchpy import OpenSearch
import os
import logging
import time
import random
from typing import Optional, Dict, List, Any
import threading

# Setup logging
logger = logging.getLogger(__name__)

# Default index name for generic functions
DEFAULT_INDEX = "innovai-search-default"

# Connection pool settings
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 30
TIMEOUT_SECONDS = 30

class OpenSearchManager:
    """Enhanced OpenSearch client with better error handling and connection management"""
    
    def __init__(self):
        self.client = None
        self.connection_lock = threading.Lock()
        self.last_connection_test = 0
        self.connection_test_interval = 300  # 5 minutes
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        
        # Connection status tracking
        self._connection_status = {
            "tested": False,
            "connected": False,
            "last_error": None,
            "last_test": None,
            "last_successful_operation": None,
            "total_operations": 0,
            "failed_operations": 0,
            "consecutive_failures": 0
        }
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenSearch client with robust configuration"""
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
                timeout=TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
                retry_on_timeout=True,
                retry_on_status={429, 502, 503, 504},  # Retry on these HTTP status codes
                # Connection pool settings
                maxsize=25,  # Maximum connections in pool
                block=True,  # Block if no connections available
                # Circuit breaker settings
                sniff_on_start=False,
                sniff_on_connection_fail=False,
                sniffer_timeout=10,
                # Performance settings
                http_compress=True,
                request_timeout=TIMEOUT_SECONDS,
                connections_per_node=10
            )
            
            logger.info(f"✅ OpenSearch client initialized: {config['url']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenSearch client: {e}")
            self.client = None
    
    def _get_config(self) -> Dict[str, Any]:
        """Get OpenSearch configuration from environment"""
        host = os.getenv("OPENSEARCH_HOST", "localhost")
        port = int(os.getenv("OPENSEARCH_PORT", "25060"))
        user = os.getenv("OPENSEARCH_USER", "admin")
        password = os.getenv("OPENSEARCH_PASS", "admin")
        
        # Clean host
        clean_host = host.replace("http://", "").replace("https://", "")
        
        # Determine protocol
        if host.startswith("https://"):
            protocol = "https"
        elif host.startswith("http://"):
            protocol = "http"
        else:
            # Auto-detect based on host patterns
            protocol = "https" if any(pattern in clean_host.lower() for pattern in ["cloud", "digitalocean", "aws", "azure"]) else "http"
        
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
            logger.warning("⚠️ Using localhost with port 25060 - this looks like a cloud config issue")
        
        return True
    
    def test_connection(self, force_test: bool = False) -> bool:
        """Test OpenSearch connection with caching and circuit breaker"""
        current_time = time.time()
        
        # Use cached result if recent and not forced
        if not force_test and self._connection_status["tested"]:
            if current_time - self._connection_status["last_test"] < self.connection_test_interval:
                return self._connection_status["connected"]
        
        # Circuit breaker - if too many consecutive failures, don't test for a while
        if self.consecutive_failures >= self.max_consecutive_failures:
            if current_time - self._connection_status["last_test"] < 600:  # 10 minutes
                logger.warning("⚠️ Circuit breaker open - too many consecutive failures")
                return False
            else:
                logger.info("🔄 Circuit breaker reset - attempting reconnection")
                self.consecutive_failures = 0
        
        with self.connection_lock:
            if not self.client:
                self._initialize_client()
                if not self.client:
                    return False
            
            try:
                start_time = time.time()
                result = self.client.ping()
                response_time = time.time() - start_time
                
                if result:
                    self._connection_status.update({
                        "tested": True,
                        "connected": True,
                        "last_error": None,
                        "last_test": current_time,
                        "last_successful_operation": current_time,
                        "response_time": response_time
                    })
                    self.consecutive_failures = 0
                    logger.info(f"✅ OpenSearch connection successful ({response_time:.2f}s)")
                    return True
                else:
                    self._handle_connection_failure("Ping returned False")
                    return False
                    
            except Exception as e:
                self._handle_connection_failure(str(e))
                return False
    
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
        
        if self.consecutive_failures <= 3:
            logger.warning(f"⚠️ OpenSearch connection failed ({self.consecutive_failures}): {error_msg}")
        elif self.consecutive_failures == self.max_consecutive_failures:
            logger.error(f"❌ OpenSearch connection failed {self.consecutive_failures} times - activating circuit breaker")
        
        # Try to reinitialize client after multiple failures
        if self.consecutive_failures == 5:
            logger.info("🔄 Attempting to reinitialize OpenSearch client")
            self.client = None
            self._initialize_client()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        return self._connection_status.copy()
    
    def index_document(self, doc_id: str, body: Dict[str, Any], index_name: str, retry_count: int = 0) -> bool:
        """Index document with exponential backoff retry logic"""
        if not index_name:
            raise ValueError("index_name is required")
        
        # Check connection
        if not self.test_connection():
            raise Exception(f"OpenSearch not available: {self._connection_status.get('last_error', 'Unknown error')}")
        
        try:
            result = self.client.index(
                index=index_name,
                id=doc_id,
                body=body,
                timeout=f'{TIMEOUT_SECONDS}s',
                refresh=False  # Don't force refresh for better performance
            )
            
            # Update success metrics
            self._connection_status["total_operations"] += 1
            self._connection_status["last_successful_operation"] = time.time()
            
            logger.debug(f"✅ Document indexed: {doc_id} -> {index_name}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            self._connection_status["failed_operations"] += 1
            self._connection_status["total_operations"] += 1
            
            # Check if this is a retryable error
            if self._is_retryable_error(error_msg) and retry_count < MAX_RETRIES:
                delay = self._calculate_retry_delay(retry_count)
                logger.warning(f"⚠️ Retryable error for {doc_id}, retrying in {delay}s ({retry_count + 1}/{MAX_RETRIES}): {error_msg}")
                
                time.sleep(delay)
                return self.index_document(doc_id, body, index_name, retry_count + 1)
            else:
                # Mark connection as potentially failed
                self._connection_status["connected"] = False
                self._connection_status["last_error"] = error_msg
                
                logger.error(f"❌ Failed to index document {doc_id}: {error_msg}")
                raise Exception(f"OpenSearch indexing failed: {error_msg}")
    
    def _is_retryable_error(self, error_msg: str) -> bool:
        """Check if an error is retryable"""
        retryable_patterns = [
            "timeout",
            "connection",
            "unreachable",
            "503",
            "502",
            "504",
            "429",
            "circuit_breaking_exception",
            "too_many_requests"
        ]
        
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)
    
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        base_delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
        max_delay = min(base_delay, MAX_RETRY_DELAY)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.5)
        return max_delay * (1 + jitter)
    
    def search(self, query: str, index_name: str = None, size: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Perform search with error handling"""
        index = index_name or DEFAULT_INDEX
        
        if not self.test_connection():
            logger.warning("OpenSearch not available for search")
            return []
        
        try:
            body = {
                "size": size,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text"]
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
            logger.debug(f"✅ Search completed: {query} -> {index} ({len(hits)} results)")
            return hits
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
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
        """Get connection statistics"""
        total_ops = self._connection_status["total_operations"]
        failed_ops = self._connection_status["failed_operations"]
        
        success_rate = 0.0
        if total_ops > 0:
            success_rate = ((total_ops - failed_ops) / total_ops) * 100
        
        return {
            **self._connection_status,
            "success_rate": f"{success_rate:.1f}%",
            "total_operations": total_ops,
            "successful_operations": total_ops - failed_ops,
            "failed_operations": failed_ops,
            "circuit_breaker_active": self.consecutive_failures >= self.max_consecutive_failures
        }

# Global OpenSearch manager instance
_opensearch_manager = None

def get_opensearch_manager() -> OpenSearchManager:
    """Get or create the global OpenSearch manager"""
    global _opensearch_manager
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

# Initialize on import (non-blocking)
logger.info("🔌 OpenSearch client v2.0.0 initialized (non-blocking)")