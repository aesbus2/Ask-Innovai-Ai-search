# opensearch_client.py - Production Ready OpenSearch Client
# Version: 3.0.0 - Clean, Simple, and Reliable

import os
import logging
import time
import json
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
import threading
import random

try:
    from opensearchpy import OpenSearch, exceptions
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    logging.warning("opensearch-py not installed. Run: pip install opensearch-py")

# Setup logging
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 100
DEFAULT_INDEX = "innovai-default"

class OpenSearchClient:
    """
    Production-ready OpenSearch client with:
    - Connection pooling
    - Automatic retry logic
    - Health monitoring
    - Error handling
    - Configuration management
    """
    
    def __init__(self):
        self.client = None
        self.config = {}
        self.is_connected = False
        self.last_health_check = 0
        self.health_check_interval = 300  # 5 minutes
        self.connection_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "connection_errors": 0,
            "last_error": None,
            "last_success": None
        }
        
        # Initialize
        self._load_config()
        self._initialize_client()
    
    def _load_config(self):
        """Load configuration from environment variables"""
        # Get environment variables
        host = os.getenv("OPENSEARCH_HOST", "localhost")
        port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        user = os.getenv("OPENSEARCH_USER", "admin")
        password = os.getenv("OPENSEARCH_PASS", "admin")
        
        # Clean host (remove protocol if present)
        clean_host = host.replace("https://", "").replace("http://", "")
        
        # Determine if we should use SSL
        use_ssl = False
        if host.startswith("https://"):
            use_ssl = True
        elif any(keyword in clean_host.lower() for keyword in ["cloud", "digitalocean", "aws", "elastic"]):
            use_ssl = True
        elif port in [443, 9243, 25060]:  # Common SSL ports
            use_ssl = True
        
        # Build configuration
        self.config = {
            "host": clean_host,
            "port": port,
            "user": user,
            "password": password,
            "use_ssl": use_ssl,
            "url": f"{'https' if use_ssl else 'http'}://{clean_host}:{port}",
            "configured": bool(host and host != "localhost" and password != "admin")
        }
        
        logger.info(f"OpenSearch config loaded: {self.config['url']}")
    
    def _initialize_client(self):
        """Initialize OpenSearch client with production settings"""
        if not OPENSEARCH_AVAILABLE:
            logger.error("OpenSearch library not available")
            return False
        
        try:
            # Create client with robust settings
            self.client = OpenSearch(
                hosts=[{
                    'host': self.config['host'],
                    'port': self.config['port']
                }],
                http_auth=(self.config['user'], self.config['password']),
                use_ssl=self.config['use_ssl'],
                verify_certs=False,  # Common for managed services
                ssl_assert_hostname=False,
                ssl_show_warn=False,
                
                # Timeout settings
                timeout=DEFAULT_TIMEOUT,
                
                # Retry settings
                max_retries=DEFAULT_MAX_RETRIES,
                retry_on_timeout=True,
                retry_on_status={502, 503, 504},
                
                # Connection pool
                maxsize=20,
                
                # Performance
                http_compress=True,
                
                # Headers
                headers={"User-Agent": "InnovAI-Client/3.0.0"}
            )
            
            logger.info("OpenSearch client initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch client: {e}")
            self.client = None
            return False
    
    def test_connection(self, force_check: bool = False) -> bool:
        """Test OpenSearch connection with caching"""
        current_time = time.time()
        
        # Use cached result if recent
        if not force_check and self.is_connected:
            if current_time - self.last_health_check < self.health_check_interval:
                return True
        
        with self.connection_lock:
            if not self.client:
                if not self._initialize_client():
                    return False
            
            try:
                # Perform ping
                result = self.client.ping()
                
                if result:
                    self.is_connected = True
                    self.last_health_check = current_time
                    self.stats["last_success"] = datetime.now().isoformat()
                    logger.debug("OpenSearch connection successful")
                    return True
                else:
                    self._handle_connection_error("Ping returned False")
                    return False
                    
            except exceptions.ConnectionError as e:
                self._handle_connection_error(f"Connection error: {e}")
                return False
            except Exception as e:
                self._handle_connection_error(f"Unexpected error: {e}")
                return False
    
    def _handle_connection_error(self, error_msg: str):
        """Handle connection errors consistently"""
        self.is_connected = False
        self.stats["connection_errors"] += 1
        self.stats["last_error"] = error_msg
        logger.warning(f"OpenSearch connection failed: {error_msg}")
    
    def _retry_operation(self, operation_func, *args, **kwargs):
        """Retry operation with exponential backoff"""
        max_retries = DEFAULT_MAX_RETRIES
        
        for attempt in range(max_retries):
            try:
                return operation_func(*args, **kwargs)
            except exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Connection error, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                    continue
                else:
                    raise
            except Exception as e:
                # Don't retry non-connection errors
                raise
    
    def index_document(self, doc_id: str, body: Dict[str, Any], index_name: str = None) -> bool:
        """Index a document with retry logic"""
        if not index_name:
            index_name = DEFAULT_INDEX
        
        # Check connection first
        if not self.test_connection():
            raise Exception("OpenSearch not available")
        
        try:
            def _index_operation():
                return self.client.index(
                    index=index_name,
                    id=doc_id,
                    body=body,
                    refresh=False  # Don't force refresh for performance
                )
            
            result = self._retry_operation(_index_operation)
            
            # Update stats
            self.stats["total_operations"] += 1
            self.stats["successful_operations"] += 1
            
            logger.debug(f"Document indexed: {doc_id} -> {index_name}")
            return True
            
        except Exception as e:
            self.stats["total_operations"] += 1
            self.stats["failed_operations"] += 1
            self.stats["last_error"] = str(e)
            
            logger.error(f"Failed to index document {doc_id}: {e}")
            raise Exception(f"Indexing failed: {e}")
    
    def search(self, query: str, index_name: str = None, size: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Search documents with text query"""
        if not index_name:
            index_name = "*"  # Search all indices
        
        if not self.test_connection():
            logger.warning("OpenSearch not available for search")
            return []
        
        try:
            # Build search body
            body = {
                "size": size,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text", "content", "title", "metadata.*"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "highlight": {
                    "fields": {
                        "text": {"fragment_size": 150},
                        "content": {"fragment_size": 150}
                    }
                }
            }
            
            def _search_operation():
                return self.client.search(
                    index=index_name,
                    body=body,
                    **kwargs
                )
            
            response = self._retry_operation(_search_operation)
            hits = response.get("hits", {}).get("hits", [])
            
            # Update stats
            self.stats["total_operations"] += 1
            self.stats["successful_operations"] += 1
            
            logger.debug(f"Search completed: {query} -> {len(hits)} results")
            return hits
            
        except Exception as e:
            self.stats["total_operations"] += 1
            self.stats["failed_operations"] += 1
            self.stats["last_error"] = str(e)
            
            logger.error(f"Search failed: {e}")
            return []
    
    def vector_search(self, embedding: List[float], index_name: str = None, size: int = 10, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search documents using vector similarity"""
        if not index_name:
            index_name = "*"
        
        if not self.test_connection():
            logger.warning("OpenSearch not available for vector search")
            return []
        
        try:
            # Build filter clauses
            filter_clauses = []
            if filters:
                for field, value in filters.items():
                    if isinstance(value, list):
                        filter_clauses.append({"terms": {field: value}})
                    else:
                        filter_clauses.append({"term": {field: value}})
            
            # Build search body
            body = {
                "size": size,
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "filter": filter_clauses
                            } if filter_clauses else {"match_all": {}}
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
            
            def _vector_search_operation():
                return self.client.search(
                    index=index_name,
                    body=body
                )
            
            response = self._retry_operation(_vector_search_operation)
            hits = response.get("hits", {}).get("hits", [])
            
            # Update stats
            self.stats["total_operations"] += 1
            self.stats["successful_operations"] += 1
            
            logger.debug(f"Vector search completed: {len(embedding)} dims -> {len(hits)} results")
            return hits
            
        except Exception as e:
            self.stats["total_operations"] += 1
            self.stats["failed_operations"] += 1
            self.stats["last_error"] = str(e)
            
            logger.error(f"Vector search failed: {e}")
            return []
    
    def get_indices(self) -> List[str]:
        """Get list of available indices"""
        if not self.test_connection():
            return []
        
        try:
            indices = self.client.indices.get_alias(index="*")
            # Filter out system indices
            user_indices = [name for name in indices.keys() if not name.startswith('.')]
            return user_indices
        except Exception as e:
            logger.error(f"Failed to get indices: {e}")
            return []
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for a specific index"""
        if not self.test_connection():
            return {"status": "connection_failed"}
        
        try:
            # Check if index exists
            if not self.client.indices.exists(index=index_name):
                return {"status": "not_found", "index": index_name}
            
            # Get stats
            stats = self.client.indices.stats(index=index_name)
            doc_count = stats["_all"]["primaries"]["docs"]["count"]
            index_size = stats["_all"]["primaries"]["store"]["size_in_bytes"]
            
            return {
                "status": "healthy",
                "index": index_name,
                "document_count": doc_count,
                "size_bytes": index_size,
                "size_mb": round(index_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def create_index(self, index_name: str, mapping: Dict[str, Any] = None) -> bool:
        """Create an index with optional mapping"""
        if not self.test_connection():
            return False
        
        try:
            # Check if index already exists
            if self.client.indices.exists(index=index_name):
                logger.info(f"Index {index_name} already exists")
                return True
            
            # Create index
            body = {}
            if mapping:
                body["mappings"] = mapping
            
            self.client.indices.create(index=index_name, body=body)
            logger.info(f"Index created: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            return False
    
    def delete_index(self, index_name: str) -> bool:
        """Delete an index"""
        if not self.test_connection():
            return False
        
        try:
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                logger.info(f"Index deleted: {index_name}")
                return True
            else:
                logger.warning(f"Index {index_name} does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            return False
    
    def get_health(self) -> Dict[str, Any]:
        """Get comprehensive health information"""
        health_info = {
            "timestamp": datetime.now().isoformat(),
            "client_initialized": self.client is not None,
            "connection_status": "unknown",
            "config": self.config,
            "stats": self.stats
        }
        
        if self.test_connection():
            health_info["connection_status"] = "connected"
            
            # Try to get cluster health
            try:
                cluster_health = self.client.cluster.health()
                health_info["cluster_health"] = {
                    "status": cluster_health.get("status", "unknown"),
                    "number_of_nodes": cluster_health.get("number_of_nodes", 0),
                    "active_shards": cluster_health.get("active_shards", 0)
                }
            except Exception as e:
                health_info["cluster_health"] = {"error": str(e)}
        else:
            health_info["connection_status"] = "disconnected"
        
        return health_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        total_ops = self.stats["total_operations"]
        success_rate = 0.0
        
        if total_ops > 0:
            success_rate = (self.stats["successful_operations"] / total_ops) * 100
        
        return {
            **self.stats,
            "success_rate": f"{success_rate:.1f}%",
            "connection_healthy": self.is_connected,
            "config_valid": self.config.get("configured", False)
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "connection_errors": 0,
            "last_error": None,
            "last_success": None
        }
        logger.info("OpenSearch statistics reset")

# Global client instance
_client = None
_client_lock = threading.Lock()

def get_client() -> OpenSearchClient:
    """Get or create global OpenSearch client instance"""
    global _client
    
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenSearchClient()
    
    return _client

# Convenience functions for backward compatibility
def test_connection() -> bool:
    """Test OpenSearch connection"""
    return get_client().test_connection()

def get_connection_status() -> Dict[str, Any]:
    """Get connection status"""
    client = get_client()
    return {
        "tested": True,
        "connected": client.is_connected,
        "last_error": client.stats.get("last_error"),
        "last_test": client.last_health_check
    }

def get_opensearch_config() -> Dict[str, Any]:
    """Get OpenSearch configuration"""
    return get_client().config

def index_document(doc_id: str, body: Dict[str, Any], index_override: str = None) -> bool:
    """Index a document"""
    return get_client().index_document(doc_id, body, index_override)

def search_opensearch(query: str, index_override: str = None, size: int = 10) -> List[Dict[str, Any]]:
    """Search OpenSearch"""
    return get_client().search(query, index_override, size)

def search_vector(embedding: List[float], index_override: str = None, size: int = 10, 
                 filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Vector similarity search"""
    return get_client().vector_search(embedding, index_override, size, filters)

def get_opensearch_manager():
    """Get OpenSearch manager (for compatibility)"""
    return get_client()

def get_opensearch_stats() -> Dict[str, Any]:
    """Get OpenSearch statistics"""
    return get_client().get_stats()

def reset_opensearch_stats():
    """Reset OpenSearch statistics"""
    get_client().reset_stats()

def get_main_data_index() -> str:
    """Get the main data index name"""
    return "ai-corporate-sptr-test"

# Health check functions
def health_check() -> Dict[str, Any]:
    """Comprehensive health check"""
    return get_client().get_health()

def get_indices() -> List[str]:
    """Get list of available indices"""
    return get_client().get_indices()

def get_index_info(index_name: str) -> Dict[str, Any]:
    """Get index information"""
    return get_client().get_index_stats(index_name)

# Initialize on import
if __name__ == "__main__":
    # Test the client
    print("🔍 Testing OpenSearch Client...")
    
    client = get_client()
    
    # Test connection
    if client.test_connection():
        print("✅ Connection successful")
        
        # Get health
        health = client.get_health()
        print(f"📊 Health: {health['connection_status']}")
        
        # Get indices
        indices = client.get_indices()
        print(f"📁 Indices: {len(indices)} found")
        
        # Get stats
        stats = client.get_stats()
        print(f"📈 Stats: {stats['success_rate']} success rate")
        
    else:
        print("❌ Connection failed")
        health = client.get_health()
        print(f"❌ Error: {health['stats'].get('last_error', 'Unknown')}")
    
    print("🏁 Test complete")
else:
    try:
        # Initialize client on import
        client = get_client()
        logger.info("🔌 OpenSearch client v3.0.0 initialized")
        logger.info(f"   Target: {client.config['url']}")
        logger.info(f"   SSL: {client.config['use_ssl']}")
        logger.info(f"   Configured: {client.config['configured']}")
    except Exception as e:
        logger.error(f"❌ OpenSearch client initialization failed: {e}")