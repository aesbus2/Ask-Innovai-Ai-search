# Enhanced opensearch_client.py - Digital Ocean Authentication Fixed
# Version: 4.0.2 - Fixed Digital Ocean OpenSearch authentication

import os
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from opensearchpy import OpenSearch
from opensearchpy.exceptions import ConnectionError, RequestError

# Setup logging
logger = logging.getLogger("ask-innovai.opensearch")

# Configuration with Digital Ocean defaults
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "not_configured")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "25060"))  # Digital Ocean default
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME", "doadmin")  # Digital Ocean default
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "")
OPENSEARCH_USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "true").lower() == "true"
OPENSEARCH_VERIFY_CERTS = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"  # Digital Ocean default
OPENSEARCH_TIMEOUT = int(os.getenv("OPENSEARCH_TIMEOUT", "30"))

# Default index for evaluation-grouped documents
DEFAULT_INDEX = "evaluations-grouped"

class OpenSearchManager:
    """
    ENHANCED: OpenSearch manager with Digital Ocean authentication support
    """
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.last_test = None
        self.last_error = None
        self.host = OPENSEARCH_HOST
        self.port = OPENSEARCH_PORT
        self.use_ssl = OPENSEARCH_USE_SSL
        self.verify_certs = OPENSEARCH_VERIFY_CERTS
        self.username = OPENSEARCH_USERNAME
        self.password = OPENSEARCH_PASSWORD
        
        # Performance tracking
        self._performance_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_response_time": 0.0,
            "avg_response_time": 0.0,
            "last_operation": None
        }
        
        if OPENSEARCH_HOST != "not_configured":
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenSearch client with Digital Ocean configuration"""
        try:
            # Build connection URL
            protocol = "https" if self.use_ssl else "http"
            self.url = f"{protocol}://{self.host}:{self.port}"
            
            logger.info(f"🔗 Initializing Digital Ocean OpenSearch client: {self.url}")
            logger.info(f"   Username: {self.username}")
            logger.info(f"   SSL: {self.use_ssl}")
            logger.info(f"   Verify Certs: {self.verify_certs}")
            
            # Digital Ocean OpenSearch client configuration
            client_config = {
                'hosts': [self.url],  # Use full URL format for Digital Ocean
                'timeout': OPENSEARCH_TIMEOUT,
                'max_retries': 3,
                'retry_on_timeout': True,
                'use_ssl': self.use_ssl,
                'verify_certs': self.verify_certs,
                'ssl_show_warn': False,  # Reduce SSL warnings
            }
            
            # Add authentication if credentials are provided
            if self.username and self.password:
                client_config['http_auth'] = (self.username, self.password)
                logger.info("✅ Using HTTP Basic Auth")
            else:
                logger.warning("⚠️ No authentication credentials provided")
            
            # Digital Ocean specific settings
            if "do-ai" in self.host or "digitalocean" in self.host:
                logger.info("🌊 Detected Digital Ocean OpenSearch - applying optimizations")
                client_config.update({
                    'verify_certs': False,  # Digital Ocean certificates can be tricky
                    'ssl_show_warn': False,
                    'connection_class': None,  # Use default connection class
                })
            
            self.client = OpenSearch(**client_config)
            logger.info("✅ Digital Ocean OpenSearch client initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Digital Ocean OpenSearch client: {e}")
            self.last_error = str(e)
            self.client = None
    
    def test_connection(self) -> bool:
        """Test Digital Ocean OpenSearch connection with enhanced error handling"""
        if not self.client or OPENSEARCH_HOST == "not_configured":
            self.last_error = "OpenSearch not configured"
            return False
        
        if not self.password:
            self.last_error = "OPENSEARCH_PASSWORD not set"
            logger.error("❌ OPENSEARCH_PASSWORD environment variable is required")
            return False
        
        try:
            start_time = time.time()
            
            logger.info(f"🧪 Testing Digital Ocean OpenSearch connection...")
            logger.info(f"   Host: {self.host}")
            logger.info(f"   Port: {self.port}")
            logger.info(f"   Username: {self.username}")
            logger.info(f"   SSL: {self.use_ssl}")
            logger.info(f"   Verify Certs: {self.verify_certs}")
            
            # Test basic connectivity with detailed error handling
            try:
                info = self.client.info()
                logger.info(f"✅ Successfully connected to OpenSearch cluster")
                logger.info(f"   Cluster: {info.get('cluster_name', 'Unknown')}")
                logger.info(f"   Version: {info.get('version', {}).get('number', 'Unknown')}")
            except Exception as info_error:
                logger.error(f"❌ Cluster info failed: {info_error}")
                raise info_error
            
            # Test index operations with a simple test
            test_index = "connection-test"
            
            try:
                # Clean up any existing test index first
                if self.client.indices.exists(index=test_index):
                    self.client.indices.delete(index=test_index)
                
                # Create a simple test index
                index_body = {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    },
                    "mappings": {
                        "properties": {
                            "test_field": {"type": "text"},
                            "timestamp": {"type": "date"}
                        }
                    }
                }
                
                self.client.indices.create(index=test_index, body=index_body)
                logger.info(f"✅ Test index created successfully")
                
                # Test document operations
                test_doc = {
                    "test_field": "digital_ocean_connection_test",
                    "timestamp": datetime.now().isoformat(),
                    "structure_version": "4.0.2"
                }
                
                # Index a test document
                index_response = self.client.index(
                    index=test_index,
                    id="do_connection_test",
                    body=test_doc,
                    refresh=True
                )
                logger.info(f"✅ Test document indexed successfully")
                
                # Test search functionality
                search_response = self.client.search(
                    index=test_index,
                    body={"query": {"match": {"test_field": "digital_ocean_connection_test"}}}
                )
                
                hits = search_response.get("hits", {}).get("total", {})
                if isinstance(hits, dict):
                    total_hits = hits.get("value", 0)
                else:
                    total_hits = hits
                
                if total_hits > 0:
                    logger.info(f"✅ Search test successful ({total_hits} results)")
                else:
                    logger.warning("⚠️ Search returned no results")
                
                # Clean up test index
                self.client.indices.delete(index=test_index)
                logger.info(f"✅ Test cleanup completed")
                
            except Exception as ops_error:
                logger.error(f"❌ Index operations failed: {ops_error}")
                # Try to clean up
                try:
                    self.client.indices.delete(index=test_index)
                except:
                    pass
                raise ops_error
            
            response_time = time.time() - start_time
            
            self.connected = True
            self.last_test = datetime.now().isoformat()
            self.last_error = None
            
            logger.info(f"🎉 Digital Ocean OpenSearch connection test PASSED ({response_time:.3f}s)")
            return True
            
        except Exception as e:
            self.connected = False
            self.last_test = datetime.now().isoformat()
            self.last_error = str(e)
            
            # Enhanced error reporting for Digital Ocean
            error_type = type(e).__name__
            error_message = str(e)
            
            logger.error(f"❌ Digital Ocean OpenSearch connection test FAILED:")
            logger.error(f"   Error Type: {error_type}")
            logger.error(f"   Error Message: {error_message}")
            
            # Provide specific guidance based on error type
            if "AuthenticationException" in error_type or "401" in error_message:
                logger.error("🔑 AUTHENTICATION ISSUE:")
                logger.error("   - Check OPENSEARCH_USERNAME (should be 'doadmin' for Digital Ocean)")
                logger.error("   - Check OPENSEARCH_PASSWORD is correct")
                logger.error("   - Verify credentials in Digital Ocean dashboard")
            elif "ConnectionError" in error_type or "timeout" in error_message.lower():
                logger.error("🌐 CONNECTION ISSUE:")
                logger.error("   - Check OPENSEARCH_HOST is correct")
                logger.error("   - Check OPENSEARCH_PORT (usually 25060 for Digital Ocean)")
                logger.error("   - Verify OpenSearch cluster is running")
            elif "SSLError" in error_type or "certificate" in error_message.lower():
                logger.error("🔒 SSL ISSUE:")
                logger.error("   - Try setting OPENSEARCH_VERIFY_CERTS=false")
                logger.error("   - Check SSL configuration")
            else:
                logger.error("❓ UNKNOWN ISSUE:")
                logger.error("   - Check all environment variables")
                logger.error("   - Verify Digital Ocean OpenSearch cluster status")
            
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status with Digital Ocean details"""
        return {
            "connected": self.connected,
            "host": self.host,
            "port": self.port,
            "url": getattr(self, 'url', 'Not configured'),
            "username": self.username,
            "last_test": self.last_test,
            "last_error": self.last_error,
            "tested": self.last_test is not None,
            "verify_certs": self.verify_certs,
            "use_ssl": self.use_ssl,
            "provider": "Digital Ocean" if "digitalocean" in self.host or "do-" in self.host else "Standard",
            "auth_configured": bool(self.username and self.password)
        }
    
    def get_opensearch_config(self) -> Dict[str, Any]:
        """Get OpenSearch configuration with Digital Ocean specifics"""
        return {
            "host": self.host,
            "port": self.port,
            "url": getattr(self, 'url', 'Not configured'),
            "username": self.username,
            "password_set": bool(self.password),
            "use_ssl": self.use_ssl,
            "verify_certs": self.verify_certs,
            "timeout": OPENSEARCH_TIMEOUT,
            "provider": "Digital Ocean" if "digitalocean" in self.host or "do-" in self.host else "Standard",
            "default_port": 25060,
            "default_username": "doadmin"
        }
    
    def _record_operation(self, success: bool, response_time: float, operation: str):
        """Record operation statistics"""
        self._performance_stats["total_operations"] += 1
        self._performance_stats["total_response_time"] += response_time
        self._performance_stats["last_operation"] = operation
        
        if success:
            self._performance_stats["successful_operations"] += 1
        else:
            self._performance_stats["failed_operations"] += 1
        
        # Calculate average
        total_ops = self._performance_stats["total_operations"]
        if total_ops > 0:
            self._performance_stats["avg_response_time"] = (
                self._performance_stats["total_response_time"] / total_ops
            )
    
    def index_evaluation_document(self, doc_id: str, document: Dict[str, Any], 
                                index_name: str = None) -> bool:
        """
        ENHANCED: Index a complete evaluation document with grouped chunks
        """
        if not self.client:
            logger.error("❌ OpenSearch client not initialized")
            return False
        
        start_time = time.time()
        index_name = index_name or DEFAULT_INDEX
        
        try:
            # Ensure the index exists with proper mapping for evaluation documents
            self._ensure_evaluation_index_exists(index_name)
            
            # Validate document structure
            if not self._validate_evaluation_document(document):
                logger.error(f"❌ Invalid evaluation document structure for {doc_id}")
                return False
            
            # Add indexing metadata
            document["_indexed_at"] = datetime.now().isoformat()
            document["_structure_version"] = "4.0.2"
            document["_document_type"] = "evaluation_grouped"
            document["_provider"] = "digital_ocean"
            
            # Index the complete evaluation document with Digital Ocean optimizations
            response = self.client.index(
                index=index_name,
                id=doc_id,
                body=document,
                refresh=True,  # Make immediately searchable
                timeout=30
            )
            
            response_time = time.time() - start_time
            self._record_operation(True, response_time, f"index_evaluation:{index_name}")
            
            logger.info(f"✅ Indexed evaluation {doc_id} in {index_name} ({response_time:.3f}s)")
            logger.info(f"   📄 Chunks: {document.get('total_chunks', 0)}")
            logger.info(f"   📋 Template: {document.get('template_name', 'Unknown')}")
            
            return True
            
        except Exception as e:
            response_time = time.time() - start_time
            self._record_operation(False, response_time, f"index_evaluation:{index_name}")
            logger.error(f"❌ Failed to index evaluation {doc_id}: {e}")
            return False
    
    def _ensure_evaluation_index_exists(self, index_name: str):
        """
        ENHANCED: Ensure index exists with proper mapping for evaluation documents
        """
        if self.client.indices.exists(index=index_name):
            return
        
        # Enhanced mapping for evaluation-grouped documents with Digital Ocean optimizations
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,  # Single replica for Digital Ocean
                "analysis": {
                    "analyzer": {
                        "evaluation_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"]
                        }
                    }
                },
                "index": {
                    "max_result_window": 10000,  # Digital Ocean compatibility
                    "mapping": {
                        "ignore_malformed": True
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
                            "embedding": {"type": "dense_vector", "dims": 384},
                            
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
                    
                    # Metadata with enhanced structure
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
                    "collection_source": {"type": "keyword"},
                    "_indexed_at": {"type": "date"},
                    "_structure_version": {"type": "keyword"},
                    "_document_type": {"type": "keyword"},
                    "_provider": {"type": "keyword"}
                }
            }
        }
        
        try:
            self.client.indices.create(index=index_name, body=mapping)
            logger.info(f"✅ Created evaluation index: {index_name}")
        except Exception as e:
            logger.error(f"❌ Failed to create index {index_name}: {e}")
            raise
    
    def _validate_evaluation_document(self, document: Dict[str, Any]) -> bool:
        """Validate evaluation document structure"""
        required_fields = ["evaluationId", "template_id", "chunks", "metadata"]
        
        for field in required_fields:
            if field not in document:
                logger.error(f"❌ Missing required field: {field}")
                return False
        
        if not isinstance(document["chunks"], list):
            logger.error("❌ Chunks must be a list")
            return False
        
        if len(document["chunks"]) == 0:
            logger.error("❌ Document must have at least one chunk")
            return False
        
        return True
    
    def search_evaluations(self, query: str, index_name: str = None, 
                         filters: Dict[str, Any] = None, size: int = 10) -> List[Dict]:
        """
        ENHANCED: Search evaluation documents (returns evaluations, not chunks)
        """
        if not self.client:
            logger.error("❌ OpenSearch client not initialized")
            return []
        
        start_time = time.time()
        index_pattern = index_name or "eval-*"
        
        try:
            # Build search query for evaluation documents
            search_body = self._build_evaluation_search_query(query, filters, size)
            
            response = self.client.search(
                index=index_pattern,
                body=search_body,
                timeout=30  # Digital Ocean timeout
            )
            
            response_time = time.time() - start_time
            self._record_operation(True, response_time, f"search_evaluations:{index_pattern}")
            
            hits = response.get("hits", {}).get("hits", [])
            total_hits = response.get("hits", {}).get("total", {})
            
            if isinstance(total_hits, dict):
                total_count = total_hits.get("value", 0)
            else:
                total_count = total_hits
            
            logger.info(f"🔍 Found {total_count} evaluations in {response_time:.3f}s")
            
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
                    
                    # For backward compatibility, create a "text" field from full_text
                    "text": source.get("full_text", "")[:500]  # First 500 chars for preview
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            response_time = time.time() - start_time
            self._record_operation(False, response_time, f"search_evaluations:{index_pattern}")
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def _build_evaluation_search_query(self, query: str, filters: Dict[str, Any] = None, 
                                     size: int = 10) -> Dict[str, Any]:
        """
        Build search query for evaluation documents
        """
        # Main text search across multiple fields
        text_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "full_text^2",           # Boost full text
                    "evaluation_text^1.5",   # Boost evaluation content
                    "transcript_text",
                    "template_name^1.2",     # Boost template name
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
                "should": [
                    text_query,
                    nested_query
                ],
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
            
            # Duration filter
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
        
        # Build final search body
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
                    "evaluation_text": {"fragment_size": 150, "number_of_fragments": 1},
                    "transcript_text": {"fragment_size": 150, "number_of_fragments": 1}
                }
            }
        }
        
        return search_body
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get OpenSearch performance statistics"""
        success_rate = 0.0
        if self._performance_stats["total_operations"] > 0:
            success_rate = (
                self._performance_stats["successful_operations"] / 
                self._performance_stats["total_operations"]
            )
        
        return {
            **self._performance_stats,
            "success_rate": f"{success_rate:.2%}",
            "connected": self.connected,
            "structure_version": "4.0.2",
            "provider": "Digital Ocean" if "digitalocean" in self.host or "do-" in self.host else "Standard"
        }

# Global manager instance
_opensearch_manager = None

def get_opensearch_manager() -> OpenSearchManager:
    """Get or create the global OpenSearch manager"""
    global _opensearch_manager
    if _opensearch_manager is None:
        _opensearch_manager = OpenSearchManager()
    return _opensearch_manager

# Enhanced convenience functions for backward compatibility
def test_connection() -> bool:
    """Test OpenSearch connection"""
    manager = get_opensearch_manager()
    return manager.test_connection()

def get_connection_status() -> Dict[str, Any]:
    """Get connection status"""
    manager = get_opensearch_manager()
    return manager.get_connection_status()

def get_opensearch_config() -> Dict[str, Any]:
    """Get OpenSearch configuration"""
    manager = get_opensearch_manager()
    return manager.get_opensearch_config()

def index_document(doc_id: str, document: Dict[str, Any], index_override: str = None) -> bool:
    """
    ENHANCED: Index evaluation document (now groups chunks within evaluation)
    """
    manager = get_opensearch_manager()
    return manager.index_evaluation_document(doc_id, document, index_override)

def search_opensearch(query: str, index_override: str = None, 
                     filters: Dict[str, Any] = None, size: int = 10) -> List[Dict]:
    """
    ENHANCED: Search evaluations (returns evaluation documents, not chunks)
    """
    manager = get_opensearch_manager()
    return manager.search_evaluations(query, index_override, filters, size)

# Health check function
def health_check() -> Dict[str, Any]:
    """Enhanced health check for Digital Ocean OpenSearch"""
    try:
        manager = get_opensearch_manager()
        
        if OPENSEARCH_HOST == "not_configured":
            return {
                "status": "not_configured",
                "message": "OPENSEARCH_HOST environment variable not set",
                "provider": "unknown"
            }
        
        # Test connection
        connection_ok = manager.test_connection()
        
        if connection_ok:
            # Get cluster info
            info = manager.client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "cluster_name": info.get("cluster_name", "Unknown"),
                "version": info.get("version", {}).get("number", "Unknown"),
                "host": manager.host,
                "port": manager.port,
                "url": getattr(manager, 'url', 'Unknown'),
                "username": manager.username,
                "structure_version": "4.0.2",
                "document_structure": "evaluation_grouped",
                "provider": "Digital Ocean" if "digitalocean" in manager.host or "do-" in manager.host else "Standard",
                "performance": manager.get_performance_stats()
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": manager.last_error,
                "host": manager.host,
                "port": manager.port,
                "username": manager.username,
                "provider": "Digital Ocean" if "digitalocean" in manager.host or "do-" in manager.host else "Standard"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "provider": "unknown"
        }

if __name__ == "__main__":
    # Test the Digital Ocean OpenSearch client
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Digital Ocean OpenSearch Client")
    print("Expected: Digital Ocean managed OpenSearch authentication")
    
    # Health check
    health = health_check()
    print(f"\n🏥 Health check: {health['status']}")
    print(f"   Provider: {health.get('provider', 'Unknown')}")
    
    if health["status"] == "healthy":
        print(f"   Cluster: {health['cluster_name']}")
        print(f"   Version: {health['version']}")
        print(f"   Username: {health['username']}")
        print(f"   Structure Version: {health['structure_version']}")
        
        print("\n✅ Digital Ocean OpenSearch client is working!")
        
    else:
        print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
        print("\n🔧 Digital Ocean OpenSearch Troubleshooting:")
        print("   1. Check OPENSEARCH_USERNAME=doadmin")
        print("   2. Check OPENSEARCH_PASSWORD is correct")
        print("   3. Verify OPENSEARCH_HOST is correct")
        print("   4. Try OPENSEARCH_VERIFY_CERTS=false")
        print("   5. Check Digital Ocean cluster status")
    
    print("\n🏁 Testing complete!")