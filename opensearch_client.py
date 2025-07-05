# Enhanced opensearch_client.py - Evaluation Grouped Search (FIXED)
# Version: 4.0.1 - Fixed verify_certs attribute error

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

# Configuration with safe defaults
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "not_configured")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "443"))
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "")
OPENSEARCH_USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "true").lower() == "true"
OPENSEARCH_VERIFY_CERTS = os.getenv("OPENSEARCH_VERIFY_CERTS", "true").lower() == "true"
OPENSEARCH_TIMEOUT = int(os.getenv("OPENSEARCH_TIMEOUT", "30"))

# Default index for evaluation-grouped documents
DEFAULT_INDEX = "evaluations-grouped"

class OpenSearchManager:
    """
    ENHANCED: OpenSearch manager for evaluation-grouped document structure
    Handles template_ID-based collections with evaluation grouping
    """
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.last_test = None
        self.last_error = None
        self.host = OPENSEARCH_HOST
        self.port = OPENSEARCH_PORT
        self.use_ssl = OPENSEARCH_USE_SSL
        self.verify_certs = OPENSEARCH_VERIFY_CERTS  # FIXED: Added this line
        
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
        """Initialize OpenSearch client with enhanced configuration"""
        try:
            # Build connection URL
            protocol = "https" if self.use_ssl else "http"
            self.url = f"{protocol}://{self.host}:{self.port}"
            
            logger.info(f"🔗 Initializing OpenSearch client: {self.url}")
            
            # Client configuration
            client_config = {
                'hosts': [{'host': self.host, 'port': self.port}],
                'http_auth': (OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD) if OPENSEARCH_PASSWORD else None,
                'use_ssl': self.use_ssl,
                'verify_certs': self.verify_certs,  # FIXED: Now correctly references self.verify_certs
                'timeout': OPENSEARCH_TIMEOUT,
                'max_retries': 3,
                'retry_on_timeout': True
            }
            
            self.client = OpenSearch(**client_config)
            logger.info("✅ OpenSearch client initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenSearch client: {e}")
            self.last_error = str(e)
            self.client = None
    
    def test_connection(self) -> bool:
        """Test OpenSearch connection with enhanced error handling"""
        if not self.client or OPENSEARCH_HOST == "not_configured":
            self.last_error = "OpenSearch not configured"
            return False
        
        try:
            start_time = time.time()
            
            # Test basic connectivity
            info = self.client.info()
            
            # Test index operations
            test_index = "connection-test"
            
            # Try to create a test index
            if not self.client.indices.exists(index=test_index):
                self.client.indices.create(
                    index=test_index,
                    body={
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
                )
            
            # Test document operations
            test_doc = {
                "test_field": "connection_test",
                "timestamp": datetime.now().isoformat(),
                "structure_version": "4.0.1"
            }
            
            self.client.index(
                index=test_index,
                id="connection_test",
                body=test_doc
            )
            
            # Test search
            search_result = self.client.search(
                index=test_index,
                body={"query": {"match": {"test_field": "connection_test"}}}
            )
            
            # Clean up test index
            try:
                self.client.indices.delete(index=test_index)
            except:
                pass  # Ignore cleanup errors
            
            response_time = time.time() - start_time
            
            self.connected = True
            self.last_test = datetime.now().isoformat()
            self.last_error = None
            
            logger.info(f"✅ OpenSearch connection test passed ({response_time:.3f}s)")
            logger.info(f"📊 Cluster: {info.get('cluster_name', 'Unknown')}")
            logger.info(f"🔢 Version: {info.get('version', {}).get('number', 'Unknown')}")
            
            return True
            
        except Exception as e:
            self.connected = False
            self.last_test = datetime.now().isoformat()
            self.last_error = str(e)
            logger.error(f"❌ OpenSearch connection test failed: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            "connected": self.connected,
            "host": self.host,
            "port": self.port,
            "url": getattr(self, 'url', 'Not configured'),
            "last_test": self.last_test,
            "last_error": self.last_error,
            "tested": self.last_test is not None,
            "verify_certs": self.verify_certs,  # Added for debugging
            "use_ssl": self.use_ssl
        }
    
    def get_opensearch_config(self) -> Dict[str, Any]:
        """Get OpenSearch configuration"""
        return {
            "host": self.host,
            "port": self.port,
            "url": getattr(self, 'url', 'Not configured'),
            "use_ssl": self.use_ssl,
            "verify_certs": self.verify_certs,
            "username": OPENSEARCH_USERNAME,
            "password_set": bool(OPENSEARCH_PASSWORD),
            "timeout": OPENSEARCH_TIMEOUT
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
            document["_structure_version"] = "4.0.1"
            document["_document_type"] = "evaluation_grouped"
            
            # Index the complete evaluation document
            response = self.client.index(
                index=index_name,
                id=doc_id,
                body=document,
                refresh=True  # Make immediately searchable
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
                    "_document_type": {"type": "keyword"}
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
                body=search_body
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
    
    def search_evaluation_chunks(self, query: str, evaluation_id: str = None,
                               content_type: str = None, index_name: str = None) -> List[Dict]:
        """
        ENHANCED: Search within chunks of evaluations using nested queries
        """
        if not self.client:
            logger.error("❌ OpenSearch client not initialized")
            return []
        
        start_time = time.time()
        index_pattern = index_name or "eval-*"
        
        try:
            # Build nested query to search within chunks
            nested_query = {
                "nested": {
                    "path": "chunks",
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"chunks.text": query}}
                            ]
                        }
                    },
                    "inner_hits": {
                        "size": 10,
                        "highlight": {
                            "fields": {
                                "chunks.text": {}
                            }
                        }
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
            
            response = self.client.search(
                index=index_pattern,
                body=search_body
            )
            
            response_time = time.time() - start_time
            self._record_operation(True, response_time, f"search_chunks:{index_pattern}")
            
            # Process nested search results
            results = []
            
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                inner_hits = hit.get("inner_hits", {}).get("chunks", {}).get("hits", [])
                
                for inner_hit in inner_hits:
                    chunk_source = inner_hit.get("_source", {})
                    highlight = inner_hit.get("highlight", {})
                    
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
                        
                        # Highlighting
                        "highlight": highlight,
                        
                        # For backward compatibility
                        "_source": chunk_source
                    }
                    results.append(result)
            
            logger.info(f"🔍 Found {len(results)} chunk matches in {response_time:.3f}s")
            return results
            
        except Exception as e:
            response_time = time.time() - start_time
            self._record_operation(False, response_time, f"search_chunks:{index_pattern}")
            logger.error(f"❌ Chunk search failed: {e}")
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
    
    def get_evaluation_by_id(self, evaluation_id: str, index_name: str = None) -> Optional[Dict]:
        """Get a specific evaluation document by ID"""
        if not self.client:
            return None
        
        try:
            index_pattern = index_name or "eval-*"
            
            # Search by evaluationId field (not document _id)
            response = self.client.search(
                index=index_pattern,
                body={
                    "query": {"term": {"evaluationId": evaluation_id}},
                    "size": 1
                }
            )
            
            hits = response.get("hits", {}).get("hits", [])
            
            if hits:
                return hits[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get evaluation {evaluation_id}: {e}")
            return None
    
    def get_template_collections(self) -> List[Dict[str, Any]]:
        """Get list of template-based collections with statistics"""
        if not self.client:
            return []
        
        try:
            # Get all indices that start with 'eval-'
            indices = self.client.indices.get(index="eval-*")
            
            collections = []
            
            for index_name in indices.keys():
                try:
                    # Get index stats
                    stats = self.client.indices.stats(index=index_name)
                    doc_count = stats["_all"]["primaries"]["docs"]["count"]
                    
                    # Get sample document to extract template info
                    sample = self.client.search(
                        index=index_name,
                        body={"query": {"match_all": {}}, "size": 1}
                    )
                    
                    template_id = None
                    template_name = None
                    
                    if sample["hits"]["hits"]:
                        source = sample["hits"]["hits"][0]["_source"]
                        template_id = source.get("template_id")
                        template_name = source.get("template_name")
                    
                    collections.append({
                        "collection_name": index_name,
                        "template_id": template_id,
                        "template_name": template_name,
                        "document_count": doc_count,
                        "status": "active"
                    })
                    
                except Exception as e:
                    collections.append({
                        "collection_name": index_name,
                        "template_id": None,
                        "template_name": None,
                        "document_count": 0,
                        "status": "error",
                        "error": str(e)[:100]
                    })
            
            return collections
            
        except Exception as e:
            logger.error(f"❌ Failed to get template collections: {e}")
            return []
    
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
            "structure_version": "4.0.1"
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

def search_vector(query_vector: List[float], index_override: str = None, 
                 size: int = 10) -> List[Dict]:
    """
    ENHANCED: Vector search for evaluation documents
    """
    manager = get_opensearch_manager()
    
    if not manager.client:
        return []
    
    try:
        index_pattern = index_override or "eval-*"
        
        # Search using document-level embeddings
        search_body = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'document_embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            },
            "size": size
        }
        
        response = manager.client.search(
            index=index_pattern,
            body=search_body
        )
        
        return response.get("hits", {}).get("hits", [])
        
    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")
        return []

def search_evaluation_chunks(query: str, evaluation_id: str = None, 
                           content_type: str = None) -> List[Dict]:
    """
    NEW: Search within chunks of evaluations (for detailed analysis)
    """
    manager = get_opensearch_manager()
    return manager.search_evaluation_chunks(query, evaluation_id, content_type)

def get_evaluation_by_id(evaluation_id: str) -> Optional[Dict]:
    """
    NEW: Get specific evaluation by ID
    """
    manager = get_opensearch_manager()
    return manager.get_evaluation_by_id(evaluation_id)

def get_template_collections() -> List[Dict[str, Any]]:
    """
    NEW: Get list of template-based collections
    """
    manager = get_opensearch_manager()
    return manager.get_template_collections()

def get_opensearch_stats() -> Dict[str, Any]:
    """Get OpenSearch performance statistics"""
    manager = get_opensearch_manager()
    return manager.get_performance_stats()

# Health check function
def health_check() -> Dict[str, Any]:
    """Enhanced health check for evaluation-grouped structure"""
    try:
        manager = get_opensearch_manager()
        
        if OPENSEARCH_HOST == "not_configured":
            return {
                "status": "not_configured",
                "message": "OPENSEARCH_HOST environment variable not set"
            }
        
        # Test connection
        connection_ok = manager.test_connection()
        
        if connection_ok:
            # Get cluster info
            info = manager.client.info()
            
            # Get template collections
            collections = get_template_collections()
            
            return {
                "status": "healthy",
                "connected": True,
                "cluster_name": info.get("cluster_name", "Unknown"),
                "version": info.get("version", {}).get("number", "Unknown"),
                "host": manager.host,
                "port": manager.port,
                "url": getattr(manager, 'url', 'Unknown'),
                "template_collections": len(collections),
                "structure_version": "4.0.1",
                "document_structure": "evaluation_grouped",
                "performance": manager.get_performance_stats()
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": manager.last_error,
                "host": manager.host,
                "port": manager.port
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    # Test the enhanced OpenSearch client
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Fixed OpenSearch Client - Evaluation Grouped Structure")
    print("Expected structure: Template_ID collections with evaluation documents")
    
    # Health check
    health = health_check()
    print(f"\n🏥 Health check: {health['status']}")
    
    if health["status"] == "healthy":
        print(f"   Cluster: {health['cluster_name']}")
        print(f"   Version: {health['version']}")
        print(f"   Template Collections: {health['template_collections']}")
        print(f"   Structure Version: {health['structure_version']}")
        print(f"   Document Structure: {health['document_structure']}")
        
        # Test search
        print("\n🔍 Testing evaluation search...")
        results = search_opensearch("customer service", size=3)
        print(f"   Found {len(results)} evaluations")
        
        for i, result in enumerate(results[:2]):
            source = result.get("_source", {})
            print(f"   {i+1}. Evaluation {source.get('evaluationId', 'Unknown')}")
            print(f"      Template: {source.get('template_name', 'Unknown')}")
            print(f"      Chunks: {source.get('total_chunks', 0)}")
            print(f"      Score: {result.get('_score', 0):.3f}")
        
        # Test template collections
        print("\n📁 Testing template collections...")
        collections = get_template_collections()
        print(f"   Found {len(collections)} template-based collections")
        
        for collection in collections[:3]:
            print(f"   - {collection['collection_name']}")
            print(f"     Template: {collection['template_name']}")
            print(f"     Documents: {collection['document_count']}")
        
        print("\n✅ Fixed OpenSearch client is working with evaluation grouping!")
        
    else:
        print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
        print("Fix OpenSearch connection before running enhanced imports!")
    
    print("\n🏁 Testing complete!")
    print("💡 Fixed verify_certs attribute error - should connect properly now")