# opensearch_client.py - Production Merged: Working Auth + Enhanced Features
# Version: 4.0.3 - Merged your proven authentication with evaluation grouping

import os
import logging
import json
import time
import threading
import random
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

try:
    from opensearchpy import OpenSearch, exceptions
    from opensearchpy.exceptions import ConnectionError, RequestError
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    logging.warning("opensearch-py not installed. Run: pip install opensearch-py")

# Setup logging
logger = logging.getLogger(__name__)

# Configuration constants (from your working file)
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 100
DEFAULT_INDEX = "evaluations-grouped"

class OpenSearchManager:
    """
    PRODUCTION MERGED: Your proven working authentication + Enhanced evaluation grouping
    """
    
    def __init__(self):
        self.client = None
        self.config = {}
        self.connected = False
        self.last_test = None
        self.last_error = None
        self.connection_lock = threading.Lock()
        
        # Performance tracking
        self._performance_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_response_time": 0.0,
            "avg_response_time": 0.0,
            "last_operation": None,
            "connection_errors": 0,
            "last_success": None
        }
        
        # Load config and initialize
        self._load_config()
        self._initialize_client()
    
    def _load_config(self):
        """Load configuration using YOUR WORKING environment variable setup"""
        # Get environment variables (using YOUR working names)
        host = os.getenv("OPENSEARCH_HOST", "localhost")
        port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        user = os.getenv("OPENSEARCH_USER", "admin")
        password = os.getenv("OPENSEARCH_PASS", "admin")
        
        # Clean host (remove protocol if present) - from your working file
        clean_host = host.replace("https://", "").replace("http://", "")
        
        # Smart SSL detection (from your working version)
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
            "configured": bool(host and host != "localhost" and password and password != "admin")
        }
        
        logger.info(f"🔗 OpenSearch config loaded: {self.config['url']}")
        logger.info(f"   User: {user}")
        logger.info(f"   SSL: {use_ssl}")
        logger.info(f"   Configured: {self.config['configured']}")
    
    def _initialize_client(self):
        """Initialize OpenSearch client using YOUR PROVEN working setup"""
        if not OPENSEARCH_AVAILABLE:
            logger.error("❌ OpenSearch library not available")
            self.last_error = "OpenSearch library not installed"
            return False
        
        if not self.config.get("configured"):
            logger.warning("⚠️ OpenSearch not properly configured")
            self.last_error = "OpenSearch not configured"
            return False
        
        try:
            # Use YOUR EXACT working client configuration
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
                headers={"User-Agent": "InnovAI-Enhanced/4.0.3"}
            )
            
            logger.info("✅ OpenSearch client initialized using YOUR proven config")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenSearch client: {e}")
            self.client = None
            self.last_error = str(e)
            return False
    
    def test_connection(self) -> bool:
        """Test connection using YOUR proven working approach"""
        if not self.client or not self.config.get("configured"):
            self.last_error = "OpenSearch not configured or client not initialized"
            return False
        
        if not self.config["password"] or self.config["password"] == "admin":
            self.last_error = "OPENSEARCH_PASS not set or using default"
            logger.error("❌ OPENSEARCH_PASS environment variable is required")
            return False
        
        try:
            start_time = time.time()
            
            logger.info(f"🧪 Testing connection with YOUR proven config...")
            logger.info(f"   Host: {self.config['host']}")
            logger.info(f"   Port: {self.config['port']}")
            logger.info(f"   User: {self.config['user']}")
            logger.info(f"   SSL: {self.config['use_ssl']}")
            
            # Test basic connectivity (from your working file)
            try:
                info = self.client.info()
                logger.info(f"✅ Successfully connected to OpenSearch cluster")
                logger.info(f"   Cluster: {info.get('cluster_name', 'Unknown')}")
                logger.info(f"   Version: {info.get('version', {}).get('number', 'Unknown')}")
            except Exception as info_error:
                logger.error(f"❌ Cluster info failed: {info_error}")
                raise info_error
            
            # Test ping for good measure (from your working file)
            ping_result = self.client.ping()
            if not ping_result:
                raise Exception("Ping returned False")
            
            response_time = time.time() - start_time
            
            self.connected = True
            self.last_test = datetime.now().isoformat()
            self.last_error = None
            self._performance_stats["last_success"] = self.last_test
            
            logger.info(f"🎉 Connection test PASSED using YOUR working config ({response_time:.3f}s)")
            return True
            
        except Exception as e:
            self.connected = False
            self.last_test = datetime.now().isoformat()
            self.last_error = str(e)
            self._performance_stats["connection_errors"] += 1
            
            # Enhanced error reporting
            error_type = type(e).__name__
            error_message = str(e)
            
            logger.error(f"❌ Connection test FAILED:")
            logger.error(f"   Error Type: {error_type}")
            logger.error(f"   Error Message: {error_message}")
            
            # Provide specific guidance using YOUR variable names
            if "AuthenticationException" in error_type or "401" in error_message:
                logger.error("🔑 AUTHENTICATION ISSUE:")
                logger.error("   - Check OPENSEARCH_USER (should be 'doadmin' for Digital Ocean)")
                logger.error("   - Check OPENSEARCH_PASS is correct")
                logger.error("   - Verify credentials in Digital Ocean dashboard")
            elif "ConnectionError" in error_type or "timeout" in error_message.lower():
                logger.error("🌐 CONNECTION ISSUE:")
                logger.error("   - Check OPENSEARCH_HOST is correct")
                logger.error("   - Check OPENSEARCH_PORT (usually 25060 for Digital Ocean)")
                logger.error("   - Verify OpenSearch cluster is running")
            elif "SSLError" in error_type or "certificate" in error_message.lower():
                logger.error("🔒 SSL ISSUE:")
                logger.error("   - SSL auto-detection might have failed")
                logger.error("   - Try setting OPENSEARCH_USE_SSL environment variable")
            
            return False
    
    def _retry_operation(self, operation_func, *args, **kwargs):
        """Retry operation with exponential backoff (from your working file)"""
        max_retries = DEFAULT_MAX_RETRIES
        
        for attempt in range(max_retries):
            try:
                return operation_func(*args, **kwargs)
            except (ConnectionError, exceptions.ConnectionError) if OPENSEARCH_AVAILABLE else Exception as e:
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
            document["_structure_version"] = "4.0.3"
            document["_document_type"] = "evaluation_grouped"
            document["_auth_method"] = "proven_working_setup"
            
            # Index the complete evaluation document
            def _index_operation():
                return self.client.index(
                    index=index_name,
                    id=doc_id,
                    body=document,
                    refresh=True,  # Make immediately searchable
                    timeout=30
                )
            
            response = self._retry_operation(_index_operation)
            
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
                    "max_result_window": 10000,
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
                    "_auth_method": {"type": "keyword"}
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
            
            def _search_operation():
                return self.client.search(
                    index=index_pattern,
                    body=search_body,
                    timeout=30
                )
            
            response = self._retry_operation(_search_operation)
            
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
            
            def _search_chunks_operation():
                return self.client.search(
                    index=index_pattern,
                    body=search_body
                )
            
            response = self._retry_operation(_search_chunks_operation)
            
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
    
    def get_evaluation_by_id(self, evaluation_id: str, index_name: str = None) -> Optional[Dict]:
        """Get a specific evaluation document by ID"""
        if not self.client:
            return None
        
        try:
            index_pattern = index_name or "eval-*"
            
            # Search by evaluationId field (not document _id)
            def _get_eval_operation():
                return self.client.search(
                    index=index_pattern,
                    body={
                        "query": {"term": {"evaluationId": evaluation_id}},
                        "size": 1
                    }
                )
            
            response = self._retry_operation(_get_eval_operation)
            
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
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status using YOUR working config structure"""
        return {
            "connected": self.connected,
            "host": self.config["host"],
            "port": self.config["port"],
            "url": self.config["url"],
            "user": self.config["user"],
            "last_test": self.last_test,
            "last_error": self.last_error,
            "tested": self.last_test is not None,
            "use_ssl": self.config["use_ssl"],
            "provider": "Digital Ocean" if "digitalocean" in self.config["host"] or "do-" in self.config["host"] else "Standard",
            "auth_configured": bool(self.config["user"] and self.config["password"] != "admin"),
            "config_valid": self.config["configured"]
        }
    
    def get_opensearch_config(self) -> Dict[str, Any]:
        """Get OpenSearch configuration using YOUR working setup"""
        return {
            "host": self.config["host"],
            "port": self.config["port"],
            "url": self.config["url"],
            "user": self.config["user"],
            "password_set": bool(self.config["password"] and self.config["password"] != "admin"),
            "use_ssl": self.config["use_ssl"],
            "timeout": DEFAULT_TIMEOUT,
            "provider": "Digital Ocean" if "digitalocean" in self.config["host"] or "do-" in self.config["host"] else "Standard",
            "default_port": 25060,
            "default_user": "doadmin",
            "env_user_var": "OPENSEARCH_USER",
            "env_pass_var": "OPENSEARCH_PASS",
            "configured": self.config["configured"]
        }
    
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
            "structure_version": "4.0.3",
            "auth_method": "proven_working_setup",
            "provider": "Digital Ocean" if "digitalocean" in self.config["host"] or "do-" in self.config["host"] else "Standard"
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
        
        def _vector_search_operation():
            return manager.client.search(
                index=index_pattern,
                body=search_body
            )
        
        response = manager._retry_operation(_vector_search_operation)
        
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
    """Enhanced health check using YOUR proven working authentication"""
    try:
        manager = get_opensearch_manager()
        
        if not manager.config.get("configured"):
            return {
                "status": "not_configured",
                "message": "OPENSEARCH_HOST or credentials not properly set",
                "provider": "unknown"
            }
        
        # Test connection using YOUR proven method
        connection_ok = manager.test_connection()
        
        if connection_ok:
            # Get cluster info
            info = manager.client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "cluster_name": info.get("cluster_name", "Unknown"),
                "version": info.get("version", {}).get("number", "Unknown"),
                "host": manager.config["host"],
                "port": manager.config["port"],
                "url": manager.config["url"],
                "user": manager.config["user"],
                "structure_version": "4.0.3",
                "document_structure": "evaluation_grouped",
                "provider": "Digital Ocean" if "digitalocean" in manager.config["host"] or "do-" in manager.config["host"] else "Standard",
                "performance": manager.get_performance_stats(),
                "auth_method": "proven_working_setup"
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": manager.last_error,
                "host": manager.config["host"],
                "port": manager.config["port"],
                "user": manager.config["user"],
                "provider": "Digital Ocean" if "digitalocean" in manager.config["host"] or "do-" in manager.config["host"] else "Standard",
                "auth_method": "proven_working_setup"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "provider": "unknown",
            "auth_method": "proven_working_setup"
        }

if __name__ == "__main__":
    # Test the proven working Digital Ocean OpenSearch client + Enhanced features
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing PRODUCTION MERGED OpenSearch Client")
    print("Expected: YOUR proven authentication + Enhanced evaluation grouping")
    
    # Health check
    health = health_check()
    print(f"\n🏥 Health check: {health['status']}")
    print(f"   Provider: {health.get('provider', 'Unknown')}")
    print(f"   Auth Method: {health.get('auth_method', 'Unknown')}")
    
    if health["status"] == "healthy":
        print(f"   Cluster: {health['cluster_name']}")
        print(f"   Version: {health['version']}")
        print(f"   User: {health['user']}")
        print(f"   Structure Version: {health['structure_version']}")
        print(f"   Document Structure: {health['document_structure']}")
        
        print("\n✅ YOUR proven authentication + Enhanced features working!")
        
    else:
        print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
        print("\n🔧 Digital Ocean Troubleshooting with YOUR Variable Names:")
        print("   1. Check OPENSEARCH_USER=doadmin")
        print("   2. Check OPENSEARCH_PASS is correct")
        print("   3. Verify OPENSEARCH_HOST is correct")
        print("   4. SSL is auto-detected")
        print("   5. Check Digital Ocean cluster status")
    
    print("\n🏁 Testing complete!")
else:
    try:
        # Initialize client on import
        client = get_opensearch_manager()
        logger.info("🔌 PRODUCTION MERGED OpenSearch client v4.0.3 initialized")
        logger.info(f"   Target: {client.config['url']}")
        logger.info(f"   SSL: {client.config['use_ssl']}")
        logger.info(f"   Configured: {client.config['configured']}")
        logger.info(f"   Auth: YOUR proven working setup")
        logger.info(f"   Features: Evaluation grouping + Template_ID collections")
    except Exception as e:
        logger.error(f"❌ PRODUCTION MERGED OpenSearch client initialization failed: {e}")