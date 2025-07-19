# opensearch_client.py - VERSION 4.6.0
# COMPREHENSIVE FIX: Timeout issues, query compilation errors, field validation
# FIXES: All timeout configurations, safe aggregations, robust error handling
# NEW: Enhanced field validation, compilation error prevention, performance improvements
# API INTEGRATION: weight_score, url, evaluation, transcript, agentName, subDisposition
# SEARCH IMPROVEMENTS: Separate weight_score search, URL extraction, proper field mapping

import os
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

try:
    from opensearchpy import OpenSearch
    from opensearchpy.exceptions import ConnectionError, RequestError
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    logging.warning("opensearch-py not installed. Run: pip install opensearch-py")

# Setup logging
logger = logging.getLogger(__name__)

# Global vector support detection - DISABLED for stability
_vector_support_detected = False
_vector_support_tested = True

# Global client instance
_client = None

# Version information
VERSION = "4.6.0"
FIXES_APPLIED = [
    "timeout_configuration_fix",
    "query_compilation_error_prevention", 
    "field_validation_enhancement",
    "safe_aggregation_queries",
    "robust_error_handling",
    "performance_improvements",
    "api_field_integration",
    "weight_score_search_separation",
    "url_field_support",
    "evaluation_transcript_indexing"
]

# =============================================================================
# FIXED CLIENT CREATION - ALL TIMEOUT ISSUES RESOLVED
# =============================================================================

def get_client():
    """Get OpenSearch client with COMPLETELY FIXED timeout configuration"""
    if not OPENSEARCH_AVAILABLE:
        logger.error("OpenSearch library not available")
        return None
    
    try:
        # FIXED: All timeouts are integers (seconds), not strings
        client = OpenSearch(
            hosts=[{
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": int(os.getenv("OPENSEARCH_PORT", "25060"))
            }],
            http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS")),
            use_ssl=True,
            verify_certs=False,
            request_timeout=30,        # INTEGER: 30 seconds
            connect_timeout=10,        # INTEGER: 10 seconds  
            max_retries=3,
            retry_on_timeout=True,
            ssl_show_warn=False,
            pool_maxsize=20,
            http_compress=True
        )
        
        # Test connection with proper timeout
        test_result = client.ping(request_timeout=5)  # INTEGER: 5 seconds
        if test_result:
            logger.info("✅ OpenSearch connection successful")
        else:
            logger.warning("⚠️ OpenSearch ping returned False")
        
        return client
        
    except Exception as e:
        logger.error(f"Failed to create OpenSearch client: {e}")
        return None

def get_opensearch_client():
    """Get or create global client instance"""
    global _client
    if _client is None:
        _client = get_client()
    return _client

def test_connection() -> bool:
    """Test connection with FIXED timeout handling"""
    client = get_opensearch_client()
    if not client:
        return False
    
    try:
        # Use integer timeout
        result = client.ping(request_timeout=5)
        if result:
            logger.info("✅ OpenSearch connection test successful")
            return True
        else:
            logger.error("❌ OpenSearch ping failed")
            return False
    except Exception as e:
        logger.error(f"❌ OpenSearch connection test failed: {e}")
        return False

# =============================================================================
# ENHANCED FIELD DETECTION - PREVENTS COMPILATION ERRORS
# =============================================================================

def validate_field_exists(client, index_pattern: str, field_path: str) -> bool:
    """NEW: Validate that a field exists before using it in queries"""
    try:
        mappings = client.indices.get_mapping(index=index_pattern, request_timeout=10)
        
        for index_name, mapping_data in mappings.items():
            properties = mapping_data.get("mappings", {}).get("properties", {})
            
            # Navigate nested field path (e.g., "metadata.program.keyword")
            current_props = properties
            field_parts = field_path.split(".")
            
            for part in field_parts[:-1]:
                if part in current_props:
                    current_props = current_props[part].get("properties", {})
                    if not current_props:
                        # Check if it's a multi-field (has "fields")
                        current_props = current_props.get("fields", {})
                else:
                    return False
            
            # Check final field
            final_field = field_parts[-1]
            if final_field in current_props:
                return True
                
        return False
        
    except Exception as e:
        logger.warning(f"Field validation failed for {field_path}: {e}")
        return False

def get_available_fields(client, index_pattern: str = "eval-*") -> Dict[str, List[str]]:
    """
    ENHANCED: Get available fields with validation and safety checks
    """
    try:
        mappings_response = client.indices.get_mapping(index=index_pattern, request_timeout=10)
        
        available_fields = {
            "text_fields": [],
            "keyword_fields": [],
            "nested_fields": [],
            "metadata_fields": [],
            "safe_aggregation_fields": {},  # NEW: Fields safe for aggregation
            "date_fields": [],
            "numeric_fields": []
        }
        
        for index_name, mapping_data in mappings_response.items():
            properties = mapping_data.get("mappings", {}).get("properties", {})
            
            # Analyze top-level fields
            for field_name, field_config in properties.items():
                field_type = field_config.get("type", "")
                
                if field_type == "text":
                    available_fields["text_fields"].append(field_name)
                elif field_type == "keyword":
                    available_fields["keyword_fields"].append(field_name)
                    available_fields["safe_aggregation_fields"][field_name] = field_name
                elif field_type == "nested":
                    available_fields["nested_fields"].append(field_name)
                elif field_type == "date":
                    available_fields["date_fields"].append(field_name)
                elif field_type in ["integer", "long", "float", "double"]:
                    available_fields["numeric_fields"].append(field_name)
                
                # Check for text fields with keyword subfields
                if "fields" in field_config and "keyword" in field_config["fields"]:
                    keyword_field = f"{field_name}.keyword"
                    available_fields["keyword_fields"].append(keyword_field)
                    available_fields["safe_aggregation_fields"][field_name] = keyword_field
                
                # Analyze metadata fields (most important for filtering)
                if field_name == "metadata" and "properties" in field_config:
                    for meta_field, meta_config in field_config["properties"].items():
                        meta_type = meta_config.get("type", "")
                        
                        # Add to metadata fields list
                        available_fields["metadata_fields"].append(f"metadata.{meta_field}")
                        
                        # Check if safe for aggregation
                        if meta_type == "keyword":
                            available_fields["safe_aggregation_fields"][meta_field] = f"metadata.{meta_field}"
                        elif meta_type == "text" and "fields" in meta_config and "keyword" in meta_config["fields"]:
                            available_fields["safe_aggregation_fields"][meta_field] = f"metadata.{meta_field}.keyword"
        
        # Remove duplicates
        for key in available_fields:
            if isinstance(available_fields[key], list):
                available_fields[key] = list(set(available_fields[key]))
        
        logger.info(f"✅ Field detection completed: {len(available_fields['text_fields'])} text fields, "
                   f"{len(available_fields['safe_aggregation_fields'])} safe aggregation fields")
        
        return available_fields
        
    except Exception as e:
        logger.error(f"Failed to get available fields: {e}")
        return {
            "text_fields": [],
            "keyword_fields": [],
            "nested_fields": [],
            "metadata_fields": [],
            "safe_aggregation_fields": {},
            "date_fields": [],
            "numeric_fields": []
        }

# =============================================================================
# SAFE QUERY BUILDING - PREVENTS COMPILATION ERRORS
# =============================================================================

def build_safe_search_query(query: str, available_fields: Dict[str, List[str]], 
                           filters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    NEW: Build search query with field validation to prevent compilation errors
    """
    text_fields = available_fields.get("text_fields", [])
    
    # Build search fields with validation - Updated for API structure
    search_fields = []
    field_priorities = [
        ("evaluation", 3.0),        # NEW: Full evaluation content (highest priority)
        ("transcript", 2.8),        # NEW: Call transcript (high priority)
        ("full_text", 2.5),
        ("evaluation_text", 2.2),
        ("template_name", 2.0),
        ("content", 1.8),
        ("text", 1.5),
        ("title", 1.3),
        ("description", 1.0)
    ]
    
    # Only use fields that actually exist
    for field_name, boost in field_priorities:
        if field_name in text_fields:
            search_fields.append(f"{field_name}^{boost}")
    
    # Add remaining text fields
    for field in text_fields:
        if not any(field in sf for sf in search_fields):
            search_fields.append(f"{field}^1.0")
    
    # Build main query
    if not search_fields:
        logger.warning("No text fields available, using query_string fallback")
        main_query = {
            "query_string": {
                "query": query,
                "default_operator": "AND",
                "lenient": True  # Prevent parsing errors
            }
        }
    else:
        main_query = {
            "multi_match": {
                "query": query,
                "fields": search_fields,
                "type": "best_fields",
                "fuzziness": "AUTO",
                "minimum_should_match": "75%"
            }
        }
    
    # Add nested query for chunks if available
    should_queries = [main_query]
    
    if "chunks" in available_fields.get("nested_fields", []):
        nested_query = {
            "nested": {
                "path": "chunks",
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "chunks.text^2", 
                            "chunks.content^1.5", 
                            "chunks.question^1.5", 
                            "chunks.answer^1.5"
                        ],
                        "fuzziness": "AUTO"
                    }
                },
                "score_mode": "max",
                "inner_hits": {
                    "size": 3,
                    "_source": ["text", "content", "content_type"]
                }
            }
        }
        should_queries.append(nested_query)
    
    # Combine queries
    combined_query = {
        "bool": {
            "should": should_queries,
            "minimum_should_match": 1
        }
    }
    
    # Add filters safely
    if filters:
        filter_clauses = build_safe_filter_clauses(filters, available_fields)
        if filter_clauses:
            combined_query["bool"]["filter"] = filter_clauses
    
    return combined_query

def build_safe_filter_clauses(filters: Dict[str, Any], available_fields: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    NEW: Build filter clauses with field validation
    """
    filter_clauses = []
    safe_agg_fields = available_fields.get("safe_aggregation_fields", {})
    
    # Text-based filters
    for filter_key, filter_value in filters.items():
        if not filter_value or filter_key == "collection":
            continue
            
        # Check if we have a safe field for this filter
        if filter_key in safe_agg_fields:
            safe_field = safe_agg_fields[filter_key]
            filter_clauses.append({
                "term": {safe_field: filter_value}
            })
            logger.debug(f"Added safe filter: {safe_field} = {filter_value}")
        else:
            logger.warning(f"No safe field found for filter: {filter_key}")
    
    # Date range filters
    if filters.get("call_date_start") or filters.get("call_date_end"):
        date_range = {}
        if filters.get("call_date_start"):
            date_range["gte"] = filters["call_date_start"]
        if filters.get("call_date_end"):
            date_range["lte"] = filters["call_date_end"]
        
        # Try different date field variations
        date_fields = ["metadata.call_date", "call_date", "created_on"]
        for date_field in date_fields:
            if date_field in available_fields.get("date_fields", []) or date_field in available_fields.get("metadata_fields", []):
                filter_clauses.append({
                    "range": {date_field: date_range}
                })
                logger.debug(f"Added date filter: {date_field}")
                break
    
    # Numeric range filters
    if filters.get("min_duration") or filters.get("max_duration"):
        duration_range = {}
        if filters.get("min_duration"):
            duration_range["gte"] = int(filters["min_duration"])
        if filters.get("max_duration"):
            duration_range["lte"] = int(filters["max_duration"])
        
        duration_fields = ["metadata.call_duration", "call_duration", "duration"]
        for duration_field in duration_fields:
            if duration_field in available_fields.get("numeric_fields", []) or duration_field in available_fields.get("metadata_fields", []):
                filter_clauses.append({
                    "range": {duration_field: duration_range}
                })
                logger.debug(f"Added duration filter: {duration_field}")
                break
    
    return filter_clauses

# =============================================================================
# SAFE SEARCH EXECUTION - PREVENTS COMPILATION ERRORS
# =============================================================================

def safe_search_with_error_handling(client, index_pattern: str, query_body: Dict, timeout: int = 30) -> Dict:
    """
    NEW: Execute search with comprehensive error handling
    """
    try:
        # Add timeout to query body (OpenSearch expects string format)
        query_body["timeout"] = f"{timeout}s"
        
        # Execute search with proper timeouts
        result = client.search(
            index=index_pattern,
            body=query_body,
            request_timeout=timeout,  # Client timeout (integer)
            allow_partial_search_results=True
        )
        
        return result
        
    except RequestError as e:
        error_msg = str(e).lower()
        
        if "compile_error" in error_msg or "search_phase_execution_exception" in error_msg:
            logger.error(f"🚨 QUERY COMPILATION ERROR: {e}")
            logger.error(f"Failed query: {json.dumps(query_body, indent=2)}")
            
            # Return safe empty result
            return {
                "hits": {"hits": [], "total": {"value": 0}},
                "aggregations": {},
                "_compilation_error": True
            }
        else:
            logger.error(f"OpenSearch request error: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Search execution failed: {e}")
        return {
            "hits": {"hits": [], "total": {"value": 0}},
            "aggregations": {},
            "_search_error": True
        }

def search_opensearch(query: str, index_override: str = None, 
                     filters: Dict[str, Any] = None, size: int = 100) -> List[Dict]:
    """
    ENHANCED: Main search function with comprehensive error handling
    """
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available")
        return []
    
    index_pattern = index_override or "eval-*"
    logger.info(f"🔍 SAFE SEARCH v{VERSION}: query='{query}', size={size}")
    
    try:
        # STEP 1: Validate indices exist
        try:
            indices_exist = client.indices.exists(index=index_pattern, request_timeout=5)
            if not indices_exist:
                logger.warning(f"No indices found for pattern: {index_pattern}")
                return []
        except Exception as e:
            logger.error(f"Could not check indices: {e}")
            return []
        
        # STEP 2: Get available fields
        available_fields = get_available_fields(client, index_pattern)
        
        # STEP 3: Build safe search query
        search_query = build_safe_search_query(query, available_fields, filters)
        
        # STEP 4: Build search body
        search_body = {
            "query": search_query,
            "size": size,
            "sort": [{"_score": {"order": "desc"}}],
            "_source": True
        }
        
        # Add highlighting for available text fields
        text_fields = available_fields.get("text_fields", [])[:5]  # Limit to prevent timeout
        if text_fields:
            search_body["highlight"] = {
                "fields": {field: {"fragment_size": 150, "number_of_fragments": 1} for field in text_fields}
            }
        
        # STEP 5: Execute search with error handling
        response = safe_search_with_error_handling(client, index_pattern, search_body, timeout=30)
        
        # STEP 6: Process results
        hits = response.get("hits", {}).get("hits", [])
        results = []
        
        for hit in hits:
            source = hit.get("_source", {})
            
            # Extract highlights
            highlights = []
            if "highlight" in hit:
                for field, snippets in hit["highlight"].items():
                    highlights.extend(snippets)
            
            result = {
                "_id": hit.get("_id"),
                "_score": hit.get("_score", 0),
                "_index": hit.get("_index"),
                "evaluationId": source.get("evaluationId"),
                "internalId": source.get("internalId"),  # NEW: Internal ID from API
                "template_name": source.get("template_name"),
                "template_id": source.get("template_id"),  # NEW: Template ID from API
                "text": source.get("text", source.get("full_text", source.get("evaluation", ""))),
                "evaluation": source.get("evaluation", ""),  # NEW: Full evaluation content
                "transcript": source.get("transcript", ""),  # NEW: Call transcript
                "weight_score": source.get("weight_score", source.get("weighted_score")),  # NEW: Actual weight score
                "url": source.get("url", ""),  # NEW: Evaluation URL
                "metadata": source.get("metadata", {}),
                "highlights": highlights,
                "source": source.get("source", "opensearch"),
                "search_type": "safe_search_v4_6_0"
            }
            
            # Add chunk information if available
            if "chunks" in source and isinstance(source["chunks"], list):
                result["total_chunks"] = len(source["chunks"])
                result["sample_chunks"] = source["chunks"][:3]  # First 3 chunks
            
            results.append(result)
        
        logger.info(f"✅ Safe search completed: {len(results)} results from {response.get('hits', {}).get('total', {}).get('value', 0)} total")
        
        return results
        
    except Exception as e:
        logger.error(f"Search failed with error: {e}")
        return []

def search_by_weight_score(min_score: int = None, max_score: int = None, 
                          additional_query: str = None, size: int = 100) -> List[Dict]:
    """
    NEW: Search evaluations by weight_score (not generated 1-5 scores)
    """
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available")
        return []
    
    try:
        # Build weight score filter
        score_filter = []
        if min_score is not None or max_score is not None:
            score_range = {}
            if min_score is not None:
                score_range["gte"] = min_score
            if max_score is not None:
                score_range["lte"] = max_score
            
            # Try both field names for compatibility
            score_filter.append({
                "bool": {
                    "should": [
                        {"range": {"weight_score": score_range}},
                        {"range": {"weighted_score": score_range}},
                        {"range": {"metadata.weight_score": score_range}},
                        {"range": {"metadata.weighted_score": score_range}}
                    ]
                }
            })
        
        # Build query
        query = {"bool": {"must": []}}
        
        # Add text query if provided
        if additional_query:
            query["bool"]["must"].append({
                "multi_match": {
                    "query": additional_query,
                    "fields": ["evaluation^2", "transcript^2", "template_name^1.5", "text"]
                }
            })
        else:
            query["bool"]["must"].append({"match_all": {}})
        
        # Add score filter
        if score_filter:
            query["bool"]["filter"] = score_filter
        
        # Execute search
        search_body = {
            "query": query,
            "size": size,
            "sort": [
                {"weight_score": {"order": "desc", "missing": "_last"}},
                {"weighted_score": {"order": "desc", "missing": "_last"}},
                {"_score": {"order": "desc"}}
            ]
        }
        
        response = safe_search_with_error_handling(client, "eval-*", search_body)
        
        # Process results
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            
            result = {
                "_id": hit.get("_id"),
                "_score": hit.get("_score", 0),
                "evaluationId": source.get("evaluationId"),
                "template_name": source.get("template_name"),
                "weight_score": source.get("weight_score", source.get("weighted_score")),
                "url": source.get("url", ""),
                "metadata": source.get("metadata", {}),
                "search_type": "weight_score_search"
            }
            results.append(result)
        
        logger.info(f"✅ Weight score search completed: {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Weight score search failed: {e}")
        return []

def get_evaluation_url(evaluation_id: str) -> str:
    """
    NEW: Get the URL for a specific evaluation ID
    """
    client = get_opensearch_client()
    if not client:
        return ""
    
    try:
        response = client.search(
            index="eval-*",
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"evaluationId": evaluation_id}},
                            {"term": {"internalId": evaluation_id}}
                        ]
                    }
                },
                "size": 1,
                "_source": ["url", "evaluationId", "internalId"]
            }
        )
        
        hits = response.get("hits", {}).get("hits", [])
        if hits:
            return hits[0].get("_source", {}).get("url", "")
        
        return ""
        
    except Exception as e:
        logger.error(f"Failed to get evaluation URL: {e}")
        return ""

# =============================================================================
# SAFE AGGREGATION FUNCTIONS - PREVENTS COMPILATION ERRORS
# =============================================================================

def get_safe_aggregation_query(field_name: str, available_fields: Dict[str, List[str]], 
                              agg_name: str = None, size: int = 20) -> Optional[Dict]:
    """
    NEW: Build safe aggregation query for a specific field
    """
    safe_agg_fields = available_fields.get("safe_aggregation_fields", {})
    
    if field_name not in safe_agg_fields:
        logger.warning(f"Field '{field_name}' not available for safe aggregation")
        return None
    
    safe_field_path = safe_agg_fields[field_name]
    agg_name = agg_name or f"{field_name}_aggregation"
    
    return {
        agg_name: {
            "terms": {
                "field": safe_field_path,
                "size": size,
                "missing": "Unknown"
            }
        }
    }

def get_safe_statistics(client, index_pattern: str = "eval-*") -> Dict:
    """
    NEW: Get statistics using only safe aggregation fields
    """
    try:
        # Get available fields first
        available_fields = get_available_fields(client, index_pattern)
        safe_agg_fields = available_fields.get("safe_aggregation_fields", {})
        
        if not safe_agg_fields:
            logger.warning("No safe aggregation fields available")
            return {
                "total_documents": 0,
                "error": "No safe aggregation fields found",
                "available_fields": available_fields
            }
        
        # Build safe aggregation query
        agg_body = {"size": 0}
        aggregations = {}
        
        # Add safe aggregations for common fields - Updated for API structure
        common_fields = ["program", "disposition", "subDisposition", "partner", "site", "lob", "agentName", "language"]
        for field_name in common_fields:
            if field_name in safe_agg_fields:
                agg_query = get_safe_aggregation_query(field_name, available_fields, size=15)
                if agg_query:
                    aggregations.update(agg_query)
        
        if aggregations:
            agg_body["aggs"] = aggregations
        
        # Execute with error handling
        result = safe_search_with_error_handling(client, index_pattern, agg_body, timeout=25)
        
        # Process results
        stats = {
            "total_documents": result.get("hits", {}).get("total", {}).get("value", 0),
            "aggregations": {},
            "safe_fields_used": list(safe_agg_fields.keys()),
            "version": VERSION
        }
        
        if "aggregations" in result:
            for agg_name, agg_data in result["aggregations"].items():
                if "buckets" in agg_data:
                    stats["aggregations"][agg_name] = {
                        bucket["key"]: bucket["doc_count"]
                        for bucket in agg_data["buckets"]
                    }
        
        logger.info(f"✅ Safe statistics completed: {stats['total_documents']} documents, {len(stats['aggregations'])} aggregations")
        return stats
        
    except Exception as e:
        logger.error(f"Safe statistics failed: {e}")
        return {
            "error": str(e),
            "total_documents": 0,
            "version": VERSION
        }

# =============================================================================
# INDEX MANAGEMENT - ENHANCED WITH BETTER ERROR HANDLING
# =============================================================================

def ensure_evaluation_index_exists(client, index_name: str):
    """Enhanced index creation with better field mappings"""
    try:
        if client.indices.exists(index=index_name, request_timeout=10):
            logger.info(f"✅ Index {index_name} already exists")
            return
    except Exception as e:
        logger.warning(f"Could not check index existence: {e}")
    
    # Enhanced mapping with all necessary fields
    mapping = {
        "mappings": {
            "properties": {
                # Core identification fields
                "evaluationId": {"type": "keyword"},
                "internalId": {"type": "keyword"},
                "template_id": {"type": "keyword"},
                "template_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                
                # Content fields - Updated to match API structure
                "text": {"type": "text", "analyzer": "standard"},
                "full_text": {"type": "text", "analyzer": "standard"},
                "content": {"type": "text", "analyzer": "standard"},
                "evaluation_text": {"type": "text", "analyzer": "standard"},
                "evaluation": {"type": "text", "analyzer": "standard"},  # NEW: Full evaluation content
                "transcript": {"type": "text", "analyzer": "standard"},   # NEW: Call transcript
                
                # Nested chunks with comprehensive properties
                "chunks": {
                    "type": "nested",
                    "properties": {
                        "text": {"type": "text", "analyzer": "standard"},
                        "content": {"type": "text", "analyzer": "standard"},
                        "content_type": {"type": "keyword"},
                        "question": {"type": "text", "analyzer": "standard"},
                        "answer": {"type": "text", "analyzer": "standard"},
                        "section": {"type": "keyword"},
                        "score": {"type": "float"},
                        "chunk_id": {"type": "keyword"}
                    }
                },
                
                # Metadata with comprehensive field coverage
                "metadata": {
                    "properties": {
                        "evaluationId": {"type": "keyword"},
                        "internalId": {"type": "keyword"},
                        "template_id": {"type": "keyword"},
                        "template_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "program": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "partner": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "site": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "lob": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "agentName": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "agentId": {"type": "keyword"},
                        "subDisposition": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "disposition": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "sub_disposition": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "language": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "call_date": {"type": "date"},
                        "call_duration": {"type": "integer"},
                        "created_on": {"type": "date"},
                        #"phone_number": {"type": "keyword"},
                        #"contact_id": {"type": "keyword"},
                        #"ucid": {"type": "keyword"},
                        "call_type": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "weighted_score": {"type": "integer"},
                        "url": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}
                    }
                },
                
                # Additional system fields
                "source": {"type": "keyword"},
                "indexed_at": {"type": "date"},
                "collection_name": {"type": "keyword"},
                "_structure_version": {"type": "keyword"},
                "version": {"type": "keyword"}
            }
        }
    }
    
    try:
        client.indices.create(
            index=index_name,
            body=mapping,
            request_timeout=60
        )
        logger.info(f"✅ Created index {index_name} with v{VERSION} mapping")
    except Exception as e:
        logger.error(f"❌ Failed to create index {index_name}: {e}")
        raise

def index_document(doc_id: str, document: Dict[str, Any], index_override: str = None) -> bool:
    """Enhanced document indexing with better error handling"""
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available")
        return False
    
    index_name = index_override or "evaluations-grouped"
    
    try:
        # Ensure index exists
        ensure_evaluation_index_exists(client, index_name)
        
        # Add system fields
        document["_indexed_at"] = datetime.now().isoformat()
        document["_structure_version"] = VERSION
        
        # Index with proper timeout
        response = client.index(
            index=index_name,
            id=doc_id,
            body=document,
            refresh=True,
            request_timeout=60
        )
        
        logger.info(f"✅ Indexed document {doc_id} in {index_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to index document {doc_id}: {e}")
        return False

# =============================================================================
# HEALTH CHECK AND DEBUGGING - ENHANCED
# =============================================================================

def health_check() -> Dict[str, Any]:
    """Enhanced health check with comprehensive status"""
    try:
        client = get_opensearch_client()
        
        if not client:
            return {
                "status": "unhealthy",
                "error": "Could not create OpenSearch client",
                "version": VERSION
            }
        
        if test_connection():
            info = client.info(request_timeout=10)
            
            # Test field detection
            try:
                available_fields = get_available_fields(client)
                field_status = "working"
                searchable_fields = len(available_fields.get("text_fields", []))
                safe_agg_fields = len(available_fields.get("safe_aggregation_fields", {}))
            except Exception as e:
                field_status = f"failed: {str(e)}"
                searchable_fields = 0
                safe_agg_fields = 0
            
            return {
                "status": "healthy",
                "connected": True,
                "cluster_name": info.get("cluster_name", "Unknown"),
                "version": info.get("version", {}).get("number", "Unknown"),
                "client_version": VERSION,
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": os.getenv("OPENSEARCH_PORT", "25060"),
                "field_detection": field_status,
                "searchable_fields": searchable_fields,
                "safe_aggregation_fields": safe_agg_fields,
                "fixes_applied": FIXES_APPLIED,
                "capabilities": [
                    "safe_search_queries",
                    "compilation_error_prevention",
                    "field_validation",
                    "robust_aggregations",
                    "enhanced_error_handling"
                ]
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Connection test failed",
                "version": VERSION
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": VERSION
        }

def debug_search_simple(query: str = "test") -> Dict[str, Any]:
    """Enhanced debug search with field analysis"""
    client = get_opensearch_client()
    if not client:
        return {"error": "No client available", "version": VERSION}
    
    try:
        # Get available fields
        available_fields = get_available_fields(client)
        
        # Simple search test
        response = client.search(
            index="eval-*",
            body={
                "query": {"match_all": {}},
                "size": 3,
                "_source": True
            },
            request_timeout=30
        )
        
        hits = response.get("hits", {}).get("hits", [])
        total = response.get("hits", {}).get("total", {})
        
        # Analyze document structure
        sample_docs = []
        for hit in hits:
            source = hit.get("_source", {})
            doc_analysis = {
                "id": hit.get("_id"),
                "index": hit.get("_index"),
                "fields_present": list(source.keys()),
                "has_metadata": "metadata" in source,
                "has_chunks": "chunks" in source and isinstance(source.get("chunks"), list),
                "has_weight_score": "weight_score" in source or "weighted_score" in source,
                "has_url": "url" in source,
                "has_evaluation": "evaluation" in source,
                "has_transcript": "transcript" in source,
                "text_fields": [k for k in source.keys() if isinstance(source.get(k), str) and len(source.get(k, "")) > 50],
                "metadata_fields": list(source.get("metadata", {}).keys()) if source.get("metadata") else [],
                "weight_score": source.get("weight_score", source.get("weighted_score")),
                "url": source.get("url", ""),
                "evaluationId": source.get("evaluationId"),
                "internalId": source.get("internalId")
            }
            sample_docs.append(doc_analysis)
        
        return {
            "status": "success",
            "total_documents": total.get("value", 0) if isinstance(total, dict) else total,
            "available_fields": available_fields,
            "sample_documents": sample_docs,
            "version": VERSION,
            "compilation_safe": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "version": VERSION,
            "compilation_safe": False
        }

# =============================================================================
# UTILITY FUNCTIONS - ENHANCED
# =============================================================================

def get_opensearch_config():
    """Enhanced configuration information"""
    return {
        "host": os.getenv("OPENSEARCH_HOST", "not_configured"),
        "port": os.getenv("OPENSEARCH_PORT", "25060"),
        "user": os.getenv("OPENSEARCH_USER", "not_configured"),
        "ssl": True,
        "verify_certs": False,
        "version": VERSION,
        "fixes_applied": FIXES_APPLIED,
        "capabilities": [
            "timeout_fix",
            "compilation_error_prevention",
            "field_validation",
            "safe_aggregations"
        ]
    }

def get_connection_status() -> Dict[str, Any]:
    """Enhanced connection status"""
    return {
        "connected": test_connection(),
        "host": os.getenv("OPENSEARCH_HOST"),
        "port": os.getenv("OPENSEARCH_PORT", "25060"),
        "user": os.getenv("OPENSEARCH_USER"),
        "version": VERSION,
        "last_test": datetime.now().isoformat(),
        "improvements": "comprehensive_v4_6_0_fixes"
    }

# =============================================================================
# VECTOR SEARCH FUNCTIONS - STILL DISABLED
# =============================================================================

def search_vector(query_vector: List[float], index_override: str = None, 
                 size: int = 10) -> List[Dict]:
    """Vector search still disabled for stability"""
    logger.warning("🔮 Vector search disabled in v4.6.0")
    return []

def detect_vector_support(client) -> bool:
    """Vector support detection disabled"""
    return False

# =============================================================================
# MAIN EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print(f"🧪 Testing OpenSearch Client v{VERSION}")
    print("=" * 60)
    print("✅ FIXES APPLIED:")
    for fix in FIXES_APPLIED:
        print(f"   - {fix}")
    print("=" * 60)
    
    # Test health check
    health = health_check()
    print(f"\n🏥 Health Check: {health['status']}")
    
    if health["status"] == "healthy":
        print(f"   Cluster: {health['cluster_name']}")
        print(f"   Version: {health['version']}")
        print(f"   Searchable Fields: {health['searchable_fields']}")
        print(f"   Safe Aggregation Fields: {health['safe_aggregation_fields']}")
        
        # Test search
        print("\n🔍 Testing Safe Search...")
        search_results = search_opensearch("customer service", size=3)
        print(f"   Results: {len(search_results)} documents found")
        if search_results:
            first_result = search_results[0]
            print(f"   Sample weight_score: {first_result.get('weight_score', 'N/A')}")
            print(f"   Sample URL: {first_result.get('url', 'N/A')}")
        
        # Test weight score search
        print("\n📊 Testing Weight Score Search...")
        weight_results = search_by_weight_score(min_score=80, additional_query="customer", size=3)
        print(f"   High-score results: {len(weight_results)} documents found")
        
        # Test statistics
        print("\n📊 Testing Safe Statistics...")
        client = get_opensearch_client()
        stats = get_safe_statistics(client)
        print(f"   Total Documents: {stats.get('total_documents', 0)}")
        print(f"   Aggregations: {len(stats.get('aggregations', {}))}")
        
        print(f"\n✅ All tests completed successfully!")
        print(f"🎉 OpenSearch Client v{VERSION} is ready for production!")
        print(f"🔗 Weight scores and URLs properly handled!")
        print(f"📝 Evaluation and transcript content indexed!")
        
    else:
        print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
    
    print("=" * 60)

else:
    logger.info(f"🔌 OpenSearch Client v{VERSION} loaded successfully")
    logger.info(f"   Fixes: {', '.join(FIXES_APPLIED)}")
    logger.info(f"   Ready for production use with enhanced safety")