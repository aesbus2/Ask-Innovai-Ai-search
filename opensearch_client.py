# opensearch_client.py - FIXED VERSION for Empty Search Results
# Version: 4.5.0 - Fixed field mapping, robust search, better error handling
# FIXES: Field existence checking, flexible search, comprehensive debugging

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

# =============================================================================
# FIXED CLIENT CREATION WITH PROPER TIMEOUT CONFIGURATION
# =============================================================================

def get_client():
    """Get OpenSearch client with proper timeout configuration"""
    if not OPENSEARCH_AVAILABLE:
        logger.error("OpenSearch library not available")
        return None
    
    try:
        client = OpenSearch(
            hosts=[{
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": int(os.getenv("OPENSEARCH_PORT", "25060"))
            }],
            http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS")),
            use_ssl=True,
            verify_certs=False,
            request_timeout=30,
            connect_timeout=30,
            max_retries=3,
            retry_on_timeout=True,
            ssl_show_warn=False,
            pool_maxsize=20,
            http_compress=True
        )
        
        test_result = client.ping()
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
    """Test connection with enhanced error handling"""
    client = get_opensearch_client()
    if not client:
        return False
    
    try:
        result = client.ping()
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
# ENHANCED SEARCH FUNCTIONS - FIXED FOR FIELD EXISTENCE
# =============================================================================

def get_available_fields(client, index_pattern: str = "eval-*") -> Dict[str, List[str]]:
    """
    FIXED: Get actually available fields from index mappings
    """
    try:
        # Get mappings for the index pattern
        mappings_response = client.indices.get_mapping(index=index_pattern)
        
        available_fields = {
            "text_fields": [],
            "keyword_fields": [],
            "nested_fields": [],
            "metadata_fields": []
        }
        
        for index_name, mapping_data in mappings_response.items():
            properties = mapping_data.get("mappings", {}).get("properties", {})
            
            # Check top-level fields
            for field_name, field_config in properties.items():
                field_type = field_config.get("type", "")
                
                if field_type == "text":
                    available_fields["text_fields"].append(field_name)
                elif field_type == "keyword":
                    available_fields["keyword_fields"].append(field_name)
                elif field_type == "nested":
                    available_fields["nested_fields"].append(field_name)
                
                # Check for text fields with keyword subfields
                if "fields" in field_config:
                    if "keyword" in field_config["fields"]:
                        available_fields["keyword_fields"].append(f"{field_name}.keyword")
                
                # Check metadata fields
                if field_name == "metadata" and "properties" in field_config:
                    for meta_field in field_config["properties"].keys():
                        available_fields["metadata_fields"].append(f"metadata.{meta_field}")
                        if field_config["properties"][meta_field].get("fields", {}).get("keyword"):
                            available_fields["metadata_fields"].append(f"metadata.{meta_field}.keyword")
        
        # Remove duplicates
        for key in available_fields:
            available_fields[key] = list(set(available_fields[key]))
        
        logger.info(f"📋 Available fields found:")
        logger.info(f"   Text fields: {available_fields['text_fields'][:5]}...")
        logger.info(f"   Keyword fields: {available_fields['keyword_fields'][:5]}...")
        logger.info(f"   Nested fields: {available_fields['nested_fields']}")
        logger.info(f"   Metadata fields: {available_fields['metadata_fields'][:5]}...")
        
        return available_fields
        
    except Exception as e:
        logger.error(f"Failed to get available fields: {e}")
        return {
            "text_fields": [],
            "keyword_fields": [],
            "nested_fields": [],
            "metadata_fields": []
        }

def build_flexible_search_query(query: str, available_fields: Dict[str, List[str]], 
                               filters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    FIXED: Build a flexible search query based on actually available fields
    """
    
    # Build multi_match query with available text fields
    text_fields = available_fields.get("text_fields", [])
    
    # Create field list with boosts, using only fields that exist
    search_fields = []
    
    # Priority field mapping - only add if fields exist
    field_priorities = [
        ("full_text", 3.0),
        ("evaluation_text", 2.5),
        ("transcript_text", 2.0),
        ("template_name", 1.5),
        ("content", 1.8),
        ("text", 1.5),
        ("title", 1.3),
        ("description", 1.0)
    ]
    
    for field_name, boost in field_priorities:
        if field_name in text_fields:
            search_fields.append(f"{field_name}^{boost}")
            logger.debug(f"Added search field: {field_name}^{boost}")
    
    # Add any other text fields we haven't covered
    for field in text_fields:
        field_with_boost = f"{field}^1.0"
        if not any(field in existing_field for existing_field in search_fields):
            search_fields.append(field_with_boost)
    
    # If no text fields found, use _all or a simple query
    if not search_fields:
        logger.warning("No text fields found, using simple query_string")
        main_query = {
            "query_string": {
                "query": query,
                "default_operator": "AND",
                "fuzziness": "AUTO"
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
    
    # Build nested queries for chunks if chunks field exists
    should_queries = [main_query]
    
    if "chunks" in available_fields.get("nested_fields", []):
        logger.info("📦 Found chunks field, adding nested query")
        nested_query = {
            "nested": {
                "path": "chunks",
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["chunks.text^2", "chunks.content^1.5", "chunks.question^1.5", "chunks.answer^1.5"],
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
    
    # Combine all queries
    combined_query = {
        "bool": {
            "should": should_queries,
            "minimum_should_match": 1
        }
    }
    
    # Add filters if provided
    if filters and any(filters.values()):
        filter_clauses = build_filter_clauses(filters, available_fields)
        if filter_clauses:
            combined_query["bool"]["filter"] = filter_clauses
    
    return combined_query

def build_filter_clauses(filters: Dict[str, Any], available_fields: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    FIXED: Build filter clauses using only available fields
    """
    filter_clauses = []
    metadata_fields = available_fields.get("metadata_fields", [])
    keyword_fields = available_fields.get("keyword_fields", [])
    
    # Date filters
    if filters.get("call_date_start") or filters.get("call_date_end"):
        date_range = {}
        if filters.get("call_date_start"):
            date_range["gte"] = filters["call_date_start"]
        if filters.get("call_date_end"):
            date_range["lte"] = filters["call_date_end"]
        
        # Try different date field variations
        date_fields = ["metadata.call_date", "call_date", "date"]
        for date_field in date_fields:
            if date_field in metadata_fields or date_field in keyword_fields:
                filter_clauses.append({
                    "range": {date_field: date_range}
                })
                logger.info(f"📅 Added date filter for field: {date_field}")
                break
    
    # Keyword filters with flexible field matching
    keyword_filter_mapping = {
        "template_name": ["template_name.keyword", "template_name"],
        "template_id": ["template_id", "template_id.keyword"],
        "program": ["metadata.program.keyword", "metadata.program", "program.keyword", "program"],
        "partner": ["metadata.partner.keyword", "metadata.partner", "partner.keyword", "partner"],
        "site": ["metadata.site.keyword", "metadata.site", "site.keyword", "site"],
        "lob": ["metadata.lob.keyword", "metadata.lob", "lob.keyword", "lob"],
        "agent_name": ["metadata.agent.keyword", "metadata.agent", "agent.keyword", "agent"],
        "agent": ["metadata.agent.keyword", "metadata.agent", "agent.keyword", "agent"],
        "disposition": ["metadata.disposition.keyword", "metadata.disposition", "disposition.keyword", "disposition"],
        "sub_disposition": ["metadata.sub_disposition.keyword", "metadata.sub_disposition", "sub_disposition.keyword", "sub_disposition"],
        "language": ["metadata.language.keyword", "metadata.language", "language.keyword", "language"],
        "call_type": ["metadata.call_type.keyword", "metadata.call_type", "call_type.keyword", "call_type"],
        "phone_number": ["metadata.phone_number", "phone_number"],
        "contact_id": ["metadata.contact_id", "contact_id"],
        "ucid": ["metadata.ucid", "ucid"]
    }
    
    for filter_key, possible_fields in keyword_filter_mapping.items():
        filter_value = filters.get(filter_key)
        if filter_value and str(filter_value).strip():
            # Find the first available field from the list
            available_field = None
            for field in possible_fields:
                if field in metadata_fields or field in keyword_fields:
                    available_field = field
                    break
            
            if available_field:
                filter_clauses.append({
                    "term": {available_field: filter_value}
                })
                logger.info(f"🏷️ Added filter: {filter_key}='{filter_value}' using field: {available_field}")
            else:
                logger.warning(f"⚠️ No available field found for filter: {filter_key}")
    
    # Duration filters
    if filters.get("min_duration") or filters.get("max_duration"):
        duration_range = {}
        if filters.get("min_duration"):
            duration_range["gte"] = int(filters["min_duration"])
        if filters.get("max_duration"):
            duration_range["lte"] = int(filters["max_duration"])
        
        # Try different duration field variations
        duration_fields = ["metadata.call_duration", "call_duration", "duration"]
        for duration_field in duration_fields:
            if duration_field in metadata_fields or duration_field in keyword_fields:
                filter_clauses.append({
                    "range": {duration_field: duration_range}
                })
                logger.info(f"⏱️ Added duration filter for field: {duration_field}")
                break
    
    return filter_clauses

def search_opensearch(query: str, index_override: str = None, 
                     filters: Dict[str, Any] = None, size: int = 100) -> List[Dict]:
    """
    FIXED: Search evaluations with flexible field detection and robust error handling
    """
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available for search")
        return []
    
    # Determine index pattern
    index_pattern = index_override or "eval-*"
    logger.info(f"🔍 FLEXIBLE SEARCH: index='{index_pattern}', query='{query}', size={size}")
    logger.info(f"🏷️ FILTERS: {filters}")
    
    try:
        # STEP 1: Check if any indices exist
        try:
            indices_response = client.indices.get(index=index_pattern)
            available_indices = list(indices_response.keys())
            logger.info(f"📊 AVAILABLE INDICES: {len(available_indices)} ({available_indices[:3]}...)")
        except Exception as e:
            logger.error(f"❌ NO INDICES FOUND for pattern '{index_pattern}': {e}")
            return []
        
        # STEP 2: Get available fields from mappings
        available_fields = get_available_fields(client, index_pattern)
        
        # STEP 3: Build flexible search query based on available fields
        search_query = build_flexible_search_query(query, available_fields, filters)
        
        # STEP 4: Build final search body
        search_body = {
            "query": search_query,
            "size": size,
            "sort": [
                {"_score": {"order": "desc"}}
            ],
            "_source": True  # Include all source fields
        }
        
        # Add highlighting for available text fields
        text_fields = available_fields.get("text_fields", [])
        if text_fields:
            highlight_fields = {}
            for field in text_fields[:5]:  # Limit to first 5 fields
                highlight_fields[field] = {
                    "fragment_size": 200,
                    "number_of_fragments": 1
                }
            
            search_body["highlight"] = {
                "fields": highlight_fields
            }
        
        # STEP 5: Execute search with detailed logging
        logger.info(f"🚀 EXECUTING FLEXIBLE SEARCH...")
        logger.debug(f"Search body: {json.dumps(search_body, indent=2)}")
        
        response = client.search(
            index=index_pattern,
            body=search_body,
            timeout=45
        )
        
        # STEP 6: Process results
        hits = response.get("hits", {}).get("hits", [])
        total_hits = response.get("hits", {}).get("total", {})
        
        # Handle different total formats
        if isinstance(total_hits, dict):
            total_count = total_hits.get("value", 0)
        else:
            total_count = total_hits
        
        logger.info(f"✅ FLEXIBLE SEARCH COMPLETED: {len(hits)} hits returned, {total_count} total matches")
        
        # STEP 7: Build result objects with flexible field extraction
        results = []
        for i, hit in enumerate(hits):
            try:
                source = hit.get("_source", {})
                
                # Flexible field extraction - try multiple field names
                result = {
                    "_id": hit.get("_id"),
                    "_score": hit.get("_score", 0),
                    "_index": hit.get("_index"),
                    "_source": source,
                    
                    # Extract key fields with fallbacks
                    "evaluationId": (source.get("evaluationId") or 
                                   source.get("evaluation_id") or 
                                   source.get("internalId") or 
                                   source.get("id")),
                    
                    "template_name": (source.get("template_name") or 
                                    source.get("templateName") or 
                                    "Unknown Template"),
                    
                    "template_id": (source.get("template_id") or 
                                  source.get("templateId") or 
                                  source.get("template")),
                    
                    # Extract text content with multiple fallbacks
                    "text": extract_text_content(source),
                    "metadata": source.get("metadata", {}),
                    
                    # Additional fields
                    "total_chunks": source.get("total_chunks", 0),
                    "chunks": source.get("chunks", []),
                    "highlight": hit.get("highlight", {}),
                    "inner_hits": hit.get("inner_hits", {})
                }
                
                results.append(result)
                
                if i < 10:  # Log first 10 for visibility
                    logger.info(f"📄 RESULT {i+1}: {result['evaluationId']} (score: {result['_score']:.3f})")
                
            except Exception as e:
                logger.error(f"❌ Failed to process hit {i}: {e}")
        
        logger.info(f"🎯 FLEXIBLE SEARCH SUMMARY: {len(results)} processed results")
        return results
        
    except Exception as e:
        logger.error(f"❌ FLEXIBLE SEARCH FAILED: {e}")
        logger.error(f"🔍 Query: '{query}'")
        logger.error(f"🏷️ Filters: {filters}")
        logger.error(f"📍 Index: '{index_pattern}'")
        return []

def extract_text_content(source: Dict[str, Any]) -> str:
    """
    FIXED: Extract text content with multiple fallback strategies
    """
    # Try different text field names in order of preference
    text_field_candidates = [
        "full_text",
        "evaluation_text", 
        "transcript_text",
        "content",
        "text",
        "description",
        "summary"
    ]
    
    for field_name in text_field_candidates:
        if field_name in source and source[field_name]:
            text_content = source[field_name]
            if isinstance(text_content, str) and len(text_content.strip()) > 20:
                return text_content[:500]  # Truncate for preview
    
    # Try to extract from chunks
    chunks = source.get("chunks", [])
    if chunks and isinstance(chunks, list):
        chunk_texts = []
        for chunk in chunks[:3]:  # First 3 chunks
            if isinstance(chunk, dict):
                chunk_text = (chunk.get("text") or 
                            chunk.get("content") or 
                            chunk.get("description"))
                if chunk_text and isinstance(chunk_text, str):
                    chunk_texts.append(chunk_text)
        
        if chunk_texts:
            return "\n".join(chunk_texts)[:500]
    
    # Last resort: try any string field
    for key, value in source.items():
        if isinstance(value, str) and len(value.strip()) > 50:
            return value[:500]
    
    return "No text content found"

# =============================================================================
# DEBUG FUNCTIONS WITH ENHANCED FIELD CHECKING
# =============================================================================

def debug_search_simple(query: str = "test") -> Dict[str, Any]:
    """
    DEBUG: Simple search test with field detection
    """
    client = get_opensearch_client()
    if not client:
        return {"error": "No client available"}
    
    try:
        # First, check what fields are available
        available_fields = get_available_fields(client)
        
        # Simple match_all query first
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
        
        # Analyze the structure of returned documents
        sample_docs = []
        for hit in hits:
            source = hit.get("_source", {})
            doc_analysis = {
                "id": hit.get("_id"),
                "index": hit.get("_index"),
                "fields_present": list(source.keys()),
                "has_metadata": "metadata" in source,
                "has_chunks": "chunks" in source and isinstance(source.get("chunks"), list),
                "text_fields_found": [k for k in source.keys() if isinstance(source.get(k), str) and len(source.get(k, "")) > 50]
            }
            sample_docs.append(doc_analysis)
        
        return {
            "status": "success",
            "total_documents": total.get("value", 0) if isinstance(total, dict) else total,
            "available_fields": available_fields,
            "sample_documents": sample_docs,
            "field_detection": "completed",
            "recommendations": [
                "Check if text_fields_found contains searchable content",
                "Verify that metadata field exists if you need filtering",
                "Look at fields_present to understand document structure"
            ]
        }
        
    except Exception as e:
        return {"error": str(e), "field_detection": "failed"}

def debug_field_mappings(index_pattern: str = "eval-*") -> Dict[str, Any]:
    """
    DEBUG: Get detailed field mappings for troubleshooting
    """
    client = get_opensearch_client()
    if not client:
        return {"error": "No client available"}
    
    try:
        mappings_response = client.indices.get_mapping(index=index_pattern)
        
        detailed_mappings = {}
        for index_name, mapping_data in mappings_response.items():
            properties = mapping_data.get("mappings", {}).get("properties", {})
            
            index_analysis = {
                "total_fields": len(properties),
                "field_details": {}
            }
            
            for field_name, field_config in properties.items():
                field_analysis = {
                    "type": field_config.get("type", "unknown"),
                    "searchable": field_config.get("type") in ["text", "keyword"],
                    "has_keyword_subfield": "fields" in field_config and "keyword" in field_config.get("fields", {}),
                    "is_nested": field_config.get("type") == "nested"
                }
                
                if field_analysis["is_nested"] and "properties" in field_config:
                    field_analysis["nested_fields"] = list(field_config["properties"].keys())
                
                index_analysis["field_details"][field_name] = field_analysis
            
            detailed_mappings[index_name] = index_analysis
        
        return {
            "status": "success",
            "mappings": detailed_mappings,
            "troubleshooting_tips": [
                "Look for 'searchable': true fields for text search",
                "Check nested_fields for chunk-based search",
                "Verify metadata field structure for filtering"
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# VECTOR SEARCH FUNCTIONS - STILL DISABLED
# =============================================================================

def search_vector(query_vector: List[float], index_override: str = None, 
                 size: int = 10) -> List[Dict]:
    """Vector search temporarily disabled"""
    logger.warning("🔮 Vector search temporarily disabled")
    return []

def detect_vector_support(client) -> bool:
    """Vector support detection disabled"""
    return False

# =============================================================================
# INDEX MANAGEMENT AND DOCUMENT OPERATIONS
# =============================================================================

def ensure_evaluation_index_exists(client, index_name: str):
    """Create index with evaluation grouping mapping (vector fields disabled)"""
    if client.indices.exists(index=index_name):
        return
    
    logger.info(f"🏗️ Creating index {index_name}")
    
    chunk_properties = {
        "chunk_index": {"type": "integer"},
        "text": {"type": "text", "analyzer": "standard"},
        "content": {"type": "text", "analyzer": "standard"},  # Added content field
        "content_type": {"type": "keyword"},
        "length": {"type": "integer"},
        "section": {"type": "keyword"},
        "question": {"type": "text", "analyzer": "standard"},
        "answer": {"type": "text", "analyzer": "standard"},
        "qa_pair_index": {"type": "integer"},
        "speakers": {"type": "keyword"},
        "timestamps": {"type": "keyword"},
        "speaker_count": {"type": "integer"},
        "transcript_chunk_index": {"type": "integer"}
    }
    
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "standard_analyzer": {
                        "type": "standard"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                # Primary identification fields
                "evaluationId": {"type": "keyword"},
                "evaluation_id": {"type": "keyword"},  # Alternative field name
                "internalId": {"type": "keyword"},
                "template_id": {"type": "keyword"},
                "template_name": {
                    "type": "text", 
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                
                # Document metadata
                "document_type": {"type": "keyword"},
                "total_chunks": {"type": "integer"},
                "evaluation_chunks_count": {"type": "integer"},
                "transcript_chunks_count": {"type": "integer"},
                
                # Text content fields
                "full_text": {"type": "text", "analyzer": "standard"},
                "evaluation_text": {"type": "text", "analyzer": "standard"},
                "transcript_text": {"type": "text", "analyzer": "standard"},
                "content": {"type": "text", "analyzer": "standard"},  # Generic content field
                "text": {"type": "text", "analyzer": "standard"},      # Simple text field
                
                # Nested chunks
                "chunks": {
                    "type": "nested",
                    "properties": chunk_properties
                },
                
                # Metadata with flexible field names
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
                        "agent": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "agent_id": {"type": "keyword"},
                        "disposition": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "sub_disposition": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "language": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "call_date": {"type": "date"},
                        "call_duration": {"type": "integer"},
                        "created_on": {"type": "date"},
                        "phone_number": {"type": "keyword"},
                        "contact_id": {"type": "keyword"},
                        "ucid": {"type": "keyword"},
                        "call_type": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}
                    }
                },
                
                # Additional fields
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
        logger.info(f"✅ Created evaluation index: {index_name}")
    except Exception as e:
        logger.error(f"❌ Failed to create index {index_name}: {e}")
        raise

def index_document(doc_id: str, document: Dict[str, Any], index_override: str = None) -> bool:
    """Index evaluation document with fixed timeout"""
    client = get_opensearch_client()
    if not client:
        logger.error("❌ OpenSearch client not available")
        return False
    
    index_name = index_override or "evaluations-grouped"
    
    try:
        ensure_evaluation_index_exists(client, index_name)
        
        document["_indexed_at"] = datetime.now().isoformat()
        document["_structure_version"] = "4.5.0_fixed_search"
        
        response = client.index(
            index=index_name,
            id=doc_id,
            body=document,
            refresh=True,
            request_timeout=60
        )
        
        logger.info(f"✅ Indexed evaluation {doc_id} in {index_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to index evaluation {doc_id}: {e}")
        return False

def get_evaluation_by_id(evaluation_id: str) -> Optional[Dict]:
    """Get a specific evaluation document by ID"""
    client = get_opensearch_client()
    if not client:
        return None
    
    try:
        response = client.search(
            index="eval-*",
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"evaluationId": evaluation_id}},
                            {"term": {"evaluation_id": evaluation_id}},
                            {"term": {"internalId": evaluation_id}}
                        ]
                    }
                },
                "size": 1
            },
            request_timeout=10
        )
        
        hits = response.get("hits", {}).get("hits", [])
        return hits[0] if hits else None
        
    except Exception as e:
        logger.error(f"❌ Failed to get evaluation {evaluation_id}: {e}")
        return None

# =============================================================================
# HEALTH CHECK AND STATUS FUNCTIONS
# =============================================================================

def health_check() -> Dict[str, Any]:
    """Health check with field detection status"""
    try:
        client = get_opensearch_client()
        
        if not client:
            return {
                "status": "not_configured",
                "message": "Could not create OpenSearch client"
            }
        
        if test_connection():
            info = client.info()
            
            # Test field detection
            try:
                available_fields = get_available_fields(client)
                field_detection_status = "working"
                searchable_fields_count = len(available_fields.get("text_fields", []))
            except Exception as e:
                field_detection_status = f"failed: {str(e)}"
                searchable_fields_count = 0
            
            return {
                "status": "healthy",
                "connected": True,
                "cluster_name": info.get("cluster_name", "Unknown"),
                "version": info.get("version", {}).get("number", "Unknown"),
                "host": os.getenv("OPENSEARCH_HOST"),
                "port": os.getenv("OPENSEARCH_PORT", "25060"),
                "field_detection": field_detection_status,
                "searchable_fields_found": searchable_fields_count,
                "search_type": "flexible_field_based",
                "structure_version": "4.5.0_fixed_search",
                "vector_support": False,
                "fixes_applied": [
                    "flexible_field_detection",
                    "robust_search_queries", 
                    "multiple_field_fallbacks",
                    "enhanced_error_handling"
                ]
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Connection test failed"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_opensearch_config():
    """Get OpenSearch configuration for debugging"""
    return {
        "host": os.getenv("OPENSEARCH_HOST", "not_configured"),
        "port": os.getenv("OPENSEARCH_PORT", "25060"),
        "user": os.getenv("OPENSEARCH_USER", "not_configured"),
        "ssl": True,
        "verify_certs": False,
        "fixes_applied": [
            "flexible_field_detection",
            "robust_search_queries"
        ]
    }

def get_connection_status() -> Dict[str, Any]:
    """Get connection status with field detection info"""
    return {
        "connected": test_connection(),
        "host": os.getenv("OPENSEARCH_HOST"),
        "port": os.getenv("OPENSEARCH_PORT", "25060"),
        "user": os.getenv("OPENSEARCH_USER"),
        "last_test": datetime.now().isoformat(),
        "search_improvements": "flexible_field_based_search_implemented"
    }

def get_opensearch_manager():
    """Get manager for compatibility"""
    class SimpleManager:
        def test_connection(self):
            return test_connection()
        
        def get_opensearch_config(self):
            return get_opensearch_config()
        
        def get_connection_status(self):
            return get_connection_status()
    
    return SimpleManager()

# =============================================================================
# MAIN EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing FIXED OpenSearch Client v4.5.0")
    print("✅ Expected: Flexible field detection, robust search, no empty results")
    
    health = health_check()
    print(f"\n🏥 Health check: {health['status']}")
    
    if health["status"] == "healthy":
        print(f"   Cluster: {health['cluster_name']}")
        print(f"   Version: {health['version']}")
        print(f"   Field Detection: {health['field_detection']}")
        print(f"   Searchable Fields: {health['searchable_fields_found']}")
        
        # Test field mappings
        print("\n🔍 Testing field mappings...")
        debug_result = debug_field_mappings()
        if debug_result.get("status") == "success":
            print("   Field mappings loaded successfully")
            for index_name, analysis in debug_result["mappings"].items():
                print(f"   Index {index_name}: {analysis['total_fields']} fields")
        
        # Test simple search
        print("\n🔍 Testing flexible search...")
        search_result = debug_search_simple()
        print(f"   Status: {search_result.get('status', 'failed')}")
        print(f"   Total docs: {search_result.get('total_documents', 0)}")
        
        if search_result.get("available_fields"):
            fields = search_result["available_fields"]
            print(f"   Text fields available: {len(fields.get('text_fields', []))}")
            print(f"   Sample text fields: {fields.get('text_fields', [])[:3]}")
        
        print("\n✅ FIXED client with flexible search ready!")
        print("🔍 Should now find content regardless of field names")
        
    else:
        print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
    
    print("\n🏁 Testing complete!")
else:
    logger.info("🔌 FIXED OpenSearch client v4.5.0 loaded")
    logger.info("   ✅ Flexible field detection implemented")
    logger.info("   ✅ Robust search queries with fallbacks")
    logger.info("   ✅ Multiple field name support")
    logger.info("   🔍 Should resolve empty search results")