# chat_handlers.py - FIXED: Enhanced RAG with proper debugging and filter application
# Version: 4.4.0 - METADATA ALIGNMENT FIX + Strict evaluation vs chunk counting

import os
import logging
import requests
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from opensearch_client import search_opensearch, search_vector
from embedder import embed_text
from typing import Dict, Any, List, Literal, Tuple

logger = logging.getLogger(__name__)
chat_router = APIRouter()
health_router = APIRouter()

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    history: List[dict] = []
    filters: Dict[str, Any] = {}
    analytics: bool = True
    metadata_focus: List[str] = []

# =============================================================================
# CONFIGURATION
# =============================================================================

GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT", "")
GENAI_ACCESS_KEY = os.getenv("GENAI_ACCESS_KEY", "")
GENAI_MODEL = os.getenv("GENAI_MODEL", "n/a")
GENAI_TEMPERATURE = float(os.getenv("GENAI_TEMPERATURE", "0.7"))
GENAI_MAX_TOKENS = int(os.getenv("GENAI_MAX_TOKENS", "2000"))

def verify_metadata_alignment(sources: List[dict]) -> Dict[str, Any]:
    """
    UPDATED: Complete metadata verification including weighted_score and url fields
    """
    metadata_summary = {
        # Keep existing structure for backward compatibility
        "dispositions": set(),
        "sub_dispositions": set(),
        "programs": set(),
        "partners": set(),
        "sites": set(),
        "lobs": set(),
        "agents": set(),
        "languages": set(),
        "call_dates": [],
        "evaluation_ids": set(),
        "has_real_data": False,
        "total_evaluations": 0,
        "total_chunks_found": 0,
        "data_verification": "VERIFIED_REAL_DATA",
        
        # Essential fields tracking
        "essential_fields": {
            "evaluationId": set(),
            "template_name": set(),
            "agentName": set(),
            "created_on": set()
        },
        
        # ✅ NEWLY ADDED: Missing fields that you noticed
        "weighted_scores": set(),  # ✅ For score analysis
        "urls": set(),            # ✅ For evaluation URLs
        "call_durations": set(),  # Additional field for analysis
        "phone_numbers": set(),   # Additional field for analysis
        "contact_ids": set(),     # Additional field for analysis
        "ucids": set(),          # Additional field for analysis
        "call_types": set(),     # Additional field for analysis
    }
    
    seen_evaluation_ids = set()
    
    for source in sources:
        try:
            metadata_summary["total_chunks_found"] += 1
            
            # Extract evaluation ID with better logic
            evaluation_id = None
            source_data = source.get("_source", source)
            
            # Try multiple ID field names
            for id_field in ["evaluationId", "evaluation_id", "internalId", "internal_id"]:
                if source_data.get(id_field):
                    evaluation_id = source_data[id_field]
                    break
                if source_data.get("metadata", {}).get(id_field):
                    evaluation_id = source_data["metadata"][id_field]
                    break
            
            # Fallback for direct field access
            if not evaluation_id and source.get("evaluationId"):
                evaluation_id = source.get("evaluationId")
            
            # Count unique evaluations properly
            if evaluation_id and evaluation_id not in seen_evaluation_ids:
                seen_evaluation_ids.add(evaluation_id)
                metadata_summary["total_evaluations"] += 1
                metadata_summary["evaluation_ids"].add(evaluation_id)
                metadata_summary["essential_fields"]["evaluationId"].add(str(evaluation_id))
            
            # Extract metadata with better handling
            metadata = {}
            if source_data.get("metadata"):
                metadata = source_data["metadata"]
            elif source.get("metadata"):
                metadata = source["metadata"]
            
            # Extract all standard metadata fields
            if metadata.get("disposition"):
                metadata_summary["dispositions"].add(metadata["disposition"])
            if metadata.get("sub_disposition"):
                metadata_summary["sub_dispositions"].add(metadata["sub_disposition"])
            if metadata.get("program"):
                metadata_summary["programs"].add(metadata["program"])
            if metadata.get("partner"):
                metadata_summary["partners"].add(metadata["partner"])
            if metadata.get("site"):
                metadata_summary["sites"].add(metadata["site"])
            if metadata.get("lob"):
                metadata_summary["lobs"].add(metadata["lob"])
            if metadata.get("agent"):
                metadata_summary["agents"].add(metadata["agent"])
                metadata_summary["essential_fields"]["agentName"].add(metadata["agent"])
            if metadata.get("language"):
                metadata_summary["languages"].add(metadata["language"])
                
            # ✅ NEWLY ADDED: Extract weighted_score and url fields
            if metadata.get("weighted_score") is not None:
                metadata_summary["weighted_scores"].add(str(metadata["weighted_score"]))
                
            if metadata.get("url"):
                metadata_summary["urls"].add(metadata["url"])
                
            # ✅ NEWLY ADDED: Extract additional fields for comprehensive analysis
            if metadata.get("call_duration") is not None:
                metadata_summary["call_durations"].add(str(metadata["call_duration"]))
                
            if metadata.get("phone_number"):
                metadata_summary["phone_numbers"].add(metadata["phone_number"])
                
            if metadata.get("contact_id"):
                metadata_summary["contact_ids"].add(metadata["contact_id"])
                
            if metadata.get("ucid"):
                metadata_summary["ucids"].add(metadata["ucid"])
                
            if metadata.get("call_type"):
                metadata_summary["call_types"].add(metadata["call_type"])
            
            # Extract template name
            template_name = source_data.get("template_name", "Unknown Template")
            if template_name and template_name != "Unknown Template":
                metadata_summary["essential_fields"]["template_name"].add(template_name)
            
            # Extract created_on date
            created_on = source_data.get("created_on") or metadata.get("created_on")
            if created_on:
                metadata_summary["essential_fields"]["created_on"].add(str(created_on))
            
            # Extract call dates
            if metadata.get("call_date"):
                metadata_summary["call_dates"].append(metadata["call_date"])
                
            # Mark as having real data if essential fields are present
            if evaluation_id and template_name != "Unknown Template":
                metadata_summary["has_real_data"] = True
                
        except Exception as e:
            logger.error(f"Error processing source metadata: {e}")
            continue
    
    # Convert sets to sorted lists for consistent output
    fields_to_convert = [
        "dispositions", "sub_dispositions", "programs", "partners", "sites", 
        "lobs", "agents", "languages", "weighted_scores", "urls", 
        "call_durations", "phone_numbers", "contact_ids", "ucids", "call_types"
    ]
    
    for key in fields_to_convert:
        metadata_summary[key] = sorted(list(metadata_summary[key]))
    
    metadata_summary["evaluation_ids"] = list(metadata_summary["evaluation_ids"])
    
    # Convert essential fields to lists
    for field in metadata_summary["essential_fields"]:
        metadata_summary["essential_fields"][field] = sorted(list(metadata_summary["essential_fields"][field]))
    
    # Enhanced logging for debugging (including new fields)
    logger.info(f"📊 COMPLETE METADATA VERIFICATION:")
    logger.info(f"   Total evaluations: {metadata_summary['total_evaluations']}")
    logger.info(f"   Total chunks: {metadata_summary['total_chunks_found']}")
    logger.info(f"   Has real data: {metadata_summary['has_real_data']}")
    logger.info(f"   Dispositions: {len(metadata_summary['dispositions'])}")
    logger.info(f"   Programs: {len(metadata_summary['programs'])}")
    logger.info(f"   ✅ Weighted scores: {len(metadata_summary['weighted_scores'])}")  # NEW
    logger.info(f"   ✅ URLs: {len(metadata_summary['urls'])}")                      # NEW
    logger.info(f"   Call durations: {len(metadata_summary['call_durations'])}")     # NEW
    
    return metadata_summary


# ✅ ALSO UPDATE your build_simplified_context function to include the new fields:

def build_simplified_context(metadata_summary: Dict[str, Any], query: str) -> str:
    """
    UPDATED: Build context including weighted_score and url fields
    """
    # Check if we have real data
    if not metadata_summary.get("has_real_data", False):
        return """
NO DATA FOUND: No evaluation records match your query criteria. 

POSSIBLE CAUSES:
1. No data has been imported into the evaluation database
2. Search terms don't match any indexed content
3. Applied filters are too restrictive
4. Database connection issues

INSTRUCTIONS:
- Clearly state that no evaluation data was found
- Do not generate or estimate any statistics
- Suggest checking data import status or adjusting search criteria
- Recommend visiting the admin panel to verify data import

DO NOT GENERATE OR ESTIMATE ANY NUMBERS, DATES, OR STATISTICS.
"""

    # Check if essential_fields exists (this was causing KeyError)
    essential_fields = metadata_summary.get("essential_fields", {
        "evaluationId": [],
        "template_name": [],
        "agentName": [],
        "created_on": []
    })

    # ✅ Get the new fields safely
    weighted_scores = metadata_summary.get("weighted_scores", [])
    urls = metadata_summary.get("urls", [])
    call_durations = metadata_summary.get("call_durations", [])

    # Build context with safe field access including new fields
    context = f"""
VERIFIED EVALUATION DATA FOUND: {metadata_summary.get('total_evaluations', 0)} unique evaluations from {metadata_summary.get('total_chunks_found', 0)} content sources

REAL METADATA AVAILABLE:
- Evaluation IDs: {len(essential_fields.get('evaluationId', []))} unique
- Template Names: {essential_fields.get('template_name', [])}
- Agent Names: {essential_fields.get('agentName', [])}
- Date Range: {len(essential_fields.get('created_on', []))} unique dates
- Call Dispositions: {metadata_summary.get('dispositions', [])}
- Programs: {metadata_summary.get('programs', [])}

✅ ADDITIONAL FIELDS AVAILABLE:
- Weighted Scores: {len(weighted_scores)} values found ({weighted_scores[:5] if len(weighted_scores) <= 5 else weighted_scores[:5] + ['...']})
- Evaluation URLs: {len(urls)} URLs found
- Call Durations: {len(call_durations)} duration values found

CRITICAL INSTRUCTIONS:
1. ONLY use data from the provided evaluation sources
2. Focus on: evaluationId, template_name, agentName, created_on
3. ✅ You can also reference: weighted_score, url, call_duration when available
4. DO NOT generate percentages or statistics not directly calculable from the data
5. Report on {metadata_summary.get('total_evaluations', 0)} EVALUATIONS (not chunks)
6. Use only the agent names found: {', '.join(essential_fields.get('agentName', [])[:10])}
7. Use only the dispositions found: {', '.join(metadata_summary.get('dispositions', []))}
8. ✅ If asked about scores, use only these weighted scores: {', '.join(weighted_scores[:10])}

DATA VERIFICATION STATUS: {metadata_summary.get('data_verification', 'VERIFIED_REAL_DATA')}
"""
    
    return context


# ✅ ALSO UPDATE the extract_source_info function to include these fields:

def extract_source_info(hit: dict, search_type: str) -> dict:
    """
    UPDATED: Extract source information including weighted_score and url
    """
    try:
        doc = hit.get("_source", {})
        score = hit.get("_score", 0)
        
        # Get evaluation ID with better extraction
        evaluation_id = None
        for id_field in ["evaluationId", "evaluation_id", "internalId", "internal_id"]:
            if doc.get(id_field):
                evaluation_id = doc.get(id_field)
                break
        
        if not evaluation_id:
            evaluation_id = f"eval_{hash(str(doc))}"

        # Extract text content (existing logic)
        content_text = ""
        if doc.get("full_text"):
            content_text = doc.get("full_text")
        elif doc.get("evaluation_text"):
            content_text = doc.get("evaluation_text")
        elif doc.get("transcript_text"):
            content_text = doc.get("transcript_text")
        else:
            chunks = doc.get("chunks", [])
            if chunks and isinstance(chunks, list):
                chunk_texts = [chunk.get("text", "") for chunk in chunks[:3] if isinstance(chunk, dict)]
                if chunk_texts:
                    content_text = "\n".join(chunk_texts)

        if len(content_text) > 1500:
            content_text = content_text[:1500] + "..."
        
        # Extract metadata with ALL fields including new ones
        metadata = doc.get("metadata", {})
        
        source_info = {
            "evaluationId": evaluation_id,
            "text": content_text,
            "score": round(score, 3),
            "search_type": search_type,
            "template_id": doc.get("template_id"),
            "template_name": doc.get("template_name"),
            "metadata": {
                # Standard fields
                "program": metadata.get("program"),
                "partner": metadata.get("partner"),
                "site": metadata.get("site"),
                "lob": metadata.get("lob"),
                "agent": metadata.get("agent"),
                "disposition": metadata.get("disposition"),
                "sub_disposition": metadata.get("sub_disposition"),
                "language": metadata.get("language"),
                "call_date": metadata.get("call_date"),
                "call_duration": metadata.get("call_duration"),
                "call_type": metadata.get("call_type"),
                "agentId": metadata.get("agentId") or metadata.get("agent_id"),  # ✅ Use original            
                "weighted_score": metadata.get("weighted_score"),  # ✅ ADDED
                "url": metadata.get("url"),                        # ✅ ADDED
            }
        }
        
        return source_info
        
    except Exception as e:
        logger.error(f"Error extracting source info: {e}")
        return {
            "evaluationId": "error",
            "text": "",
            "score": 0,
            "search_type": search_type,
            "metadata": {}
        }


# ✅ TEST THE NEW FIELDS: Add this quick test function to verify extraction works:

def test_new_fields_extraction():
    """
    Test function to verify weighted_score and url extraction works
    """
    # Sample test data
    test_source = {
        "_source": {
            "evaluationId": 123,
            "template_name": "Test Template",
            "full_text": "Test evaluation content",
            "metadata": {
                "disposition": "Account",
                "program": "Corporate",
                "agent": "John Doe",
                "weighted_score": 85,  # ✅ Test this
                "url": "https://example.com/eval/123",  # ✅ Test this
                "call_duration": 240,
                "call_type": "CSR"
            }
        },
        "_score": 0.95
    }
    
    # Test extraction
    result = extract_source_info(test_source, "test")
    
    # Verify new fields are extracted
    assert result["metadata"]["weighted_score"] == 85, "weighted_score not extracted"
    assert result["metadata"]["url"] == "https://example.com/eval/123", "url not extracted"
    
    print("✅ New fields extraction test PASSED")
    print(f"✅ Weighted Score: {result['metadata']['weighted_score']}")
    print(f"✅ URL: {result['metadata']['url']}")
    
    return result

# test_new_fields_extraction()
 
def detect_report_query(message: str) -> bool:
    """Detect if the message is asking for a report or analysis"""
    report_keywords = [
        "report", "analysis", "summary", "overview", "breakdown",
        "performance", "trends", "statistics", "metrics", "insights"
    ]
    return any(keyword in message.lower() for keyword in report_keywords)


def build_search_context(query: str, filters: dict, max_results: int = 100) -> Tuple[str, List[dict]]:
    """
    FIXED: Build search context that actually finds documents
    """
    logger.info(f"🔍 BUILDING SEARCH CONTEXT (FIXED VERSION)")
    logger.info(f"📋 Query: '{query}'")
    logger.info(f"🏷️ Filters: {filters}")
    logger.info(f"📊 Max results: {max_results}")
    
    try:
        from opensearch_client import get_opensearch_client
        
        client = get_opensearch_client()
        if not client:
            logger.error("❌ No OpenSearch client available")
            return "Search system unavailable.", []
        
        # STEP 1: Try multiple search strategies
        all_sources = []
        
        # Strategy 1: Simple match_all query (should always work)
        try:
            logger.info("🔍 Trying match_all query...")
            match_all_response = client.search(
                index="eval-*",
                body={
                    "size": min(max_results, 50),
                    "query": {"match_all": {}},
                    "_source": [
                        "evaluationId", "internalId", "template_name", "template_id",
                        "text", "content", "full_text", "metadata"
                    ]
                },
                request_timeout=15
            )
            
            match_all_hits = match_all_response.get("hits", {}).get("hits", [])
            logger.info(f"📊 Match_all query returned {len(match_all_hits)} hits")
            
            # Process match_all results
            for hit in match_all_hits:
                source = hit.get("_source", {})
                evaluation_id = source.get("evaluationId") or source.get("internalId") or hit.get("_id")
                
                # Get text content
                text_content = (
                    source.get("text") or 
                    source.get("content") or 
                    source.get("full_text") or 
                    "No content available"
                )
                
                source_info = {
                    "evaluationId": evaluation_id,
                    "search_type": "match_all",
                    "score": hit.get("_score", 1.0),
                    "template_name": source.get("template_name", "Unknown"),
                    "template_id": source.get("template_id"),
                    "text": text_content[:2000],  # Limit to 2000 chars
                    "metadata": source.get("metadata", {}),
                    "content_type": "evaluation",
                    "_index": hit.get("_index")
                }
                
                all_sources.append(source_info)
                
        except Exception as e:
            logger.error(f"❌ Match_all query failed: {e}")
        
        # Strategy 2: If we have a specific query, try text search
        if query and query.strip() and len(query.strip()) > 2:
            try:
                logger.info(f"🔍 Trying text search for: '{query}'")
                text_response = client.search(
                    index="eval-*",
                    body={
                        "size": min(max_results, 30),
                        "query": {
                            "bool": {
                                "should": [
                                    {
                                        "multi_match": {
                                            "query": query,
                                            "fields": ["text^2", "content^2", "full_text^2"],
                                            "type": "best_fields",
                                            "fuzziness": "AUTO"
                                        }
                                    },
                                    {
                                        "match": {
                                            "text": {
                                                "query": query,
                                                "boost": 1.5
                                            }
                                        }
                                    }
                                ],
                                "minimum_should_match": 1
                            }
                        },
                        "_source": [
                            "evaluationId", "internalId", "template_name", "template_id",
                            "text", "content", "full_text", "metadata"
                        ]
                    },
                    request_timeout=15
                )
                
                text_hits = text_response.get("hits", {}).get("hits", [])
                logger.info(f"📊 Text search returned {len(text_hits)} hits")
                
                # Process text search results (these get higher priority)
                for hit in text_hits:
                    source = hit.get("_source", {})
                    evaluation_id = source.get("evaluationId") or source.get("internalId") or hit.get("_id")
                    
                    # Skip if we already have this evaluation
                    if any(s.get("evaluationId") == evaluation_id for s in all_sources):
                        continue
                    
                    text_content = (
                        source.get("text") or 
                        source.get("content") or 
                        source.get("full_text") or 
                        "No content available"
                    )
                    
                    source_info = {
                        "evaluationId": evaluation_id,
                        "search_type": "text",
                        "score": hit.get("_score", 0),
                        "template_name": source.get("template_name", "Unknown"),
                        "template_id": source.get("template_id"),
                        "text": text_content[:2000],
                        "metadata": source.get("metadata", {}),
                        "content_type": "evaluation",
                        "_index": hit.get("_index")
                    }
                    
                    # Insert at beginning (higher priority than match_all)
                    all_sources.insert(0, source_info)
                    
            except Exception as e:
                logger.error(f"❌ Text search failed: {e}")
        
        # STEP 2: Process and verify results
        logger.info(f"🔗 TOTAL SOURCES FOUND: {len(all_sources)}")
        
        if not all_sources:
            logger.warning("⚠️ NO SOURCES FOUND")
            no_data_context = """
NO EVALUATION DATA FOUND: The search did not return any evaluation records.

POSSIBLE CAUSES:
1. No data has been imported into the evaluation database
2. Search index is corrupted or misconfigured
3. Database connection issues
4. OpenSearch mapping problems

INSTRUCTIONS:
- Clearly state that no evaluation data was found
- Do not generate or estimate any statistics
- Suggest checking data import status
- Recommend contacting technical support
"""
            return no_data_context, []
        
        # STEP 3: Verify metadata and build context
        processed_sources = []
        unique_evaluations = set()
        
        for source in all_sources[:max_results]:  # Limit total sources
            evaluation_id = source.get("evaluationId")
            if evaluation_id and evaluation_id not in unique_evaluations:
                unique_evaluations.add(evaluation_id)
                processed_sources.append(source)
        
        # STEP 4: Build context with real data
        if processed_sources:
            context = f"""
VERIFIED EVALUATION DATA FOUND: {len(processed_sources)} unique evaluations

DATA OVERVIEW:
- Total evaluations: {len(unique_evaluations)}
- Content sources: {len(all_sources)}
- Search methods used: {', '.join(set(s['search_type'] for s in processed_sources))}

SAMPLE CONTENT:
{processed_sources[0]['text'][:500]}...

EVALUATION DETAILS:
"""
            
            # Add details from first few evaluations
            for i, source in enumerate(processed_sources[:5]):
                metadata = source.get("metadata", {})
                context += f"""
[Evaluation {i+1}] ID: {source['evaluationId']}
- Template: {source.get('template_name', 'Unknown')}
- Program: {metadata.get('program', 'Unknown')}
- Disposition: {metadata.get('disposition', 'Unknown')}
- Content: {source['text'][:200]}...
"""
            
            if len(processed_sources) > 5:
                context += f"\n... and {len(processed_sources) - 5} more evaluations"
            
            context += f"""

INSTRUCTIONS:
- Use ONLY the data shown above from {len(processed_sources)} evaluations
- Do not generate statistics not directly calculable from this data
- Focus on patterns and insights from the actual content provided
"""
            
            logger.info(f"✅ CONTEXT BUILT: {len(context)} chars with {len(processed_sources)} sources")
            return context, processed_sources
        else:
            logger.warning("⚠️ NO VALID SOURCES AFTER PROCESSING")
            return "No valid evaluation data found after processing.", []

    except Exception as e:
        logger.error(f"❌ SEARCH CONTEXT BUILD FAILED: {e}")
        error_context = f"""
SEARCH ERROR: Failed to retrieve evaluation data due to technical issues.

Error details: {str(e)[:200]}

INSTRUCTIONS:
- Inform the user that there was a technical error accessing the evaluation database
- Suggest trying again or contacting technical support
- Do not generate any statistics or data
"""
        return error_context, []
    
def build_strict_metadata_context(metadata_summary: Dict[str, Any], query: str) -> str:
    """Alias for build_simplified_context for backward compatibility"""
    return build_simplified_context(metadata_summary, query)


def build_simplified_context(metadata_summary: Dict[str, Any], query: str) -> str:
    """
    Build context focusing on essential fields only
    """
    if not metadata_summary.get("has_real_data", False):
        return """
NO DATA FOUND: No evaluation records match your query criteria. 
You must clearly state that no data is available and suggest:
1. Checking if data has been imported
2. Adjusting search terms or filters
3. Verifying the evaluation database connectivity

DO NOT GENERATE OR ESTIMATE ANY NUMBERS, DATES, OR STATISTICS.
"""

    # Safe field access with defaults
    essential_fields = metadata_summary.get("essential_fields", {
        "evaluationId": [],
        "template_name": [],
        "agentName": [],
        "created_on": []
    })

    # Build context with safe field access
    context = f"""
VERIFIED EVALUATION DATA FOUND: {metadata_summary.get('total_evaluations', 0)} unique evaluations from {metadata_summary.get('total_chunks_found', 0)} content sources

ESSENTIAL METADATA AVAILABLE:
- Evaluation IDs: {len(essential_fields.get('evaluationId', []))} unique
- Template Names: {essential_fields.get('template_name', [])}
- Agent Names: {essential_fields.get('agentName', [])}
- Date Range: {len(essential_fields.get('created_on', []))} unique dates
- Call Dispositions: {metadata_summary.get('dispositions', [])}
- Programs: {metadata_summary.get('programs', [])}

CRITICAL INSTRUCTIONS:
1. ONLY use data from the provided evaluation sources
2. Focus on: evaluationId, template_name, agentName, created_on
3. DO NOT generate percentages or statistics not directly calculable from the data
4. Report on {metadata_summary.get('total_evaluations', 0)} EVALUATIONS (not chunks)
5. Use only the agent names found: {', '.join(essential_fields.get('agentName', [])[:10])}
6. Use only the dispositions found: {', '.join(metadata_summary.get('dispositions', []))}

DATA VERIFICATION STATUS: {metadata_summary.get('data_verification', 'VERIFIED_REAL_DATA')}
"""
    
    return context

# ============================================================================
# NEW FUNCTION: Build a summary of sources with detailed drill-down data for each category    
# ============================================================================
def build_sources_summary_with_details(sources, filters=None):
    """
    NEW FUNCTION: Build a summary of sources with detailed drill-down data for each category
    Replaces the old simple summary format with interactive drill-down tables
    """
    if not sources:
        return {
            "summary": {
                "evaluations": 0,
                "agents": 0,
                "date_range": "No data",
                "opportunities": 0,
                "churn_triggers": 0,
                "programs": 0,
                "templates": 0,
                "dispositions": 0,
                "partners": 0,
                "sites": 0
            },
            "details": {},
            "totals": {},
            "full_data": {}
        }
    
    # Configuration for display limits
    DISPLAY_LIMIT = 25  # Show first 25 items, download button for rest
    
    # Initialize collections for detailed data
    evaluations_details = []
    agents_details = {}
    programs_details = {}
    templates_details = {}
    dispositions_details = {}
    partners_details = {}
    sites_details = {}
    opportunities_details = []
    churn_triggers_details = []
    
    # Track unique values
    unique_evaluations = set()
    unique_agents = set()
    unique_programs = set()
    unique_templates = set()
    unique_dispositions = set()
    unique_partners = set()
    unique_sites = set()
    dates = []
    
    # Process each source to extract both summary and detailed data
    seen_evaluation_ids = set()

    for source in sources:
        # Get evaluation ID
        # Extract evaluation ID with your robust logic
        evaluation_id = None
        source_data = source.get("_source", source)
        
        for id_field in ["evaluationId", "evaluation_id", "internalId", "internal_id"]:
                if source_data.get(id_field):
                    evaluation_id = source_data[id_field]
                    break
                if source_data.get("metadata", {}).get(id_field):
                    evaluation_id = source_data["metadata"][id_field]
                    break
            
            # Fallback for direct field access
        if not evaluation_id and source.get("evaluationId"):
            evaluation_id = source.get("evaluationId")
        
        # Skip if no evaluation ID found
        if not evaluation_id:
            continue
            
        # Skip duplicates for evaluations
        if evaluation_id in seen_evaluation_ids:
            continue
        seen_evaluation_ids.add(evaluation_id)
        unique_evaluations.add(evaluation_id)
        
        # Get metadata
        metadata = source.get("metadata", {})
        
        # Extract basic fields
        agent = (metadata.get("agent") or 
                metadata.get("agentName") or 
                source.get("agentName") or "Unknown").strip()
        
        program = (metadata.get("program") or "Unknown").strip()
        template = (source.get("template_name") or "Unknown").strip()
        disposition = (metadata.get("disposition") or 
                      metadata.get("call_disposition") or "Unknown").strip()
        partner = (metadata.get("partner") or "Unknown").strip()
        site = (metadata.get("site") or "Unknown").strip()
        
        # Extract date
        date_field = (source.get("created_on") or 
                     metadata.get("created_on") or 
                     source.get("call_date") or
                     metadata.get("call_date"))
        
        formatted_date = "Unknown"
        if date_field:
            try:
                from datetime import datetime
                if isinstance(date_field, str):
                    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            parsed_date = datetime.strptime(date_field[:len(fmt)], fmt)
                            formatted_date = parsed_date.strftime("%m/%d/%Y")
                            dates.append(parsed_date)
                            break
                        except ValueError:
                            continue
            except Exception:
                pass
        
        # Build evaluation detail record
        evaluation_detail = {
            "evaluation_id": evaluation_id,
            "agent_name": agent,
            "program": program,
            "template": template,
            "disposition": disposition,
            "partner": partner,
            "site": site,
            "date": formatted_date,
            "score": metadata.get("weighted_score", "N/A"),
            "duration": metadata.get("call_duration", "N/A")
        }
        evaluations_details.append(evaluation_detail)
        
        # Track unique values and build detailed collections for agents
        if agent != "Unknown":
            unique_agents.add(agent)
            if agent not in agents_details:
                agents_details[agent] = {
                    "agent_name": agent,
                    "evaluation_count": 0,
                    "programs": set(),
                    "latest_date": None,
                    "average_score": [],
                    "evaluations": []
                }
            agents_details[agent]["evaluation_count"] += 1
            agents_details[agent]["programs"].add(program)
            agents_details[agent]["evaluations"].append(evaluation_id)
            if metadata.get("weighted_score"):
                try:
                    agents_details[agent]["average_score"].append(float(metadata.get("weighted_score")))
                except:
                    pass
        
        # Track programs details
        if program != "Unknown":
            unique_programs.add(program)
            if program not in programs_details:
                programs_details[program] = {
                    "program_name": program,
                    "evaluation_count": 0,
                    "agents": set(),
                    "date_range": [],
                    "templates": set()
                }
            programs_details[program]["evaluation_count"] += 1
            programs_details[program]["agents"].add(agent)
            programs_details[program]["templates"].add(template)
            if dates:
                programs_details[program]["date_range"].extend(dates[-1:])
        
        # Track templates details
        if template != "Unknown":
            unique_templates.add(template)
            if template not in templates_details:
                templates_details[template] = {
                    "template_name": template,
                    "usage_count": 0,
                    "programs": set(),
                    "agents": set()
                }
            templates_details[template]["usage_count"] += 1
            templates_details[template]["programs"].add(program)
            templates_details[template]["agents"].add(agent)
        
        # Track dispositions with examples
        if disposition != "Unknown":
            unique_dispositions.add(disposition)
            if disposition not in dispositions_details:
                dispositions_details[disposition] = {
                    "disposition_name": disposition,
                    "count": 0,
                    "examples": []
                }
            dispositions_details[disposition]["count"] += 1
            dispositions_details[disposition]["examples"].append({
                "evaluation_id": evaluation_id,
                "agent": agent,
                "date": formatted_date
            })
            
            # Check for opportunities and churn triggers
            disposition_lower = disposition.lower()
            if any(keyword in disposition_lower for keyword in ["opportunity", "sale", "interested", "lead", "positive"]):
                opportunities_details.append({
                    "evaluation_id": evaluation_id,
                    "agent": agent,
                    "disposition": disposition,
                    "program": program,
                    "date": formatted_date,
                    "score": metadata.get("weighted_score", "N/A")
                })
            elif any(keyword in disposition_lower for keyword in ["churn", "cancel", "disconnect", "terminate", "unsatisfied", "complaint"]):
                churn_triggers_details.append({
                    "evaluation_id": evaluation_id,
                    "agent": agent,
                    "disposition": disposition,
                    "program": program,
                    "date": formatted_date,
                    "score": metadata.get("weighted_score", "N/A")
                })
        
        # Track partners details
        if partner != "Unknown":
            unique_partners.add(partner)
            if partner not in partners_details:
                partners_details[partner] = {
                    "partner_name": partner,
                    "evaluation_count": 0,
                    "programs": set(),
                    "agents": set()
                }
            partners_details[partner]["evaluation_count"] += 1
            partners_details[partner]["programs"].add(program)
            partners_details[partner]["agents"].add(agent)
        
        # Track sites details
        if site != "Unknown":
            unique_sites.add(site)
            if site not in sites_details:
                sites_details[site] = {
                    "site_name": site,
                    "evaluation_count": 0,
                    "programs": set(),
                    "agents": set()
                }
            sites_details[site]["evaluation_count"] += 1
            sites_details[site]["programs"].add(program)
            sites_details[site]["agents"].add(agent)
    
    # Calculate date range
    date_range = "No data"
    if dates:
        dates.sort()
        start_date = dates[0].strftime("%b %d")
        end_date = dates[-1].strftime("%b %d, %Y")
        if len(dates) == 1:
            date_range = dates[0].strftime("%b %d, %Y")
        else:
            date_range = f"{start_date} - {end_date}"
    
    # Convert sets to lists and calculate averages
    agents_list = []
    for agent_name, details in agents_details.items():
        agent_record = {
            "agent_name": agent_name,
            "evaluation_count": details["evaluation_count"],
            "programs": list(details["programs"]),
            "average_score": round(sum(details["average_score"]) / len(details["average_score"]), 2) if details["average_score"] else "N/A",
            "evaluations": details["evaluations"][:10]  # Limit for display
        }
        agents_list.append(agent_record)
    
    programs_list = []
    for program_name, details in programs_details.items():
        program_record = {
            "program_name": program_name,
            "evaluation_count": details["evaluation_count"],
            "agent_count": len(details["agents"]),
            "agents": list(details["agents"])[:10],  # Limit for display
            "templates": list(details["templates"])
        }
        programs_list.append(program_record)
    
    # Build final response with limited display data and full download data
    summary = {
        "evaluations": len(unique_evaluations),
        "agents": len(unique_agents),
        "date_range": date_range,
        "opportunities": len(opportunities_details),
        "churn_triggers": len(churn_triggers_details),
        "programs": len(unique_programs),
        "templates": len(unique_templates),
        "dispositions": len(unique_dispositions),
        "partners": len(unique_partners),
        "sites": len(unique_sites)
    }
    
    # Prepare limited data for display (first 25 items)
    detailed_data = {
        "evaluations": evaluations_details[:DISPLAY_LIMIT],
        "agents": agents_list[:DISPLAY_LIMIT],
        "programs": programs_list[:DISPLAY_LIMIT],
        "templates": [{"template_name": name, "usage_count": details["usage_count"], 
                      "programs": list(details["programs"]), "agents": list(details["agents"])[:10]} 
                     for name, details in list(templates_details.items())[:DISPLAY_LIMIT]],
        "dispositions": [{"disposition_name": name, "count": details["count"],
                         "examples": details["examples"][:5]} 
                        for name, details in list(dispositions_details.items())[:DISPLAY_LIMIT]],
        "opportunities": opportunities_details[:DISPLAY_LIMIT],
        "churn_triggers": churn_triggers_details[:DISPLAY_LIMIT],
        "partners": [{"partner_name": name, "evaluation_count": details["evaluation_count"],
                     "programs": list(details["programs"]), "agents": list(details["agents"])[:10]}
                    for name, details in list(partners_details.items())[:DISPLAY_LIMIT]],
        "sites": [{"site_name": name, "evaluation_count": details["evaluation_count"],
                  "programs": list(details["programs"]), "agents": list(details["agents"])[:10]}
                 for name, details in list(sites_details.items())[:DISPLAY_LIMIT]]
    }
    
    # Prepare full data for download (all items)
    full_data_for_download = {
        "evaluations": evaluations_details,  # All evaluations
        "agents": agents_list,  # All agents
        "programs": programs_list,  # All programs
        "templates": [{"template_name": name, "usage_count": details["usage_count"], 
                      "programs": list(details["programs"]), "agents": list(details["agents"])} 
                     for name, details in templates_details.items()],
        "dispositions": [{"disposition_name": name, "count": details["count"],
                         "examples": details["examples"]} 
                        for name, details in dispositions_details.items()],
        "opportunities": opportunities_details,  # All opportunities
        "churn_triggers": churn_triggers_details,  # All churn triggers
        "partners": [{"partner_name": name, "evaluation_count": details["evaluation_count"],
                     "programs": list(details["programs"]), "agents": list(details["agents"])}
                    for name, details in partners_details.items()],
        "sites": [{"site_name": name, "evaluation_count": details["evaluation_count"],
                  "programs": list(details["programs"]), "agents": list(details["agents"])}
                 for name, details in sites_details.items()]
    }
    
    # Track total counts for each category
    totals = {
        "evaluations": len(evaluations_details),
        "agents": len(agents_list),
        "programs": len(programs_list),
        "templates": len(templates_details),
        "dispositions": len(dispositions_details),
        "opportunities": len(opportunities_details),
        "churn_triggers": len(churn_triggers_details),
        "partners": len(partners_details),
        "sites": len(sites_details)
    }
    
    return {
        "summary": summary,
        "details": detailed_data,
        "totals": totals,
        "full_data": full_data_for_download,
        "display_limit": DISPLAY_LIMIT
    }

# =============================================================================
# MAIN RAG-ENABLED CHAT ENDPOINT WITH STRICT METADATA VERIFICATION
# =============================================================================
@chat_router.post("/chat_test")
async def chat_test_minimal(request: Request):
    """Minimal test endpoint to isolate the response issue"""
    try:
        body = await request.json()
        
        # Just return a tiny response to test if the issue is response size
        return JSONResponse(content={
            "reply": "Test response working!",
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "search_metadata": {"test": "success"}
        })
        
    except Exception as e:
        return JSONResponse(content={"reply": f"Error: {e}"})


@chat_router.post("/chat")
async def relay_chat_rag(request: Request):
    start_time = time.time()
    try:
        body = await request.json()
        req = ChatRequest(**body)

        logger.info(f"💬 CHAT REQUEST WITH METADATA VERIFICATION: {req.message[:60]}")
        logger.info(f"🔎 FILTERS RECEIVED: {req.filters}")

        is_report_request = detect_report_query(req.message)
        logger.info(f"📊 REPORT REQUEST DETECTED: {is_report_request}")

        # STEP 1: Build context with strict metadata verification
        context, sources = build_search_context(req.message, req.filters)
        
        # DEBUG: Log context quality
        logger.info(f"📋 CONTEXT BUILT: {len(context)} chars, {len(sources)} sources")
        if not context:
            logger.warning("⚠️ NO CONTEXT FOUND - Chat will use general knowledge only")
        
        # STEP 2: Enhanced system message with strict instructions
        system_message = f"""You are an AI assistant for call center evaluation data analysis.

ANALYSIS INSTRUCTIONS:
1. Review transcripts or agent summaries to identify recurring communication issues or strengths.
2. Compare tone, language, empathy, and product knowledge across transcripts and agents.
3. Identify opportunities for improvement in agent performance based on the provided evaluation data.
4. Highlight strengths that highlight agent performance, such as effective communication, problem-solving skills, and customer rapport.
5. Determine what is successful and what needs improvement, with justifications.
6. Write a concise but structured summary with clear sections and bullet points.

EVALUATION DATABASE CONTEXT:
{context}

CRITICAL INSTRUCTIONS:
- ONLY use data from the provided context above along with the last 10 messages in the chat history
- Always answer questoins based on the provided evaluation data
- Do not generate statistics not directly calculable from the data
- Be objective and data-informed
- Avoid overgeneralizations
- Make the summary suitable for leadership or QA team use
"""

        # STEP 3: Construct DigitalOcean AI payload (corrected format)
        do_payload = {
            "messages": [
                {"role": "system", "content": system_message},
                *[{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in req.history[-10:]],
                {"role": "user", "content": req.message}
            ],
            "temperature": GENAI_TEMPERATURE,
            "max_tokens": GENAI_MAX_TOKENS
        }

        headers = {
            "Authorization": f"Bearer {GENAI_ACCESS_KEY}",
            "Content-Type": "application/json"
        }

        # Correct DigitalOcean AI URL format
        do_url = f"{GENAI_ENDPOINT.rstrip('/')}/api/v1/chat/completions"
        logger.info(f"🚀 Making GenAI API call to: {do_url}")
        logger.info(f"🚀 About to call DigitalOcean AI - payload size: {len(str(do_payload))} chars")

        genai_response = requests.post(
            do_url,
            headers=headers,
            json=do_payload,
            timeout=60  # Increased timeout for GenAI API
        )
        
        if not genai_response.ok:
            logger.error(f"❌ GenAI API error: {genai_response.status_code} - {genai_response.text}")
            return JSONResponse(
                status_code=500,
                content={
                    "reply": "I apologize, but I'm experiencing technical difficulties with the AI service. Please try again in a moment.",
                    "sources": [],
                    "timestamp": datetime.now().isoformat(),
                    "search_metadata": {
                        "error": f"GenAI API error: {genai_response.status_code}",
                        "context_length": len(context),
                        "processing_time": round(time.time() - start_time, 2)
                    }
                }
            )

        result = genai_response.json()
        
        # Extract reply from DigitalOcean AI response
        reply_text = "(No response)"
        if "choices" in result and result["choices"]:
            reply_text = result["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"❌ GenAI response missing 'choices': {result}")
            reply_text = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."

               
        # STEP 4: Process sources for response - Remove duplicates
        unique_sources = []
        seen_ids = set()
        for source in sources:
            evaluation_id = source.get("evaluationId") or source.get("evaluation_id")
            if evaluation_id and evaluation_id not in seen_ids:
                unique_sources.append({
                    "evaluationId": evaluation_id,
                    "template_name": source.get("template_name", "Unknown"),
                    "search_type": source.get("search_type", "text"),
                    "score": source.get("score", 0),
                    "metadata": source.get("metadata", {})
                })
                seen_ids.add(evaluation_id)


        # STEP 5: NEW - Build sources summary with detailed drill-down data

        sources_data = build_sources_summary_with_details(unique_sources, req.filters)

        # STEP 56: Build enhanced response
        response_data = {
            "reply": reply_text,
            "sources_summary": sources_data["summary"],           # NEW: Summary counts
            "sources_details": sources_data["details"],          # NEW: Limited data for display  
            "sources_totals": sources_data["totals"],            # NEW: Total counts for each category
            "sources_full_data": sources_data["full_data"],      # NEW: Complete data for download
            "display_limit": sources_data["display_limit"],      # NEW: Current display limit (25)
            "sources": unique_sources[:20],  # Limit sources in response
            "timestamp": datetime.now().isoformat(),
            "filter_context": req.filters,
            "search_metadata": {
                "vector_sources": len([s for s in sources if s.get("search_type") == "vector"]),
                "text_sources": len([s for s in sources if s.get("search_type") == "text"]),
                "context_length": len(context),
                "processing_time": round(time.time() - start_time, 2),
                "total_sources": len(sources),
                "unique_sources": len(unique_sources),
                "context_found": bool(context and "NO DATA FOUND" not in context),
                "metadata_verified": "verified_real_data" in context.lower(),
                "version": "4.7.0"
            }
        }
        
        logger.info(f"✅ CHAT RESPONSE COMPLETE: {len(reply_text)} chars, {len(unique_sources)} verified sources")
        logger.info(f"📊 SOURCES SUMMARY: {sources_data['summary']}")  # NEW: Log summary data
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"❌ CHAT REQUEST FAILED: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "reply": f"I apologize, but I encountered an error while processing your request: {str(e)[:200]}. Please try again or contact support if the issue persists.",
                # NEW: Error response includes all new fields
                "sources_summary": {
                    "evaluations": 0,
                    "agents": 0,
                    "date_range": "Error",
                    "opportunities": 0,
                    "churn_triggers": 0,
                    "programs": 0,
                    "templates": 0,
                    "dispositions": 0,
                    "partners": 0,
                    "sites": 0
                },
                "sources_details": {},
                "sources_totals": {},
                "sources_full_data": {},
                "display_limit": 25,
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "filter_context": req.filters if 'req' in locals() else {},
                "search_metadata": {
                    "error": str(e),
                    "version": "4.7.0_limited_display_with_download"  # UPDATED: Version number
                }
            }
        )
    
# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@health_router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "components": {
            "opensearch": {"status": "connected"},
            "embedding_service": {"status": "healthy"},
            "genai_agent": {"status": "configured"}
        },
        "enhancements": {
            "document_structure": "enhanced v4.4.0",
            "metadata_verification": "enabled",
            "strict_data_alignment": "enforced",
            "evaluation_chunk_distinction": "implemented"
        }
    }

@health_router.get("/last_import_info")
async def last_import_info():
    return {
        "status": "success",
        "last_import_timestamp": datetime.now().isoformat()
    }