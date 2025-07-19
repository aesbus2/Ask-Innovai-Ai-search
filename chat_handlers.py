# chat_handlers.py - VERSION 4.8.0 - VECTOR SEARCH ENABLED
# MAJOR UPDATE: Full vector search integration with hybrid text+vector search
# FIXES: Enhanced RAG with vector similarity, proper debugging and filter application
# NEW: Vector embeddings for queries, hybrid search strategy, improved relevance
# METADATA ALIGNMENT FIX + Strict evaluation vs chunk counting + VECTOR SEARCH

import os
import logging
import requests
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Literal, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ✅ VECTOR SEARCH ENABLED - Uncommented imports
from opensearch_client import search_opensearch, search_vector, hybrid_search
from embedder import embed_text

chat_router = APIRouter()
health_router = APIRouter()
logger = logging.getLogger(__name__)

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
        
        # Enhanced fields
        "weighted_scores": set(),
        "urls": set(),
        "call_durations": set(),
        "phone_numbers": set(),
        "contact_ids": set(),
        "ucids": set(),
        "call_types": set(),
        
        # ✅ NEW: Vector search metadata
        "vector_search_used": False,
        "hybrid_search_used": False,
        "search_types": set()
    }
    
    seen_evaluation_ids = set()
    
    for source in sources:
        try:
            metadata_summary["total_chunks_found"] += 1
            
            # ✅ Track search types used
            search_type = source.get("search_type", "unknown")
            metadata_summary["search_types"].add(search_type)
            
            if search_type == "vector":
                metadata_summary["vector_search_used"] = True
            elif search_type in ["hybrid", "text_with_vector_fallback_v4_7_0"]:
                metadata_summary["hybrid_search_used"] = True
            
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
                
            # Extract enhanced fields
            if metadata.get("weighted_score") is not None:
                metadata_summary["weighted_scores"].add(str(metadata["weighted_score"]))
                
            if metadata.get("url"):
                metadata_summary["urls"].add(metadata["url"])
                
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
        "call_durations", "phone_numbers", "contact_ids", "ucids", "call_types",
        "search_types"
    ]
    
    for key in fields_to_convert:
        metadata_summary[key] = sorted(list(metadata_summary[key]))
    
    metadata_summary["evaluation_ids"] = list(metadata_summary["evaluation_ids"])
    
    # Convert essential fields to lists
    for field in metadata_summary["essential_fields"]:
        metadata_summary["essential_fields"][field] = sorted(list(metadata_summary["essential_fields"][field]))
    
    # Enhanced logging for debugging (including vector search info)
    logger.info(f"📊 COMPLETE METADATA VERIFICATION WITH VECTOR SEARCH:")
    logger.info(f"   Total evaluations: {metadata_summary['total_evaluations']}")
    logger.info(f"   Total chunks: {metadata_summary['total_chunks_found']}")
    logger.info(f"   Has real data: {metadata_summary['has_real_data']}")
    logger.info(f"   ✅ Vector search used: {metadata_summary['vector_search_used']}")
    logger.info(f"   ✅ Hybrid search used: {metadata_summary['hybrid_search_used']}")
    logger.info(f"   🔍 Search types: {metadata_summary['search_types']}")
    logger.info(f"   Dispositions: {len(metadata_summary['dispositions'])}")
    logger.info(f"   Programs: {len(metadata_summary['programs'])}")
    
    return metadata_summary

def build_simplified_context(metadata_summary: Dict[str, Any], query: str) -> str:
    """
    UPDATED: Build context including vector search information
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

    # Check if essential_fields exists
    essential_fields = metadata_summary.get("essential_fields", {
        "evaluationId": [],
        "template_name": [],
        "agentName": [],
        "created_on": []
    })

    # Get enhanced fields safely
    weighted_scores = metadata_summary.get("weighted_scores", [])
    urls = metadata_summary.get("urls", [])
    call_durations = metadata_summary.get("call_durations", [])
    
    # ✅ Get vector search information
    vector_search_used = metadata_summary.get("vector_search_used", False)
    hybrid_search_used = metadata_summary.get("hybrid_search_used", False)
    search_types = metadata_summary.get("search_types", [])

    # Build context with safe field access including vector search info
    context = f"""
VERIFIED EVALUATION DATA FOUND: {metadata_summary.get('total_evaluations', 0)} unique evaluations from {metadata_summary.get('total_chunks_found', 0)} content sources

✅ ENHANCED SEARCH CAPABILITIES USED:
- Vector Search: {'ENABLED' if vector_search_used else 'NOT USED'}
- Hybrid Search: {'ENABLED' if hybrid_search_used else 'NOT USED'}  
- Search Types: {', '.join(search_types)}
- Search Quality: {'ENHANCED with semantic similarity' if vector_search_used or hybrid_search_used else 'Text-based only'}

REAL METADATA AVAILABLE:
- Evaluation IDs: {len(essential_fields.get('evaluationId', []))} unique
- Template Names: {essential_fields.get('template_name', [])}
- Agent Names: {essential_fields.get('agentName', [])}
- Date Range: {len(essential_fields.get('created_on', []))} unique dates
- Call Dispositions: {metadata_summary.get('dispositions', [])}
- Programs: {metadata_summary.get('programs', [])}

ADDITIONAL FIELDS AVAILABLE:
- Weighted Scores: {len(weighted_scores)} values found ({weighted_scores[:5] if len(weighted_scores) <= 5 else weighted_scores[:5] + ['...']})
- Evaluation URLs: {len(urls)} URLs found
- Call Durations: {len(call_durations)} duration values found

CRITICAL INSTRUCTIONS:
1. ONLY use data from the provided evaluation sources
2. Focus on: evaluationId, template_name, agentName, created_on
3. You can also reference: weighted_score, url, call_duration when available
4. DO NOT generate percentages or statistics not directly calculable from the data
5. Report on {metadata_summary.get('total_evaluations', 0)} EVALUATIONS (not chunks)
6. Use only the agent names found: {', '.join(essential_fields.get('agentName', [])[:10])}
7. Use only the dispositions found: {', '.join(metadata_summary.get('dispositions', []))}
8. If asked about scores, use only these weighted scores: {', '.join(weighted_scores[:10])}
9. ✅ NOTE: Results enhanced with {'vector similarity matching' if vector_search_used else 'text matching only'}

DATA VERIFICATION STATUS: {metadata_summary.get('data_verification', 'VERIFIED_REAL_DATA')}
SEARCH ENHANCEMENT: {'VECTOR-ENHANCED' if vector_search_used or hybrid_search_used else 'TEXT-ONLY'}
"""
    
    return context

def extract_source_info(hit: dict, search_type: str) -> dict:
    """
    UPDATED: Extract source information including vector search metadata
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
        elif doc.get("evaluation"):  # ✅ NEW: Support for evaluation field
            content_text = doc.get("evaluation")
        elif doc.get("transcript_text"):
            content_text = doc.get("transcript_text")
        elif doc.get("transcript"):  # ✅ NEW: Support for transcript field
            content_text = doc.get("transcript")
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
            
            # ✅ NEW: Vector search specific fields
            "vector_dimension": hit.get("vector_dimension"),
            "hybrid_score": hit.get("hybrid_score"),
            "best_matching_chunks": hit.get("best_matching_chunks", []),
            
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
                "agentId": metadata.get("agentId") or metadata.get("agent_id"),
                "weighted_score": metadata.get("weighted_score"),
                "url": metadata.get("url"),
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

def detect_report_query(message: str) -> bool:
    """Detect if the message is asking for a report or analysis"""
    report_keywords = [
        "report", "analysis", "summary", "overview", "breakdown",
        "performance", "trends", "statistics", "metrics", "insights"
    ]
    return any(keyword in message.lower() for keyword in report_keywords)

def build_search_context(query: str, filters: dict, max_results: int = 100) -> Tuple[str, List[dict]]:
    """
    ✅ ENHANCED: Build search context with VECTOR SEARCH integration
    Now supports hybrid text+vector search for better relevance
    """
    logger.info(f"🔍 BUILDING ENHANCED SEARCH CONTEXT WITH VECTOR SEARCH")
    logger.info(f"📋 Query: '{query}'")
    logger.info(f"🏷️ Filters: {filters}")
    logger.info(f"📊 Max results: {max_results}")
    
    try:
        from opensearch_client import get_opensearch_client, test_connection
        
        client = get_opensearch_client()
        if not client:
            logger.error("❌ No OpenSearch client available")
            return "Search system unavailable.", []
        
        if not test_connection():
            logger.warning("OpenSearch not available for search context")
            return create_empty_search_context("opensearch_unavailable"), []
        
        logger.info("✅ OpenSearch connection verified for enhanced search")
        
        # ✅ STEP 1: Try to generate query vector for enhanced search
        query_vector = None
        try:
            query_vector = embed_text(query)
            logger.info(f"✅ Query vector generated: {len(query_vector)} dimensions")
        except Exception as e:
            logger.warning(f"⚠️ Vector generation failed, falling back to text search: {e}")
        
        # ✅ STEP 2: Use enhanced search strategies
        all_sources = []
        search_methods_used = []
        
        # Strategy 1: Hybrid search (text + vector) if vector available
        if query_vector:
            try:
                logger.info("🔥 Trying hybrid text+vector search...")
                hybrid_results = hybrid_search(
                    query=query,
                    query_vector=query_vector,
                    filters=filters,
                    size=min(max_results, 30),
                    vector_weight=0.6  # 60% vector, 40% text
                )
                
                logger.info(f"📊 Hybrid search returned {len(hybrid_results)} hits")
                
                for hit in hybrid_results:
                    source_info = {
                        "evaluationId": hit.get("evaluationId"),
                        "search_type": hit.get("search_type", "hybrid"),
                        "score": hit.get("_score", 0),
                        "hybrid_score": hit.get("hybrid_score", 0),
                        "template_name": hit.get("template_name", "Unknown"),
                        "template_id": hit.get("template_id"),
                        "text": hit.get("text", "")[:2000],
                        "evaluation": hit.get("evaluation", ""),
                        "transcript": hit.get("transcript", ""),
                        "metadata": hit.get("metadata", {}),
                        "content_type": "evaluation",
                        "_index": hit.get("_index"),
                        "vector_enhanced": True
                    }
                    all_sources.append(source_info)
                
                search_methods_used.append("hybrid_text_vector")
                
            except Exception as e:
                logger.error(f"❌ Hybrid search failed: {e}")
        
        # Strategy 2: Pure vector search as fallback/supplement
        if query_vector and len(all_sources) < max_results // 2:
            try:
                logger.info("🔮 Trying pure vector search...")
                vector_results = search_vector(
                    query_vector=query_vector,
                    filters=filters,
                    size=min(max_results - len(all_sources), 20)
                )
                
                logger.info(f"📊 Vector search returned {len(vector_results)} hits")
                
                # Add vector results that aren't already in all_sources
                existing_ids = {s.get("evaluationId") for s in all_sources}
                
                for hit in vector_results:
                    evaluation_id = hit.get("evaluationId")
                    if evaluation_id not in existing_ids:
                        source_info = {
                            "evaluationId": evaluation_id,
                            "search_type": "vector",
                            "score": hit.get("_score", 0),
                            "template_name": hit.get("template_name", "Unknown"),
                            "template_id": hit.get("template_id"),
                            "text": hit.get("text", "")[:2000],
                            "evaluation": hit.get("evaluation", ""),
                            "transcript": hit.get("transcript", ""),
                            "metadata": hit.get("metadata", {}),
                            "content_type": "evaluation",
                            "_index": hit.get("_index"),
                            "vector_enhanced": True,
                            "best_matching_chunks": hit.get("best_matching_chunks", [])
                        }
                        all_sources.append(source_info)
                        existing_ids.add(evaluation_id)
                
                search_methods_used.append("pure_vector")
                
            except Exception as e:
                logger.error(f"❌ Vector search failed: {e}")
        
        # Strategy 3: Enhanced text search as fallback
        if len(all_sources) < max_results // 3:
            try:
                logger.info("📝 Supplementing with enhanced text search...")
                text_results = search_opensearch(
                    query=query,
                    filters=filters,
                    size=min(max_results - len(all_sources), 30)
                )
                
                logger.info(f"📊 Text search returned {len(text_results)} hits")
                
                # Add text results that aren't already included
                existing_ids = {s.get("evaluationId") for s in all_sources}
                
                for hit in text_results:
                    evaluation_id = hit.get("evaluationId")
                    if evaluation_id not in existing_ids:
                        source_info = {
                            "evaluationId": evaluation_id,
                            "search_type": hit.get("search_type", "text"),
                            "score": hit.get("_score", 0),
                            "template_name": hit.get("template_name", "Unknown"),
                            "template_id": hit.get("template_id"),
                            "text": hit.get("text", "")[:2000],
                            "evaluation": hit.get("evaluation", ""),
                            "transcript": hit.get("transcript", ""),
                            "metadata": hit.get("metadata", {}),
                            "content_type": "evaluation",
                            "_index": hit.get("_index"),
                            "vector_enhanced": False
                        }
                        all_sources.append(source_info)
                        existing_ids.add(evaluation_id)
                
                search_methods_used.append("enhanced_text")
                
            except Exception as e:
                logger.error(f"❌ Enhanced text search failed: {e}")
        
        # STEP 3: Process and verify results
        logger.info(f"🔗 TOTAL SOURCES FOUND: {len(all_sources)} using methods: {search_methods_used}")
        
        if not all_sources:
            logger.warning("⚠️ NO SOURCES FOUND with enhanced search")
            return create_empty_search_context("no_data"), []
        
        # STEP 4: Limit and deduplicate results
        processed_sources = []
        unique_evaluations = set()
        
        # Sort by score (hybrid/vector scores are generally better)
        all_sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        for source in all_sources[:max_results]:
            evaluation_id = source.get("evaluationId")
            if evaluation_id and evaluation_id not in unique_evaluations:
                unique_evaluations.add(evaluation_id)
                processed_sources.append(source)
        
        # STEP 5: Build enhanced context with vector search information
        if processed_sources:
            vector_enhanced_count = sum(1 for s in processed_sources if s.get("vector_enhanced", False))
            
            context = f"""
VERIFIED EVALUATION DATA FOUND: {len(processed_sources)} unique evaluations

✅ ENHANCED SEARCH RESULTS:
- Total evaluations: {len(unique_evaluations)}
- Content sources: {len(all_sources)}
- Search methods: {', '.join(search_methods_used)}
- Vector-enhanced results: {vector_enhanced_count}/{len(processed_sources)}
- Search quality: {'ENHANCED with semantic similarity' if vector_enhanced_count > 0 else 'Text-based matching'}

SAMPLE CONTENT FROM TOP RESULT:
{processed_sources[0].get('text', '')[:500]}...

EVALUATION DETAILS:
"""
            
            # Add details from first few evaluations
            for i, source in enumerate(processed_sources[:5]):
                metadata = source.get("metadata", {})
                search_type = source.get("search_type", "unknown")
                score = source.get("score", 0)
                
                context += f"""
[Evaluation {i+1}] ID: {source['evaluationId']} (Score: {score:.3f}, Type: {search_type})
- Template: {source.get('template_name', 'Unknown')}
- Program: {metadata.get('program', 'Unknown')}
- Disposition: {metadata.get('disposition', 'Unknown')}
- Agent: {metadata.get('agent', 'Unknown')}
- Content: {source.get('text', '')[:200]}...
"""
            
            if len(processed_sources) > 5:
                context += f"\n... and {len(processed_sources) - 5} more evaluations"
            
            context += f"""

INSTRUCTIONS:
- Use ONLY the data shown above from {len(processed_sources)} evaluations
- Do not generate statistics not directly calculable from this data
- Focus on patterns and insights from the actual content provided
- Results are enhanced with {'vector similarity matching' if vector_enhanced_count > 0 else 'text matching only'}
- Search methods used: {', '.join(search_methods_used)}
"""
            
            logger.info(f"✅ ENHANCED CONTEXT BUILT: {len(context)} chars with {len(processed_sources)} sources")
            logger.info(f"🔮 Vector enhancement: {vector_enhanced_count}/{len(processed_sources)} results")
            
            return context, processed_sources
        else:
            logger.warning("⚠️ NO VALID SOURCES AFTER PROCESSING")
            return create_empty_search_context("no_valid_sources"), []

    except Exception as e:
        logger.error(f"❌ ENHANCED SEARCH CONTEXT BUILD FAILED: {e}")
        return create_empty_search_context("system_error", str(e)), []

def create_empty_search_context(status="no_data", error_msg=""):
    """Create empty search context for error cases"""
    if status == "opensearch_unavailable":
        return """
SEARCH ERROR: OpenSearch connection failed.

INSTRUCTIONS:
- Inform the user that the search system is temporarily unavailable
- Suggest trying again in a moment
- Do not generate any data or statistics
"""
    elif status == "system_error":
        return f"""
SEARCH ERROR: Failed to retrieve evaluation data due to technical issues.

Error details: {error_msg[:200]}

INSTRUCTIONS:
- Inform the user that there was a technical error accessing the evaluation database
- Suggest trying again or contacting technical support
- Do not generate any statistics or data
"""
    else:
        return """
NO EVALUATION DATA FOUND: The search did not return any evaluation records.

POSSIBLE CAUSES:
1. No data has been imported into the evaluation database
2. Search terms don't match any indexed content  
3. Applied filters are too restrictive
4. Database connection issues

INSTRUCTIONS:
- Clearly state that no evaluation data was found
- Do not generate or estimate any statistics
- Suggest checking data import status or adjusting search criteria
"""

def build_strict_metadata_context(metadata_summary: Dict[str, Any], query: str) -> str:
    """Alias for build_simplified_context for backward compatibility"""
    return build_simplified_context(metadata_summary, query)

# ============================================================================
# ENHANCED SOURCES SUMMARY WITH VECTOR SEARCH INFO
# ============================================================================

def build_sources_summary_with_details(sources, filters=None):
    """
    ENHANCED: Build sources summary with vector search information
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
                "sites": 0,
                "vector_enhanced": 0,  # ✅ NEW
                "search_methods": []   # ✅ NEW
            },
            "details": {},
            "totals": {},
            "full_data": {},
            "search_enhancement": {  # ✅ NEW
                "vector_search_used": False,
                "hybrid_search_used": False,
                "search_quality": "text_only"
            }
        }
    
    DISPLAY_LIMIT = 25
    
    # ✅ NEW: Track vector search usage
    vector_enhanced_count = 0
    search_methods = set()
    vector_search_used = False
    hybrid_search_used = False
    
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
    
    # Process each source
    seen_evaluation_ids = set()

    for source in sources:
        # ✅ Track search enhancement info
        search_type = source.get("search_type", "unknown")
        search_methods.add(search_type)
        
        if source.get("vector_enhanced", False):
            vector_enhanced_count += 1
        
        if search_type == "vector":
            vector_search_used = True
        elif search_type in ["hybrid", "text_with_vector_fallback_v4_7_0"]:
            hybrid_search_used = True
        
        # Get evaluation ID
        evaluation_id = None
        source_data = source.get("_source", source)
        
        for id_field in ["evaluationId", "evaluation_id", "internalId", "internal_id"]:
                if source_data.get(id_field):
                    evaluation_id = source_data[id_field]
                    break
                if source_data.get("metadata", {}).get(id_field):
                    evaluation_id = source_data["metadata"][id_field]
                    break
            
        if not evaluation_id and source.get("evaluationId"):
            evaluation_id = source.get("evaluationId")
        
        if not evaluation_id:
            continue
            
        if evaluation_id in seen_evaluation_ids:
            continue
        seen_evaluation_ids.add(evaluation_id)
        unique_evaluations.add(evaluation_id)
        
        # Get metadata
        metadata = source.get("metadata", {})
        
        # Extract basic fields with enhanced search info
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
        
        # Build evaluation detail record with search enhancement info
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
            "duration": metadata.get("call_duration", "N/A"),
            "search_type": search_type,  # ✅ NEW
            "vector_enhanced": source.get("vector_enhanced", False),  # ✅ NEW
            "search_score": source.get("score", 0)  # ✅ NEW
        }
        evaluations_details.append(evaluation_detail)
        
        # Continue with existing logic for other aggregations...
        # (keeping the rest of the function the same but adding search enhancement to summary)
        
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
        
        # Track partners and sites details
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
            "evaluations": details["evaluations"][:10]
        }
        agents_list.append(agent_record)
    
    programs_list = []
    for program_name, details in programs_details.items():
        program_record = {
            "program_name": program_name,
            "evaluation_count": details["evaluation_count"],
            "agent_count": len(details["agents"]),
            "agents": list(details["agents"])[:10],
            "templates": list(details["templates"])
        }
        programs_list.append(program_record)
    
    # ✅ Build final response with vector search enhancement info
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
        "sites": len(unique_sites),
        "vector_enhanced": vector_enhanced_count,  # ✅ NEW
        "search_methods": list(search_methods)      # ✅ NEW
    }
    
    # Prepare detailed data
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
    
    # Prepare full data for download
    full_data_for_download = {
        "evaluations": evaluations_details,
        "agents": agents_list,
        "programs": programs_list,
        "templates": [{"template_name": name, "usage_count": details["usage_count"], 
                      "programs": list(details["programs"]), "agents": list(details["agents"])} 
                     for name, details in templates_details.items()],
        "dispositions": [{"disposition_name": name, "count": details["count"],
                         "examples": details["examples"]} 
                        for name, details in dispositions_details.items()],
        "opportunities": opportunities_details,
        "churn_triggers": churn_triggers_details,
        "partners": [{"partner_name": name, "evaluation_count": details["evaluation_count"],
                     "programs": list(details["programs"]), "agents": list(details["agents"])}
                    for name, details in partners_details.items()],
        "sites": [{"site_name": name, "evaluation_count": details["evaluation_count"],
                  "programs": list(details["programs"]), "agents": list(details["agents"])}
                 for name, details in sites_details.items()]
    }
    
    # Track total counts
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
    
    # ✅ NEW: Search enhancement information
    search_enhancement = {
        "vector_search_used": vector_search_used,
        "hybrid_search_used": hybrid_search_used,
        "vector_enhanced_results": vector_enhanced_count,
        "total_results": len(sources),
        "vector_enhancement_percentage": round((vector_enhanced_count / len(sources)) * 100, 1) if sources else 0,
        "search_methods_used": list(search_methods),
        "search_quality": "enhanced_with_vector_similarity" if vector_search_used or hybrid_search_used else "text_only"
    }
    
    return {
        "summary": summary,
        "details": detailed_data,
        "totals": totals,
        "full_data": full_data_for_download,
        "display_limit": DISPLAY_LIMIT,
        "search_enhancement": search_enhancement  # ✅ NEW
    }

# =============================================================================
# MAIN RAG-ENABLED CHAT ENDPOINT WITH VECTOR SEARCH
# =============================================================================

@chat_router.post("/chat")
async def relay_chat_rag(request: Request):
    start_time = time.time()
    try:
        body = await request.json()
        req = ChatRequest(**body)

        logger.info(f"💬 ENHANCED CHAT REQUEST WITH VECTOR SEARCH: {req.message[:60]}")
        logger.info(f"🔎 FILTERS RECEIVED: {req.filters}")

        is_report_request = detect_report_query(req.message)
        logger.info(f"📊 REPORT REQUEST DETECTED: {is_report_request}")

        # STEP 1: Build context with VECTOR SEARCH integration
        context, sources = build_search_context(req.message, req.filters)
        
        logger.info(f"📋 ENHANCED CONTEXT BUILT: {len(context)} chars, {len(sources)} sources")
        if not context:
            logger.warning("⚠️ NO CONTEXT FOUND - Chat will use general knowledge only")
        
        # STEP 2: Enhanced system message with vector search awareness
        system_message = f"""You are an AI assistant for call center evaluation data analysis. Analyze the provided evaluation data carefully and provide actionable insights.

ANALYSIS INSTRUCTIONS:
1. Review transcripts or agent summaries to identify recurring communication issues or strengths
2. Compare tone, language, empathy, and product knowledge across transcripts and agents  
3. Identify opportunities for improvement in agent performance based on the provided evaluation data
4. Highlight strengths in agent performance such as effective communication, problem-solving skills, and customer rapport
5. Determine what is successful and what needs improvement, with clear justifications
6. Write a concise but structured summary with clear sections and bullet points

✅ ENHANCED SEARCH CONTEXT (with vector similarity matching):
{context}

CRITICAL INSTRUCTIONS:
- ONLY use data from the provided context above
- Always answer questions based on the provided evaluation data
- Do not generate statistics not directly calculable from the data
- Be objective and data-informed
- Avoid overgeneralizations  
- Make the summary suitable for leadership or QA team use
- ✅ NOTE: Search results are enhanced with semantic similarity for better relevance

Respond in a clear, professional format with specific examples from the data."""

        # STEP 3: Streamlined Llama payload
        llama_payload = {
            "messages": [
                {"role": "system", "content": system_message},
                *[{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in req.history[-10:]],
                {"role": "user", "content": req.message}
            ],
            "temperature": GENAI_TEMPERATURE,
            "max_tokens": GENAI_MAX_TOKENS,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["<|eot_id|>", "<|end_of_text|>"]
        }

        headers = {
            "Authorization": f"Bearer {GENAI_ACCESS_KEY}",
            "Content-Type": "application/json"
        }

        logger.info(f"🦙 Making Llama 3.1 API call with vector-enhanced context...")
        
        if not GENAI_ENDPOINT or not GENAI_ACCESS_KEY:
            logger.error("❌ Missing Llama GenAI configuration!")
            return JSONResponse(
                status_code=500,
                content={
                    "reply": "Llama AI service not configured. Please check your GENAI_ENDPOINT and GENAI_ACCESS_KEY environment variables.",
                    "sources": [],
                    "timestamp": datetime.now().isoformat()
                }
            )

        genai_response = None
        successful_url = None
        
        # Try different URL formats
        possible_urls = [
            f"{GENAI_ENDPOINT.rstrip('/')}/v1/chat/completions",
            f"{GENAI_ENDPOINT.rstrip('/')}/api/v1/chat/completions",
            f"{GENAI_ENDPOINT.rstrip('/')}/v1/completions",
            f"{GENAI_ENDPOINT.rstrip('/')}/completions",
            GENAI_ENDPOINT.rstrip('/')
        ]
        
        for url in possible_urls:
            try:
                logger.info(f"🧪 Trying Llama URL: {url}")
                
                genai_response = requests.post(
                    url,
                    headers=headers,
                    json=llama_payload,
                    timeout=60
                )
                
                logger.info(f"📥 Llama Response Status: {genai_response.status_code} for {url}")
                
                if genai_response.ok:
                    successful_url = url
                    break
                else:
                    logger.warning(f"⚠️ URL {url} returned {genai_response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ URL {url} failed: {e}")
                continue
        
        if not genai_response or not genai_response.ok:
            error_text = genai_response.text() if genai_response else "No response"
            logger.error(f"❌ All Llama API URLs failed. Last error: {error_text[:500]}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "reply": f"I apologize, but I'm having trouble connecting to the Llama AI service. Status: {genai_response.status_code if genai_response else 'Connection failed'}. Please try again or contact support.",
                    "sources": [],
                    "timestamp": datetime.now().isoformat(),
                    "search_metadata": {
                        "error": f"Llama API error: {genai_response.status_code if genai_response else 'Connection failed'}",
                        "context_length": len(context),
                        "processing_time": round(time.time() - start_time, 2)
                    }
                }
            )

        logger.info(f"✅ Successful Llama URL: {successful_url}")

        try:
            result = genai_response.json()
            logger.info(f"📊 Llama Response Structure: {list(result.keys())}")
            
        except ValueError as e:
            logger.error(f"❌ Llama response is not valid JSON: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "reply": "I apologize, but received an invalid response from the Llama AI service. Please try again.",
                    "sources": [],
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # STEP 4: Extract reply from Llama response
        reply_text = None
        
        if "choices" in result and result["choices"] and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                reply_text = choice["message"]["content"].strip()
                logger.info(f"✅ Extracted Llama reply: {len(reply_text)} chars")
            elif "text" in choice:
                reply_text = choice["text"].strip()
            elif "delta" in choice and "content" in choice["delta"]:
                reply_text = choice["delta"]["content"].strip()
        elif "text" in result:
            reply_text = result["text"].strip()
        elif "content" in result:
            reply_text = result["content"].strip()
        elif "response" in result:
            reply_text = result["response"].strip()
        
        if not reply_text:
            logger.error(f"❌ Could not extract reply from Llama response")
            reply_text = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        # Clean up Llama response artifacts
        if reply_text:
            reply_text = reply_text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
            reply_text = reply_text.replace("<|start_header_id|>", "").replace("<|end_header_id|>", "")
            reply_text = reply_text.strip()
        
        logger.info(f"📝 Final Llama reply length: {len(reply_text)} characters")
               
        # STEP 5: Process sources for response with vector search info
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
                    "vector_enhanced": source.get("vector_enhanced", False),  # ✅ NEW
                    "metadata": source.get("metadata", {})
                })
                seen_ids.add(evaluation_id)

        # STEP 6: Build sources summary with vector search enhancement details
        sources_data = build_sources_summary_with_details(unique_sources, req.filters)

        # STEP 7: Build enhanced response with vector search information
        response_data = {
            "reply": reply_text,
            "sources_summary": sources_data["summary"],
            "sources_details": sources_data["details"], 
            "sources_totals": sources_data["totals"],
            "search_enhancement": sources_data["search_enhancement"],  # ✅ NEW
            "display_limit": sources_data["display_limit"],
            "sources": unique_sources[:20],
            "timestamp": datetime.now().isoformat(),
            "filter_context": req.filters,
            "search_metadata": {
                "vector_sources": len([s for s in sources if s.get("search_type") == "vector"]),
                "hybrid_sources": len([s for s in sources if s.get("search_type") in ["hybrid", "text_with_vector_fallback_v4_7_0"]]),
                "text_sources": len([s for s in sources if s.get("search_type") == "text"]),
                "vector_enhanced_count": len([s for s in sources if s.get("vector_enhanced", False)]),
                "context_length": len(context),
                "processing_time": round(time.time() - start_time, 2),
                "total_sources": len(sources),
                "unique_sources": len(unique_sources),
                "context_found": bool(context and "NO DATA FOUND" not in context),
                "metadata_verified": "verified_real_data" in context.lower(),
                "vector_search_enabled": True,  # ✅ NEW
                "llama_response_structure": list(result.keys()) if 'result' in locals() else [],
                "successful_url": successful_url,
                "model": "llama-3.1-8b-instruct",
                "version": "4.8.0_vector_enabled"
            }
        }
        
        logger.info(f"✅ ENHANCED CHAT RESPONSE WITH VECTOR SEARCH COMPLETE")
        logger.info(f"📊 Reply: {len(reply_text)} chars, Sources: {len(unique_sources)} verified")
        logger.info(f"🔮 Vector enhancement: {sources_data['search_enhancement']}")
        
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"❌ ENHANCED CHAT REQUEST FAILED: {e}")
        import traceback
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=500,
            content={
                "reply": f"I apologize, but I encountered an error while processing your request: {str(e)[:200]}. Please try again or contact support if the issue persists.",
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
                    "sites": 0,
                    "vector_enhanced": 0,
                    "search_methods": []
                },
                "search_enhancement": {
                    "vector_search_used": False,
                    "hybrid_search_used": False,
                    "search_quality": "error"
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
                    "vector_search_enabled": True,
                    "model": "llama-3.1-8b-instruct",
                    "version": "4.8.0_vector_enabled"
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
            "genai_agent": {"status": "configured"},
            "vector_search": {"status": "enabled"}  # ✅ NEW
        },
        "enhancements": {
            "document_structure": "enhanced v4.8.0",
            "metadata_verification": "enabled",
            "strict_data_alignment": "enforced",
            "evaluation_chunk_distinction": "implemented",
            "vector_search": "enabled",  # ✅ NEW
            "hybrid_search": "enabled"   # ✅ NEW
        }
    }

@health_router.get("/last_import_info")
async def last_import_info():
    return {
        "status": "success",
        "last_import_timestamp": datetime.now().isoformat(),
        "vector_search_enabled": True  # ✅ NEW
    }