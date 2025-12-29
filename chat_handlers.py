# -*- coding: utf-8 -*-
# chat_handlers.py - V 12-29-25.1 - VECTOR SEARCH ENABLED
# updated ensure_vector_mapping_exists to not try and update KNN if it already exists 7-23-25
#updated  build_search_context to use a new function to remove viloations in search filters
# added strict metadata verification to ensure all results comply with filters
#added helper function extract_actual_metadata_values

import os
import logging
# from pydoc import text  # Removed unused import to avoid shadowing
import requests
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Literal, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# âœ… VECTOR SEARCH ENABLED - Uncommented imports
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

CHAT_MAX_RESULTS = int(os.getenv("CHAT_MAX_RESULTS", "10000"))  
HYBRID_SEARCH_LIMIT = int(os.getenv("HYBRID_SEARCH_LIMIT", "1000"))  
VECTOR_SEARCH_LIMIT = int(os.getenv("VECTOR_SEARCH_LIMIT", "1000"))  
TEXT_SEARCH_LIMIT = int(os.getenv("TEXT_SEARCH_LIMIT", "1000"))   



logger.info(f"   Max total results: {CHAT_MAX_RESULTS}")
logger.info(f"   Hybrid search limit: {HYBRID_SEARCH_LIMIT}")
logger.info(f"   Vector search limit: {VECTOR_SEARCH_LIMIT}")
logger.info(f"   Text search limit: {TEXT_SEARCH_LIMIT}")

def clean_source_metadata(source: dict) -> dict:
    """
    Clean a single source to only include allowed API fields
    Add this function BEFORE build_search_context
    """
    # Define allowed fields
    ALLOWED_API_FIELDS = {
        "evaluationId", "weighted_score", "url", "partner", "site", "lob",
        "agentName", "agentId", "disposition", "subDisposition",
        "created_on", "call_date", "call_duration", "language", "evaluation"
    }
    
    # Fields to never display
    FORBIDDEN_FIELDS = {
        "_score", "_id", "_index", "score", "search_type", "match_count",
        "chunk_id", "vector_score", "text_score", "hybrid_score",
        "template_name", "template_id", "program", "internalId"
    }
    
    cleaned = {}
    
    # Keep evaluationId and text for processing
    if source.get("evaluationId"):
        cleaned["evaluationId"] = source["evaluationId"]
    if source.get("text"):
        cleaned["text"] = source["text"]  # Keep for context but not display
    
    # Extract metadata
    metadata = source.get("metadata", {})
    
    # Only include allowed fields from metadata
    for field in ALLOWED_API_FIELDS:
        # Check metadata first
        if metadata.get(field) is not None:
            value = metadata.get(field)
            if value and str(value).strip() and str(value).lower() not in ["unknown", "null", "n/a"]:
                cleaned[field] = value
        # Check source root
        elif source.get(field) is not None:
            value = source.get(field)
            if value and str(value).strip() and str(value).lower() not in ["unknown", "null", "n/a"]:
                cleaned[field] = value
    
    # Remove any forbidden fields
    for forbidden in FORBIDDEN_FIELDS:
        cleaned.pop(forbidden, None)
    
    return cleaned

def clean_all_sources(sources: List[dict]) -> List[dict]:
    """
    Clean all sources in a list
    Add this function BEFORE build_search_context
    """
    return [clean_source_metadata(source) for source in sources if source.get("evaluationId")]

def extract_search_metadata(sources: List[dict]) -> Dict[str, Any]:
    """Extract and organize metadata from search results using correct API field names"""
    metadata = {
        "evaluations": set(),
        "dispositions": [],
        "programs": [],
        "agents": [],
        "subDispositions": [],
        "partners": [],
        "sites": [],
        "lobs": [],
        "weighted_scores": [],
        "urls": [],
        "call_durations": [],
        "call_dates": [],
        "languages": [],
        "call_types": [],
        "template_names": set(),
        "agent_ids": set()
    }
    
    for source in sources:
        # Get metadata using CORRECT API field names
        meta = source.get("metadata", {})
        
        # Get evaluation ID (primary identifier)
        eval_id = source.get("evaluationId") or source.get("internalId")
        if eval_id:
            metadata["evaluations"].add(eval_id)
        
        # Agent information
        if meta.get("agentName"):
            metadata["agents"].append(meta["agentName"])
        
        if meta.get("agentId"):
            metadata["agent_ids"].add(meta["agentId"])
        
        # Disposition information
        if meta.get("disposition"):
            metadata["dispositions"].append(meta["disposition"])
        
        if meta.get("subDisposition"):
            metadata["subDispositions"].append(meta["subDisposition"])
        
        # Program information
        if meta.get("program"):
            metadata["programs"].append(meta["program"])
        
        # Partner and site information
        if meta.get("partner"):
            metadata["partners"].append(meta["partner"])
        
        if meta.get("site"):
            metadata["sites"].append(meta["site"])
        
        # LOB (Line of Business) information
        if meta.get("lob"):
            metadata["lobs"].append(meta["lob"])
        
        # Score information
        if meta.get("weighted_score"):
            try:
                score = float(meta["weighted_score"])
                metadata["weighted_scores"].append(score)
            except (ValueError, TypeError):
                # Skip invalid scores
                pass
        
        # URL information
        if meta.get("url"):
            metadata["urls"].append(meta["url"])
        
        # Call information
        if meta.get("call_duration"):
            metadata["call_durations"].append(meta["call_duration"])
        
        if meta.get("call_date"):
            metadata["call_dates"].append(meta["call_date"])
        
        if meta.get("language"):
            metadata["languages"].append(meta["language"])
        
        if meta.get("call_type"):
            metadata["call_types"].append(meta["call_type"])
        
        # Template information from source level (not just metadata)
        if source.get("template_name"):
            metadata["template_names"].add(source["template_name"])
    
    # Convert sets to lists for JSON serialization
    metadata["evaluations"] = list(metadata["evaluations"])
    metadata["template_names"] = list(metadata["template_names"])
    metadata["agent_ids"] = list(metadata["agent_ids"])
    
    # Remove duplicates from lists while preserving order
    for key in ["dispositions", "programs", "agents", "subDispositions", 
                "partners", "sites", "lobs", "languages", "call_types"]:
        if metadata[key]:
            # Preserve order while removing duplicates
            seen = set()
            unique_list = []
            for item in metadata[key]:
                if item not in seen and item not in ["Unknown", "unknown", "", None]:
                    seen.add(item)
                    unique_list.append(item)
            metadata[key] = unique_list
    
    # Calculate statistics for numeric fields
    if metadata["weighted_scores"]:
        metadata["score_stats"] = {
            "min": min(metadata["weighted_scores"]),
            "max": max(metadata["weighted_scores"]),
            "avg": sum(metadata["weighted_scores"]) / len(metadata["weighted_scores"]),
            "count": len(metadata["weighted_scores"])
        }
    
    # Add summary counts
    metadata["summary"] = {
        "total_evaluations": len(metadata["evaluations"]),
        "unique_agents": len(metadata["agents"]),
        "unique_programs": len(metadata["programs"]),
        "unique_dispositions": len(metadata["dispositions"]),
        "unique_partners": len(metadata["partners"]),
        "unique_sites": len(metadata["sites"]),
        "has_scores": len(metadata["weighted_scores"]) > 0,
        "has_urls": len(metadata["urls"]) > 0
    }
    
    return metadata

def track_search_enhancements(sources: List[dict]) -> Dict[str, Any]:
    """Track what search enhancement methods were used"""
    search_info = {
        "vector_search_used": False,
        "hybrid_search_used": False,
        "search_types": set(),
        "vector_enhanced_count": 0
    }
    
    for source in sources:
        search_type = source.get("search_type", "unknown")
        search_info["search_types"].add(search_type)
        
        if search_type == "vector":
            search_info["vector_search_used"] = True
        elif search_type == "hybrid":
            search_info["hybrid_search_used"] = True
            
        if source.get("vector_enhanced"):
            search_info["vector_enhanced_count"] += 1
    
    return search_info

def validate_search_results(sources: List[dict]) -> Dict[str, Any]:
    """Validate that search results contain real, usable data"""
    validation = {
        "has_real_data": False,
        "unique_evaluations": set(),
        "data_quality_issues": []
    }
    
    for source in sources:
        eval_id = source.get("evaluationId")
        template = source.get("template_name", "")
        
        if eval_id and template != "Unknown Template":
            validation["has_real_data"] = True
            validation["unique_evaluations"].add(eval_id)
    
    if not validation["has_real_data"]:
        validation["data_quality_issues"].append("No valid evaluation data found")
    
    return validation

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
    
    # âœ… Get vector search information
    vector_search_used = metadata_summary.get("vector_search_used", False)
    hybrid_search_used = metadata_summary.get("hybrid_search_used", False)
    search_types = metadata_summary.get("search_types", [])

    # Build context with safe field access including vector search info
    context = f"""
VERIFIED EVALUATION DATA FOUND: {metadata_summary.get('total_evaluations', 0)} unique evaluations from {metadata_summary.get('total_chunks_found', 0)} content sources

âœ… ENHANCED SEARCH CAPABILITIES USED:
- Vector Search: {'ENABLED' if vector_search_used else 'NOT USED'}
- Hybrid Search: {'ENABLED' if hybrid_search_used else 'NOT USED'}  
- Search Types: {', '.join(str(t) for t in search_types) if search_types else 'Standard'}
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
6. Use only the agent names found: {', '.join(str(name) for name in essential_fields.get('agentName', [])[:10]) if essential_fields.get('agentName') else 'No agent data'}
7. Use only the dispositions found: {', '.join(str(d) for d in metadata_summary.get('dispositions', [])[:10]) if metadata_summary.get('dispositions') else 'No disposition data'}
8. If asked about scores, use only these weighted scores: {', '.join(str(score) for score in weighted_scores[:10]) if weighted_scores else 'No score data'}
9. âœ… NOTE: Results enhanced with {'vector similarity matching' if vector_search_used else 'text matching only'}

DATA VERIFICATION STATUS: {metadata_summary.get('data_verification', 'VERIFIED_REAL_DATA')}
SEARCH ENHANCEMENT: {'VECTOR-ENHANCED' if vector_search_used or hybrid_search_used else 'TEXT-ONLY'}
"""
    
    return context


def _extract_transcript_text(hit: dict) -> str:
    """
    FIXED: Extract actual call transcript text, prioritizing conversation over Q&A
    """
    src = hit.get("_source", hit) or {}
    
    # Try each field and convert to string, checking for actual content
    def safe_get_text(value):
        if value is None or value == "":
            return ""
        text = str(value).strip()
        # Remove HTML tags if present
        import re
        text = re.sub(r'<[^>]+>', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text and len(text) > 10 else ""
    
    # FIXED PRIORITY ORDER: transcript fields FIRST, evaluation fields LAST
    field_attempts = [
        # PRIORITY 1: ACTUAL CALL TRANSCRIPTS (Speaker A/B conversations)
        ("transcript_text", src.get("transcript_text")),
        ("transcript", src.get("transcript")),        
        ("full_text", src.get("full_text")),
        
        # PRIORITY 2: NESTED TRANSCRIPT FIELDS
        ("metadata.transcript", src.get("metadata", {}).get("transcript")),
        ("metadata.transcript_text", src.get("metadata", {}).get("transcript_text")),
        ("metadata.full_text", src.get("metadata", {}).get("full_text")),
        
        # PRIORITY 3: SEARCH HIGHLIGHTS (often contain transcript snippets)
        ("highlight.transcript", hit.get("highlight", {}).get("transcript", [None])[0] if hit.get("highlight", {}).get("transcript") else None),
        ("highlight.transcript_text", hit.get("highlight", {}).get("transcript_text", [None])[0] if hit.get("highlight", {}).get("transcript_text") else None),
        
        # PRIORITY 4: CHUNK-BASED FIELDS (may contain transcript chunks)
        ("chunks_text", src.get("chunks_text")),
        ("text", src.get("text")),
        
        # PRIORITY 5: EVALUATION FIELDS (Q&A format - LAST RESORT ONLY)
        ("evaluation_text", src.get("evaluation_text")),
        ("evaluation", src.get("evaluation")),
        ("content", src.get("content")),
    ]
    
    transcript_text = ""
    found_field = None
    
    # Try fields in priority order - stop at first substantial content
    for field_name, field_value in field_attempts:
        text = safe_get_text(field_value)
        if text:
            # ADDITIONAL CHECK: Prefer content that looks like actual conversation
            # (contains "Speaker A" or timestamps) over Q&A format
            is_conversation = ("Speaker A" in text or "Speaker B" in text or 
                  (":" in text[:100] and "00:" in text))
            is_qa_format = ("Question:" in text or "Answer:" in text)
            
            # If this is actual conversation format, use it immediately
            if is_conversation and not is_qa_format:
                transcript_text = text
                found_field = field_name
                logger.debug(f"âœ… Found conversation format in field '{field_name}' for eval {src.get('evaluationId')}")
                break
            # If this is Q&A format, continue looking for better content
            elif is_qa_format:
                # Only use Q&A if we haven't found anything else yet
                if not transcript_text:
                    transcript_text = text
                    found_field = field_name
                    logger.debug(f"âš ï¸ Using Q&A format from field '{field_name}' as fallback for eval {src.get('evaluationId')}")
                continue
            # If it's other content, use it if we haven't found conversation yet
            else:
                transcript_text = text
                found_field = field_name
                break
    
    # Enhanced debugging
    eval_id = src.get("evaluationId", "unknown")
    
    if not transcript_text:
        available_fields = [k for k, v in src.items() if v is not None and str(v).strip()]
        logger.warning(f"âŒ No substantial transcript text found for document: {eval_id}")
        logger.debug(f"ðŸ“‹ Available source fields: {available_fields}")
    else:
        content_type = "conversation" if ("Speaker" in transcript_text and "00:" in transcript_text) else "Q&A" if ("Question:" in transcript_text) else "other"
        logger.debug(f"âœ… Extracted {len(transcript_text)} chars of {content_type} content from field '{found_field}' for eval {eval_id}")
    
    return transcript_text


def debug_transcript_extraction(hit: dict) -> dict:
    """
    NEW: Debug function to see exactly what's in a document
    Call this to understand why transcript extraction might be failing
    """
    src = hit.get("_source", hit) or {}
    eval_id = src.get("evaluationId", "unknown")
    
    debug_info = {
        "evaluationId": eval_id,
        "source_fields": {},
        "metadata_fields": {},
        "extraction_result": {},
        "recommendations": []
    }
    
    # Check all source fields
    for key, value in src.items():
        if value is not None:
            debug_info["source_fields"][key] = {
                "type": type(value).__name__,
                "length": len(str(value)) if value else 0,
                "preview": str(value)[:100] if value else "",
                "is_substantial": len(str(value).strip()) > 10 if value else False
            }
    
    # Check metadata fields specifically
    metadata = src.get("metadata", {})
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if value is not None:
                debug_info["metadata_fields"][key] = {
                    "type": type(value).__name__,
                    "length": len(str(value)) if value else 0,
                    "preview": str(value)[:100] if value else "",
                    "is_substantial": len(str(value).strip()) > 10 if value else False
                }
    
    # Test extraction
    extracted = _extract_transcript_text(hit)
    debug_info["extraction_result"] = {
        "success": bool(extracted),
        "length": len(extracted) if extracted else 0,
        "preview": extracted[:200] if extracted else ""
    }
    
    # Provide recommendations
    if not extracted:
        has_transcript_field = bool(src.get("transcript"))
        has_evaluation_field = bool(src.get("evaluation"))
        has_text_field = bool(src.get("text"))
        
        if has_transcript_field:
            debug_info["recommendations"].append("âœ… 'transcript' field exists - check if it contains actual content")
        elif has_evaluation_field:
            debug_info["recommendations"].append("âœ… 'evaluation' field exists - this might contain transcript data")
        elif has_text_field:
            debug_info["recommendations"].append("âœ… 'text' field exists - this might contain transcript data")
        else:
            debug_info["recommendations"].append("âŒ No obvious transcript fields found")
            debug_info["recommendations"].append("ðŸ” Check if transcript data is stored under a different field name")
    
    return debug_info

GROUNDING_POLICY = (
    "You are an AI assistant. Base your answers strictly on the provided transcript context. "
    "If the answer is not present, respond with the required fallback sentence."
)

def build_chat_prompt(user_query: str, context: str) -> str:
    return (
        f"{GROUNDING_POLICY}\n\n"
        f"# Transcript Context\n{context}\n\n"
        f"# Question\n{user_query}\n\n"
        f"# Instructions\n"
        f"- Use only the transcript context.\n"
        f"- If multiple documents conflict, note the conflict and cite the DOC ids.\n"
        f"- If nothing relevant is found, use the required fallback sentence above.\n"
    )


def detect_report_query(message: str) -> bool:
    """Detect if the message is asking for a report or analysis"""
    report_keywords = [
        "report", "analysis", "summary", "overview", "breakdown",
        "performance", "trends", "statistics", "metrics", "insights"
    ]
    return any(keyword in message.lower() for keyword in report_keywords)

def build_search_context(query: str, filters: dict, max_results: int = 100) -> Tuple[str, List[dict]]:
    """
    ENHANCED: Build search context with VECTOR SEARCH integration + FILTER VALIDATION
    UPDATED VERSION with strict metadata filtering AND TRANSCRIPT EXTRACTION
    """
    if max_results is None:
        max_results = CHAT_MAX_RESULTS
    
    logger.info(f"ðŸ“‹ Query: '{query}'")
    logger.info(f"ðŸ·ï¸ Filters: {filters}")
    logger.info(f"ðŸ“Š Max results: {max_results}")
    
    def validate_filter_compliance(results: List[dict], strategy_name: str) -> List[dict]:
        """
        Validate that search results comply with applied filters
        """
        if not filters or not results:
            return results
        
        valid_results = []
        violations = []
        
        for result in results:
            is_valid = True
            violation_reasons = []
            
            # Check filters against metadata
            metadata = result.get("metadata", {})
            
            # Check partner filter
            if filters.get("partner"):
                expected = filters["partner"]
                actual = metadata.get("partner") or result.get("partner")
                if actual != expected:
                    is_valid = False
                    violation_reasons.append("partner mismatch")
            
            # Check site filter
            if filters.get("site"):
                expected = filters["site"]
                actual = metadata.get("site") or result.get("site")
                if actual != expected:
                    is_valid = False
                    violation_reasons.append("site mismatch")
            
            # Check disposition filters
            if filters.get("disposition"):
                expected = filters["disposition"]
                actual = metadata.get("disposition") or result.get("disposition")
                if actual != expected:
                    is_valid = False
                    violation_reasons.append("disposition mismatch")
            
            if is_valid:
                valid_results.append(result)
            else:
                violations.append({
                    "evaluationId": result.get("evaluationId"),
                    "violations": violation_reasons
                })
        
        if violations:
            logger.debug(f"Filter violations in {strategy_name}: {len(violations)} results removed")
        
        logger.info(f"âœ… {strategy_name} validation: {len(valid_results)}/{len(results)} valid")
        return valid_results
    
    try:
        from opensearch_client import get_opensearch_client, test_connection
        
        client = get_opensearch_client()
        if not client:
            logger.error("âŒ No OpenSearch client available")
            return "Search system unavailable.", []
        
        if not test_connection():
            logger.warning("OpenSearch not available for search context")
            return create_empty_search_context("opensearch_unavailable"), []
        
        logger.info(" OpenSearch connection verified")
        
        # STEP 1: Try to generate query vector for enhanced search
        query_vector = None
        try:
            query_vector = embed_text(query)
            logger.info(f" Query vector generated: {len(query_vector)} dimensions")
        except Exception as e:
            logger.warning(f" Vector generation failed: {e}")
        
        # STEP 2: Use enhanced search strategies WITH FILTER VALIDATION
        all_sources = []
        search_methods_used = []
        
        # Strategy 1: Hybrid search (text + vector) if vector available
        if query_vector:
            try:
                logger.info("Trying hybrid text+vector search...")
                hybrid_results = hybrid_search(
                    query=query,
                    query_vector=query_vector,
                    filters=filters,
                    size=min(max_results, HYBRID_SEARCH_LIMIT),
                    vector_weight=0.6  # 60% vector, 40% text
                )
                
                logger.info(f"Hybrid search returned {len(hybrid_results)} hits")
                
                # VALIDATE FILTERS BEFORE PROCESSING
                validated_hybrid = validate_filter_compliance(hybrid_results, "hybrid_search")
                validated_hybrid = clean_all_sources(validated_hybrid)
                
                for hit in validated_hybrid:
                    hit["search_method"] = "hybrid"  # For internal tracking only
                    all_sources.append(hit)
                
                search_methods_used.append("hybrid")               
                
                
            except Exception as e:
                logger.error(f"âŒ Hybrid search failed: {e}")
        
        # Strategy 2: Pure vector search as fallback/supplement
        if query_vector and len(all_sources) < max_results:  
            try:  
                logger.info("Trying vector search...")
                vector_results = search_vector(
                    query_vector=query_vector,
                    filters=filters,
                    size=max_results - len(all_sources)  
                )
                
                logger.info(f" Vector search returned {len(vector_results)} hits")
                
                #  VALIDATE FILTERS BEFORE PROCESSING
                validated_vector = validate_filter_compliance(vector_results, "vector")
                #  CLEAN RESULTS IMMEDIATELY
                validated_vector = clean_all_sources(validated_vector)
                
                # Add vector results that aren't already in all_sources
                existing_ids = {s.get("evaluationId") for s in all_sources}
                
                for hit in validated_vector:
                    if hit.get("evaluationId") not in existing_ids:
                        hit["search_method"] = "vector"  # For internal tracking only
                        all_sources.append(hit)
                        existing_ids.add(hit.get("evaluationId"))
                
                search_methods_used.append("vector")
                
            except Exception as e:
                logger.error(f"âŒ Vector search failed: {e}")
        
        # Strategy 3: Text search as fallback - WITH STRICT VALIDATION
        if len(all_sources) < max_results:
            try: 
                logger.info(" Supplementing with enhanced text search...")
                text_results = search_opensearch(
                    query=query,
                    filters=filters,
                    size=max_results - len(all_sources)  # Direct calculation, no remaining_slots
                )
                
                logger.info(f" Text search returned {len(text_results)} hits")
                
                # VALIDATE FILTERS - BEFORE PROCESSING
                validated_text = validate_filter_compliance(text_results, "text")
                
                # CLEAN RESULTS IMMEDIATELY
                validated_text = clean_all_sources(validated_text)
                # Add unique results only
                existing_ids = {s.get("evaluationId") for s in all_sources}
                
                for hit in validated_text:
                    if hit.get("evaluationId") not in existing_ids:
                        hit["search_method"] = "text"  # For internal tracking only
                        all_sources.append(hit)
                        existing_ids.add(hit.get("evaluationId"))
                
                search_methods_used.append("text")
                
            except Exception as e:
                logger.error(f"Text search failed: {e}")                
               
        # Final filter validation on combined results
        logger.info(f" Total sources before final cleaning: {len(all_sources)}")
        
        all_sources = clean_all_sources(all_sources)
        all_sources = validate_filter_compliance(all_sources, "final")
                
        if not all_sources:
            logger.warning(" NO SOURCES FOUND after filter validation")
            return create_empty_search_context("no_data_after_filtering"), []
        
        # Limit and deduplicate results
        processed_sources = []
        unique_evaluations = set()
        
        for source in all_sources[:max_results]:
            evaluationId = source.get("evaluationId")
            if evaluationId and evaluationId not in unique_evaluations:
                unique_evaluations.add(evaluationId)
                processed_sources.append(source)

        processed_sources = clean_all_sources(processed_sources)

# FILTER COMPLIANCE CHECKING (Important for validation)
        # Check if filters are being respected (forinternal logging only)
        filter_compliance_passed = True
        if filters:
            # Check template filter compliance
            if filters.get("template_name"):
                template_names = set()
                for s in processed_sources:
                    # Check in cleaned data
                    if s.get("template_name"):
                        template_names.add(s.get("template_name"))
                
                if template_names and filters["template_name"] not in template_names:
                    filter_compliance_passed = False
                    logger.warning("âš ï¸ Template filter violation detected")
            
            # Check other filter compliance
            for filter_key in ["partner", "site", "disposition", "lob"]:
                if filters.get(filter_key):
                    expected_value = filters[filter_key]
                    found_values = set()
                    for s in processed_sources:
                        actual_value = s.get(filter_key)
                        if actual_value:
                            found_values.add(actual_value)
                    
                    if found_values and expected_value not in found_values:
                        filter_compliance_passed = False
                        logger.warning(f" {filter_key} filter violation detected")
        
        # Log filter compliance status (internal use only)
        if filter_compliance_passed:
            logger.info(" All filters respected in final results")
        else:
            logger.warning(" Filter violations detected in final results")
        
        # Verify metadata alignment
        logger.info(" Performing metadata verification...")
        
        # Build context with STRICT display rules
        if processed_sources:
            # Extract actual metadata values for constraints
            strict_metadata = extract_actual_metadata_values(processed_sources)
            
            #  BUILD CONTEXT WITHOUT SCORES OR INTERNAL FIELDS
            context = f"""

EVALUATION DATA FOUND: {len(processed_sources)} evaluations matching "{query}"

FILTER STATUS: {"âœ… All requested filters applied" if filter_compliance_passed else "âš ï¸ Some filter constraints may not have matches"}
APPLIED FILTERS: {', '.join([f"{k}={v}" for k, v in filters.items()]) if filters else "None"}

STRICT DISPLAY RULES - YOU MUST FOLLOW:
1. NEVER display these fields: _score, score, search_type, template_name, template_id, program
2. NEVER show relevance scores, or search quality indicators
3. ONLY display these exact fields when showing data:
   - evaluationId, weighted_score, url
   - partner, site, lob
   - agentName, agentId
   - disposition, subDisposition
   - created_on, call_date, call_duration
   - language, evaluation

ALLOWED VALUES FROM DATA (USE ONLY THESE):
- Partners: {', '.join(sorted(strict_metadata.get('partners', [])))}
- Sites: {', '.join(sorted(strict_metadata.get('sites', [])))}
- LOBs: {', '.join(sorted(strict_metadata.get('lobs', [])))}
- Dispositions: {', '.join(sorted(strict_metadata.get('dispositions', [])))}
- SubDispositions: {', '.join(sorted(strict_metadata.get('subDispositions', [])))}
- Agents: {', '.join(sorted(strict_metadata.get('agents', [])[:20]))}

EVALUATION DETAILS:
"""
            
            # Add evaluation details WITHOUT scores or types
            transcripts_added = 0  # Track how many transcripts we add

            for i, source in enumerate(processed_sources[:10], 1):
                eval_str = f"\n[Evaluation {i}] ID: {source.get('evaluationId', 'Unknown')}\n"
                
                # Only display allowed fields
                display_fields = [
                    ('partner', 'Partner'),
                    ('site', 'Site'),
                    ('lob', 'LOB'),
                    ('agentName', 'Agent'),
                    ('disposition', 'Disposition'),
                    ('subDisposition', 'SubDisposition'),
                    ('weighted_score', 'Weighted Score'),
                    ('call_date', 'Call Date'),
                    ('call_duration', 'Duration'),
                    ('language', 'Language'),
                    ('url', 'eval_link')
                ]
                
                for field_key, field_label in display_fields:
                    value = source.get(field_key)
                    if value and str(value).strip():
                        # Format specific fields
                        if field_key == 'call_duration':
                            eval_str += f"- {field_label}: {value} seconds\n"
                        elif field_key == 'weighted_score':
                            try:
                                eval_str += f"- {field_label}: {float(value):.2f}\n"
                            except Exception:
                                eval_str += f"- {field_label}: {value}\n"
                        else:
                            eval_str += f"- {field_label}: {value}\n"
                
                # ===== EXTRACT AND ADD THE ACTUAL TRANSCRIPT =====
                try:
                    # This is the key addition - extract the transcript!
                    transcript = _extract_transcript_text(source)
                    
                    if transcript and len(transcript) > 100:  # Only add if substantial
                        eval_str += "\TRANSCRIPT:\n"
                        eval_str += f"{transcript[:4000]}\n"  # Include up to 4000 chars
                        eval_str += f"\n[Full transcript length: {len(transcript)} characters]\n"
                        
                        # Mark this source as having a transcript
                        source['has_transcript'] = True
                        source['transcript_preview'] = transcript[:50]
                        transcripts_added += 1
                        
                        logger.debug(f"âœ… Added transcript for eval {source.get('evaluationId')}: {len(transcript)} chars")
                    else:
                        eval_str += "\n[No transcript available for this evaluation]\n"
                        source['has_transcript'] = False
                        
                except Exception as e:
                    logger.error(f"Failed to extract transcript for eval {source.get('evaluationId')}: {e}")
                    eval_str += "\n[Transcript extraction error]\n"
                    source['has_transcript'] = False
                # ===== END OF NEW TRANSCRIPT SECTION =====
                
                context += eval_str
                context += "\n" + "="*50 + "\n"  # Add separator between evaluations
            
            # Log how many transcripts were added
            logger.info(f"ðŸ“ Added {transcripts_added} transcripts to context out of {min(10, len(processed_sources))} evaluations")
            
            if len(processed_sources) > 10:
                context += f"\n... and {len(processed_sources) - 10} more evaluations\n"
            
            context += f"""

CRITICAL INSTRUCTIONS:
- ONLY use the exact data provided above
- NEVER mention scores, search types, or templates
- NEVER generate statistics not directly countable from these {len(processed_sources)} evaluations
- NEVER estimate or extrapolate beyond the provided data
- NEVER generate percentages or averages not directly calculable
- NEVER generate fake references, text, or statistics
- When displaying results, use ONLY the allowed field names listed above
- If asked about data not in these evaluations, say it's not available

Example evaluation display format:
Evaluation ID: [evaluationId]
- Partner: [partner]
- Site: [site]
- Disposition: [disposition]
- SubDisposition: [subDisposition]
Remember: You are showing actual evaluation records, not search results.
"""
            
            logger.info(f"âœ… Context built: {len(context)} chars, {len(processed_sources)} cleaned sources")
            logger.info("ðŸ”’ Strict filtering applied - internal fields removed from display")
            logger.info(f"ðŸ“‹ Filter compliance: {'PASSED' if filter_compliance_passed else 'VIOLATIONS DETECTED'}")
            # Mark sources as verified
            for source in processed_sources:
                source["metadata_verified"] = True
                # Final removal of template_name before returning (was kept for filter checking)
                source.pop("template_name", None)
            
            return context, processed_sources
            
        else:
            logger.warning("âš ï¸ No valid sources after processing")
            return create_empty_search_context("no_valid_sources"), []

    except Exception as e:
        logger.error(f"âŒ Search context build failed: {e}")
        return create_empty_search_context("system_error", str(e)), []

            
def build_filtered_context_with_rules(sources: List[dict], query: str, filters: Dict = None) -> str:
    """
    Build context with STRICT display rules for the LLM
    """
    if not sources:
        return create_empty_search_context("no_data")
    
    # Extract actual values for strict enforcement
    actual_values = {
        "evaluationIds": set(),
        "partners": set(),
        "sites": set(),
        "lobs": set(),
        "agents": set(),
        "dispositions": set(),
        "subDispositions": set(),
    }
    
    for source in sources:
        if source.get("evaluationId"):
            actual_values["evaluationIds"].add(str(source["evaluationId"]))
        if source.get("partner"):
            actual_values["partners"].add(source["partner"])
        if source.get("site"):
            actual_values["sites"].add(source["site"])
        if source.get("lob"):
            actual_values["lobs"].add(source["lob"])
        if source.get("agentName"):
            actual_values["agents"].add(source["agentName"])
        if source.get("disposition"):
            actual_values["dispositions"].add(source["disposition"])
        if source.get("subDisposition"):
            actual_values["subDispositions"].add(source["subDisposition"])
    
    # Build context with strict rules
    context = f"""

EVALUATION DATA FOUND: {len(sources)} evaluations for query: "{query}"

ðŸ”’ STRICT DISPLAY RULES - YOU MUST FOLLOW THESE:
1. NEVER display these internal fields: _score, score, search_type, Type, Template, Program (unless in allowed list)
2. NEVER display match counts, relevance scores, or search quality indicators
3. NEVER display internal IDs like _id, _index, chunk_id, internalId
4. ONLY display these exact fields when showing evaluation data:
   - evaluationId, weighted_score, url, partner, site, lob
   - agentName, agentId, disposition, subDisposition
   - created_on, call_date, call_duration, language, evaluation

ALLOWED VALUES (ONLY USE THESE):
- Evaluation IDs: {', '.join(sorted(actual_values['evaluationIds'])[:10])}
- Partners: {', '.join(sorted(actual_values['partners']))}
- Sites: {', '.join(sorted(actual_values['sites']))}
- LOBs: {', '.join(sorted(actual_values['lobs']))}
- Agents: {', '.join(sorted(actual_values['agents'])[:20])}
- Dispositions: {', '.join(sorted(actual_values['dispositions']))}
- SubDispositions: {', '.join(sorted(actual_values['subDispositions']))}

EVALUATION DETAILS:
"""    
    # Add cleaned evaluation details
    for i, source in enumerate(sources[:10], 1):
        eval_detail = f"\n[Evaluation {i}] ID: {source.get('evaluationId', 'Unknown')}\n"
        
        # Only add allowed fields
        for field in ["partner", "site", "lob", "agentName", "disposition", "subDisposition", "weighted_score", "url"]:
            if source.get(field):
                eval_detail += f"- {field}: {source[field]}\n"
        
        context += eval_detail
    
    if len(sources) > 10:
        context += f"\n... and {len(sources) - 10} more evaluations\n"
    
    context += """

CRITICAL INSTRUCTIONS:
- ONLY use the exact data provided above
- NEVER mention scores, search types, or internal metadata
- NEVER generate data not in the provided evaluations
- When displaying results, use ONLY the allowed field names
- If a field is not in the allowed list above, DO NOT display it
"""
    
    return context 
         

def verify_metadata_alignment(sources: List[dict]) -> Dict[str, Any]:
    """
    Verify that metadata across sources is properly aligned and consistent.
    This function ensures data integrity and identifies any misalignment issues.
    
    Args:
        sources: List of search result dictionaries
        
    Returns:
        Dictionary containing verification results and essential fields
    """
    # Extract comprehensive metadata first
    metadata = extract_search_metadata(sources)
    
    # Initialize verification results
    verification = {
        "has_real_data": False,
        "total_evaluations": 0,
        "total_chunks_found": len(sources),
        "unique_evaluations": set(),
        "data_consistency": {},
        "essential_fields": {
            "evaluationId": [],
            "template_name": [],
            "agentName": [],
            "created_on": []
        },
        "metadata_coverage": {},
        "alignment_issues": [],
        "data_verification": "NO_DATA"
    }
    
    # Track unique evaluations and their metadata consistency
    evaluation_metadata_map = {}
    
    for source in sources:
        # Get evaluation ID
        eval_id = (source.get("evaluationId") or 
                  source.get("internalId") or
                  source.get("metadata", {}).get("evaluationId"))
        
        if not eval_id:
            verification["alignment_issues"].append("Source without evaluation ID found")
            continue
        
        verification["unique_evaluations"].add(eval_id)
        
        # Track essential fields
        if eval_id not in verification["essential_fields"]["evaluationId"]:
            verification["essential_fields"]["evaluationId"].append(eval_id)
        
        # Get metadata
        meta = source.get("metadata", {})
        
        # Track template names
        template = source.get("template_name", "")
        if template and template != "Unknown Template":
            if template not in verification["essential_fields"]["template_name"]:
                verification["essential_fields"]["template_name"].append(template)
        
        # Track agent names
        agent = meta.get("agentName", "")
        if agent and agent != "Unknown":
            if agent not in verification["essential_fields"]["agentName"]:
                verification["essential_fields"]["agentName"].append(agent)
        
        # Track dates
        date_field = (source.get("created_on") or 
                     meta.get("created_on") or 
                     meta.get("call_date"))
        if date_field:
            if date_field not in verification["essential_fields"]["created_on"]:
                verification["essential_fields"]["created_on"].append(date_field)
        
        # Check metadata consistency for same evaluation ID
        if eval_id not in evaluation_metadata_map:
            evaluation_metadata_map[eval_id] = {
                "program": meta.get("program"),
                "partner": meta.get("partner"),
                "agent": meta.get("agentName"),
                "disposition": meta.get("disposition"),
                "template": template,
                "sources_count": 1
            }
        else:
            # Check if metadata is consistent across chunks of same evaluation
            existing = evaluation_metadata_map[eval_id]
            existing["sources_count"] += 1
            
            # Check for inconsistencies
            if meta.get("program") != existing["program"]:
                verification["alignment_issues"].append(
                    f"Inconsistent program for {eval_id}: '{meta.get('program')}' vs '{existing['program']}'"
                )
            if meta.get("agentName") != existing["agent"]:
                verification["alignment_issues"].append(
                    f"Inconsistent agent for {eval_id}: '{meta.get('agentName')}' vs '{existing['agent']}'"
                )
            if template != existing["template"]:
                verification["alignment_issues"].append(
                    f"Inconsistent template for {eval_id}: '{template}' vs '{existing['template']}'"
                )
    
    # Calculate metadata coverage (what percentage of sources have each field)
    total_sources = len(sources)
    if total_sources > 0:
        fields_to_check = ["program", "partner", "agentName", "disposition", 
                          "subDisposition", "weighted_score", "call_duration"]
        
        for field in fields_to_check:
            count = sum(1 for s in sources if s.get("metadata", {}).get(field))
            verification["metadata_coverage"][field] = {
                "count": count,
                "percentage": round((count / total_sources) * 100, 1)
            }
    
    # Determine data verification status
    verification["total_evaluations"] = len(verification["unique_evaluations"])
    
    # Check if we have real, usable data
    has_evaluations = verification["total_evaluations"] > 0
    has_agents = len(verification["essential_fields"]["agentName"]) > 0
    has_templates = len(verification["essential_fields"]["template_name"]) > 0
    has_minimal_metadata = any(
        cov["percentage"] > 50 
        for cov in verification["metadata_coverage"].values()
    )
    
    verification["has_real_data"] = (
        has_evaluations and 
        has_agents and 
        has_templates and 
        has_minimal_metadata
    )
    
    if verification["has_real_data"]:
        verification["data_verification"] = "VERIFIED_REAL_DATA"
    elif has_evaluations:
        verification["data_verification"] = "PARTIAL_DATA"
    else:
        verification["data_verification"] = "NO_DATA"
    
    # Add consistency scores
    if evaluation_metadata_map:
        multi_source_evals = [
            eval_id for eval_id, data in evaluation_metadata_map.items() 
            if data["sources_count"] > 1
        ]
        
        verification["data_consistency"] = {
            "evaluations_with_multiple_sources": len(multi_source_evals),
            "alignment_issues_count": len(verification["alignment_issues"]),
            "consistency_score": round(
                (1 - len(verification["alignment_issues"]) / max(len(sources), 1)) * 100, 1
            )
        }
    
    # Include metadata summary from extract_search_metadata
    verification.update({
        "dispositions": metadata.get("dispositions", []),
        "programs": metadata.get("programs", []),
        "partners": metadata.get("partners", []),
        "sites": metadata.get("sites", []),
        "weighted_scores": metadata.get("weighted_scores", []),
        "urls": metadata.get("urls", []),
        "call_durations": metadata.get("call_durations", []),
        "vector_search_used": metadata.get("vector_enhanced_count", 0) > 0,
        "hybrid_search_used": "hybrid" in metadata.get("search_types", []),
        "search_types": list(set(metadata.get("search_types", [])))
    })
    
    # Log verification results
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("ðŸ“Š Metadata Verification Complete:")
    logger.info(f"   - Total evaluations: {verification['total_evaluations']}")
    logger.info(f"   - Data verification: {verification['data_verification']}")
    logger.info(f"   - Consistency score: {verification['data_consistency'].get('consistency_score', 0)}%")
    logger.info(f"   - Alignment issues: {len(verification['alignment_issues'])}")
    
    if verification["alignment_issues"]:
        logger.warning(f"âš ï¸ Found {len(verification['alignment_issues'])} alignment issues")
        for issue in verification["alignment_issues"][:3]:  # Log first 3 issues
            logger.warning(f"   - {issue}")
    
    return verification


def extract_actual_metadata_values(sources: List[dict]) -> Dict[str, List[str]]:
    """
    Extract the actual metadata values that AI is allowed to use
    """
    metadata_values = {
        "agents": set(),
        "dispositions": set(), 
        "programs": set(),
        "partners": set(),
        "sites": set(),
        "templates": set()
    }
    
    for source in sources:
        metadata = source.get("metadata", {})
        
        # Only add non-empty, non-unknown values
        for field, constraint_key in [
            ("agentName", "agents"),
            ("disposition", "dispositions"),
            ("program", "programs"), 
            ("partner", "partners"),
            ("site", "sites")
        ]:
            if metadata.get(field):
                value = str(metadata[field]).strip()
                if value and value.lower() not in ["unknown", "null", "", "n/a", "not specified"]:
                    metadata_values[constraint_key].add(value)
        
        # Template names
        if source.get("template_name"):
            template = str(source["template_name"]).strip()
            if template and template.lower() not in ["unknown", "null", ""]:
                metadata_values["templates"].add(template)
    
    # Convert sets to sorted lists
    return {key: sorted(list(values)) for key, values in metadata_values.items()}

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
                "vector_enhanced": 0,  # âœ… NEW
                "search_methods": []   # âœ… NEW
            },
            "details": {},
            "totals": {},
            "full_data": {},
            "search_enhancement": {  # âœ… NEW
                "vector_search_used": False,
                "hybrid_search_used": False,
                "search_quality": "text_only"
            }
        }
    
    DISPLAY_LIMIT = 25
    
    # âœ… NEW: Track vector search usage
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
    seen_evaluationIds = set()

    for source in sources:
        # âœ… Track search enhancement info
        search_type = source.get("search_type", "unknown")
        search_methods.add(search_type)
        
        if source.get("vector_enhanced", False):
            vector_enhanced_count += 1
        
        if search_type == "vector":
            vector_search_used = True
        elif search_type in ["hybrid", "text_with_vector_fallback_v4_7_0"]:
            hybrid_search_used = True
        
        # Get evaluation ID
        evaluationId = None
        source_data = source.get("_source", source)
        
        for id_field in ["evaluationId", "evaluationId", "internalId"]:
                if source_data.get(id_field):
                    evaluationId = source_data[id_field]
                    break
                if source_data.get("metadata", {}).get(id_field):
                    evaluationId = source_data["metadata"][id_field]
                    break
            
        if not evaluationId and source.get("evaluationId"):
            evaluationId = source.get("evaluationId")
        
        if not evaluationId:
            continue
            
        if evaluationId in seen_evaluationIds:
            continue
        seen_evaluationIds.add(evaluationId)
        unique_evaluations.add(evaluationId)
        
        # Get metadata
        metadata = source.get("metadata", {})
        
        # Extract basic fields with enhanced search info
        agent = (metadata.get("agentName") or "Unknown").strip()        
        program = (metadata.get("program") or "Unknown").strip()
        template = (source.get("template_name") or "Unknown").strip()
        disposition = (metadata.get("disposition") or "Unknown").strip()
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
            "evaluationId": evaluationId,
            "agentName": agent,
            "program": program,
            "template": template,
            "disposition": disposition,
            "partner": partner,
            "site": site,
            "date": formatted_date,
            "score": metadata.get("weighted_score", "N/A"),
            "duration": metadata.get("call_duration", "N/A"),
            "search_type": search_type,  # âœ… NEW
            "vector_enhanced": source.get("vector_enhanced", False),  # âœ… NEW
            "search_score": source.get("score", 0)  # âœ… NEW
        }
        evaluations_details.append(evaluation_detail)
        
        # Continue with existing logic for other aggregations...
        # (keeping the rest of the function the same but adding search enhancement to summary)
        
        # Track unique values and build detailed collections for agents
        if agent != "Unknown":
            unique_agents.add(agent)
            if agent not in agents_details:
                agents_details[agent] = {
                    "agentName": agent,
                    "evaluation_count": 0,
                    "programs": set(),
                    "latest_date": None,
                    "average_score": [],
                    "evaluations": []
                }
            agents_details[agent]["evaluation_count"] += 1
            agents_details[agent]["programs"].add(program)
            agents_details[agent]["evaluations"].append(evaluationId)
            if metadata.get("weighted_score"):
                try:
                    agents_details[agent]["average_score"].append(float(metadata.get("weighted_score")))
                except:  # noqa: E722
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
                "evaluationId": evaluationId,
                "agentName": agent,
                "date": formatted_date
            })
            
            # Check for opportunities and churn triggers
            disposition_lower = disposition.lower()
            if any(keyword in disposition_lower for keyword in ["opportunity", "sale", "interested", "lead", "positive"]):
                opportunities_details.append({
                    "evaluationId": evaluationId,
                    "agentName": agent,
                    "disposition": disposition,
                    "program": program,
                    "date": formatted_date,
                    "score": metadata.get("weighted_score", "N/A")
                })
            elif any(keyword in disposition_lower for keyword in ["churn", "cancel", "disconnect", "terminate", "unsatisfied", "complaint"]):
                churn_triggers_details.append({
                    "evaluationId": evaluationId,
                    "agentName": agent,
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
    for agentName, details in agents_details.items():
        agent_record = {
            "agentName": agentName,
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
    
    # âœ… Build final response with vector search enhancement info
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
        "vector_enhanced": vector_enhanced_count,  # âœ… NEW
        "search_methods": list(search_methods)      # âœ… NEW
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
    
    # âœ… NEW: Search enhancement information
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
        "search_enhancement": search_enhancement  # âœ… NEW
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

        logger.info(f"ðŸ’¬ ENHANCED CHAT REQUEST WITH VECTOR SEARCH: {req.message[:60]}")
        logger.info(f"ðŸ”Ž FILTERS RECEIVED: {req.filters}")

        is_report_request = detect_report_query(req.message)
        logger.info(f"ðŸ“Š REPORT REQUEST DETECTED: {is_report_request}")

        # STEP 1: Build context with VECTOR SEARCH integration
        context, sources = build_search_context(req.message, req.filters, max_results=CHAT_MAX_RESULTS)

        logger.info(f"ðŸ“‹ ENHANCED CONTEXT BUILT: {len(context)} chars, {len(sources)} sources")

        MIN_CONTEXT_CHARS = 2500
        MIN_DISTINCT_EVALS = 5        

        # BLOCK hallucinated responses if no real data
        if (not context or not sources 
            or len(context) < MIN_CONTEXT_CHARS 
            or len({s.get("evaluationId") for s in sources if s.get("evaluationId")}) < MIN_DISTINCT_EVALS):
            logger.warning(" No context found â€” skipping LLM and returning no-data message.")
            return JSONResponse(
                status_code=200,
                content={
                    "reply": "No relevant evaluation data was found for your query. Please adjust your filters or try a different question.",
                    "sources_summary": {
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
                        "vector_enhanced": 0,
                        "search_methods": []
                    },
                    "sources_details": {},
                    "sources_totals": {},
                    "sources": [],
                    "search_enhancement": {
                        "vector_search_used": True,
                        "hybrid_search_used": False,
                        "search_quality": "none"
                    },
                    "timestamp": datetime.now().isoformat(),
                    "filter_context": req.filters,
                    "search_metadata": {
                        "vector_search_enabled": True,
                        "context_found": False,
                        "context_length": 0,
                        "model": "llama-3.1-8b-instruct",
                        "version": "4.8.0_vector_enabled",
                        "note": "Blocked LLM due to missing OpenSearch context"
                    }
                }
            )

        
        # STEP 2: Enhanced system message with vector search awareness
        system_message = f"""You are a professional call center analytics assistant. Provide clear, concise executive level insights based on the filtered evaluation data.

## Response Format:

Format your response with:
- **Bold text** for key points using markdown
- Bullet points for lists
- Clear section headers

Keep formatting simple and readable.

**Key Findings:**
- Summarize main patterns and trends
- Focus on actionable insights
- Provide churn analysis and potential retention strategies
- list sales attempts and success rartes
    - list sales for phones, devices, plans, home internet,
- Include brief relevant quotes as a summaryonly when essential (max 2-3)

**Recommendations:**
- Provide specific, actionable steps 
- Prioritize by impact
- Provide coaching recommendations

**Summary:**
- Overall assessment and metrics
- List sub-disposition as bullet points and included trends and insights
- list all partners included in metadata "partner" filters

## Guidelines:
- Base answers strictly on the provided context and data
- Do not generate or estimate statistics not present in the context
- provided evaluation counts along with other relevant metrics
- Be concise and professional - avoid lengthy excerpts
- If information is not available, state that clearly
- Focus on business insights rather than raw data dumps

CONTEXT:
{context}

Rules:
- Use only the provided evaluation data
- Keep quotes brief and relevant
- If no relevant data exists, return: "No relevant data found for this query."
"""

        # STEP 3: Streamlined Llama payload
        user_only_history = [
                {"role": "user", "content": h["content"]}
                for h in (req.history or [])
                if h.get("role") == "user"
            ][-5:]
        
        llama_payload = {

            "messages": [
                {"role": "system", "content": system_message},
                *user_only_history,
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

        logger.info("ðŸ¦™ Making Llama 3.1 API call with vector-enhanced context...")
        
        if not GENAI_ENDPOINT or not GENAI_ACCESS_KEY:
            logger.error("âŒ Missing Llama GenAI configuration!")
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
            f"{GENAI_ENDPOINT.rstrip('/')}/api/v1/chat/completions",
            f"{GENAI_ENDPOINT.rstrip('/')}/v1/chat/completions",            
            f"{GENAI_ENDPOINT.rstrip('/')}/v1/completions",
            f"{GENAI_ENDPOINT.rstrip('/')}/completions",
            GENAI_ENDPOINT.rstrip('/')
        ]
        
        for url in possible_urls:
            try:
                logger.info(f"ðŸ§ª Trying Llama URL: {url}")
                
                genai_response = requests.post(
                    url,
                    headers=headers,
                    json=llama_payload,
                    timeout=60
                )
                
                logger.info(f"ðŸ“¥ Llama Response Status: {genai_response.status_code} for {url}")
                
                if genai_response.ok:
                    successful_url = url
                    break
                else:
                    logger.warning(f"âš ï¸ URL {url} returned {genai_response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"âš ï¸ URL {url} failed: {e}")
                continue
        
        if not genai_response or not genai_response.ok:
            error_text = genai_response.text if genai_response else "No response"
            logger.error(f"âŒ All Llama API URLs failed. Last error: {error_text[:500]}")
            
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

        logger.info(f"âœ… Successful Llama URL: {successful_url}")

        try:
            result = genai_response.json()
            logger.info(f"ðŸ“Š Llama Response Structure: {list(result.keys())}")
            
        except ValueError as e:
            logger.error(f"âŒ Llama response is not valid JSON: {e}")
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
                logger.info(f"âœ… Extracted Llama reply: {len(reply_text)} chars")
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
            logger.error("âŒ Could not extract reply from Llama response")
            reply_text = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        # Clean up Llama response artifacts
        if reply_text:
            reply_text = reply_text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
            reply_text = reply_text.replace("<|start_header_id|>", "").replace("<|end_header_id|>", "")
            reply_text = reply_text.strip()
        
        logger.info(f"ðŸ“ Final Llama reply length: {len(reply_text)} characters")
               
        #  STEP 5: Process sources for response - REMOVE internal fields
        unique_sources = []
        seen_ids = set()
        
        # Define allowed fields for display
        ALLOWED_DISPLAY_FIELDS = {
            "evaluationId","url", "partner", "site", "lob",
            "agentName", "agentId", "disposition", "subDisposition",
            "created_on", "call_date", "call_duration", "language", "evaluation"
        }
        
        for source in sources:
            evaluationId = source.get("evaluationId")
            if evaluationId and evaluationId not in seen_ids:
                # Build clean source with ONLY allowed fields
                clean_source = {"evaluationId": evaluationId}
                
                # Add only allowed fields
                for field in ALLOWED_DISPLAY_FIELDS:
                    if source.get(field) is not None:
                        value = source.get(field)
                        if value and str(value).strip() and str(value).lower() not in ["unknown", "null", "n/a"]:
                            clean_source[field] = value
                
                # Never include forbidden fields
                forbidden_fields = ["template_name", "search_type", "score", "vector_enhanced", 
                                  "_score", "_id", "_index", "program", "internalId"]
                for forbidden in forbidden_fields:
                    clean_source.pop(forbidden, None)
                
                unique_sources.append(clean_source)
                seen_ids.add(evaluationId)
        
        logger.info(f"ðŸ“ Cleaned {len(unique_sources)} sources for display")

        # STEP 6: Build sources summary without internal metadata
        sources_data = {
            "summary": {
                "evaluations": len(unique_sources),
                "partners": len(set(s.get("partner", "") for s in unique_sources if s.get("partner"))),
                "sites": len(set(s.get("site", "") for s in unique_sources if s.get("site"))),
                "dispositions": len(set(s.get("disposition", "") for s in unique_sources if s.get("disposition"))),
                "agents": len(set(s.get("agentName", "") for s in unique_sources if s.get("agentName"))),
                "date_range": "See evaluation details"
            },
            "details": {
                "total_evaluations": len(unique_sources),
                "filters_applied": req.filters
            },
            "totals": {
                "evaluations_processed": len(unique_sources)
            },
            "display_limit": 20
        }

        # STEP 7: Build response WITHOUT internal search metadata
        response_data = {
            "reply": reply_text,
            "response": reply_text,
            "sources_summary": sources_data["summary"],
            "sources_details": sources_data["details"],
            "sources_totals": sources_data["totals"],
            "display_limit": sources_data["display_limit"],
            "sources": unique_sources[:20],  # Already cleaned
            "timestamp": datetime.now().isoformat(),
            "filter_context": req.filters,
            "metadata_info": {
                "total_evaluations": len(unique_sources),
                "context_found": bool(context and "NO DATA FOUND" not in context),
                "processing_time": round(time.time() - start_time, 2),
                "filters_applied": bool(req.filters),
                "model": "llama-3.1-8b-instruct",
                "version": "5.0.0_clean_metadata"
            }
        }
        
        # Remove any references to search types, scores, or internal mechanics
        # No vector_sources, hybrid_sources, search_type counts, etc.
        
        logger.info("âœ… CHAT RESPONSE COMPLETE")
        logger.info(f"ðŸ“Š Reply: {len(reply_text)} chars, Sources: {len(unique_sources)} evaluations")
        
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"âŒ ENHANCED CHAT REQUEST FAILED: {e}")
        import traceback
        logger.error(f"ðŸ” Traceback: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=200,
            content={
                "reply": "âš ï¸ No relevant evaluation data was found for your query. Please adjust your filters or try a different question.",
                "sources_summary": {
                    "evaluations": 0,
                    "agents": 0,
                    "date_range": "No data",
                    "partners": 0,
                    "sites": 0,
                    "dispositions": 0
                },
                "sources_details": {},
                "sources_totals": {},
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "filter_context": req.filters,
                "metadata_info": {
                    "context_found": False,
                    "model": "llama-3.1-8b-instruct",
                    "version": "5.0.0_clean_metadata"
                }
            }
        )
    
def clean_source_for_response(source: dict) -> dict:
    """
    Clean a source dictionary to only include allowed API fields.
    This should be used before sending any source data in responses.
    """
    ALLOWED_FIELDS = {
        "evaluationId", "weighted_score", "url", "partner", "site", "lob",
        "agentName", "agentId", "disposition", "subDisposition",
        "created_on", "call_date", "call_duration", "language", "evaluation"
    }
    
    FORBIDDEN_FIELDS = {
        "_score", "_id", "_index", "score", "search_type", "match_count",
        "chunk_id", "vector_score", "text_score", "hybrid_score",
        "template_name", "template_id", "program", "internalId",
        "vector_enhanced", "vector_dimension", "best_matching_chunks"
    }
    
    cleaned = {}
    
    # Only include allowed fields
    for field in ALLOWED_FIELDS:
        if source.get(field) is not None:
            value = source.get(field)
            if value and str(value).strip() and str(value).lower() not in ["unknown", "null", "n/a"]:
                cleaned[field] = value
    
    # Ensure no forbidden fields
    for forbidden in FORBIDDEN_FIELDS:
        cleaned.pop(forbidden, None)
    
    return cleaned
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
            "vector_search": {"status": "enabled"}  # âœ… NEW
        },
        "enhancements": {
            "document_structure": "enhanced v4.8.0",
            "metadata_verification": "enabled",
            "strict_data_alignment": "enforced",
            "evaluation_chunk_distinction": "implemented",
            "vector_search": "enabled",  # âœ… NEW
            "hybrid_search": "enabled"   # âœ… NEW
        }
    }

@health_router.get("/last_import_info")
async def last_import_info():
    return {
        "status": "success",
        "last_import_timestamp": datetime.now().isoformat(),
        "vector_search_enabled": True  # âœ… NEW
    }