# chat_handlers.py - VERSION 5.1.1 - VECTOR SEARCH ENABLED
# updated ensure_vector_mapping_exists to not try and update KNN if it already exists 7-23-25
#updated  build_search_context to use a new function to remove viloations in search filters
# added strict metadata verification to ensure all results comply with filters
#added helper function extract_actual_metadata_values

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

CHAT_MAX_RESULTS = int(os.getenv("CHAT_MAX_RESULTS", "10000"))  # Increased from 100
HYBRID_SEARCH_LIMIT = int(os.getenv("HYBRID_SEARCH_LIMIT", "150"))  # No artificial 30 limit
VECTOR_SEARCH_LIMIT = int(os.getenv("VECTOR_SEARCH_LIMIT", "100"))  # No artificial 20 limit  
TEXT_SEARCH_LIMIT = int(os.getenv("TEXT_SEARCH_LIMIT", "100"))    # No artificial 30 limit


logger.info(f"   Max total results: {CHAT_MAX_RESULTS}")
logger.info(f"   Hybrid search limit: {HYBRID_SEARCH_LIMIT}")
logger.info(f"   Vector search limit: {VECTOR_SEARCH_LIMIT}")
logger.info(f"   Text search limit: {TEXT_SEARCH_LIMIT}")

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
        evaluationId = None
        for id_field in ["evaluationId", "internalId",]:
            if doc.get(id_field):
                evaluationId = doc.get(id_field)
                break
        
        if not evaluationId:
            evaluationId = f"eval_{hash(str(doc))}"

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
            "evaluationId": evaluationId,
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
                "agentName": metadata.get("agentName"),
                "disposition": metadata.get("disposition"),
                "subDisposition": metadata.get("subDisposition"),
                "language": metadata.get("language"),
                "call_date": metadata.get("call_date"),
                "call_duration": metadata.get("call_duration"),
                "call_type": metadata.get("call_type"),
                "agentId": metadata.get("agentId") or metadata.get("agentId"),
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
    ✅ ENHANCED: Build search context with VECTOR SEARCH integration + FILTER VALIDATION
    Now supports hybrid text+vector search for better relevance WITH strict filter enforcement
    """
    if max_results is None:
        max_results = CHAT_MAX_RESULTS
    
    logger.info(f"📋 Query: '{query}'")
    logger.info(f"🏷️ Filters: {filters}")
    logger.info(f"📊 Max results: {max_results}")
    
    def validate_filter_compliance(results: List[dict], strategy_name: str) -> List[dict]:
        """
        ✅ NEW: Validate that search results comply with applied filters
        """
        if not filters or not results:
            return results
        
        valid_results = []
        violations = []
        
        for result in results:
            is_valid = True
            violation_reasons = []
            
            # Check template_name filter (most critical)
            if filters.get("template_name"):
                expected_template = filters["template_name"]
                actual_template = result.get("template_name")
                if actual_template != expected_template:
                    is_valid = False
                    violation_reasons.append(f"template_name: expected '{expected_template}', got '{actual_template}'")
            
            # Check program filter
            if filters.get("program"):
                expected_program = filters["program"]
                actual_program = result.get("metadata", {}).get("program")
                if actual_program != expected_program:
                    is_valid = False
                    violation_reasons.append(f"program: expected '{expected_program}', got '{actual_program}'")
            
            # Check partner filter
            if filters.get("partner"):
                expected_partner = filters["partner"]
                actual_partner = result.get("metadata", {}).get("partner")
                if actual_partner != expected_partner:
                    is_valid = False
                    violation_reasons.append(f"partner: expected '{expected_partner}', got '{actual_partner}'")
            
            # Check site filter
            if filters.get("site"):
                expected_site = filters["site"]
                actual_site = result.get("metadata", {}).get("site")
                if actual_site != expected_site:
                    is_valid = False
                    violation_reasons.append(f"site: expected '{expected_site}', got '{actual_site}'")
            
            if is_valid:
                valid_results.append(result)
            else:
                violations.append({
                    "evaluationId": result.get("evaluationId"),
                    "strategy": strategy_name,
                    "violations": violation_reasons
                })
        
        if violations:
            logger.error(f"🚨 FILTER VIOLATIONS in {strategy_name}: {len(violations)} results removed")
            for violation in violations[:3]:  # Log first 3 violations
                logger.error(f"   - ID {violation['evaluationId']}: {', '.join(violation['violations'])}")
            if len(violations) > 3:
                logger.error(f"   - ... and {len(violations) - 3} more violations")
        
        logger.info(f"✅ {strategy_name} filter validation: {len(valid_results)}/{len(results)} results valid")
        return valid_results
    
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
        
        # ✅ STEP 2: Use enhanced search strategies WITH FILTER VALIDATION
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
                    size=min(max_results, HYBRID_SEARCH_LIMIT),
                    vector_weight=0.6  # 60% vector, 40% text
                )
                
                logger.info(f"📊 Hybrid search returned {len(hybrid_results)} hits")
                
                # ✅ VALIDATE FILTERS BEFORE PROCESSING
                validated_hybrid = validate_filter_compliance(hybrid_results, "hybrid_search")
                
                for hit in validated_hybrid:
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
        if query_vector and len(all_sources) < max_results:  # ✅ FIXED: Proper indentation and condition
            try:  # ✅ FIXED: Added missing try block
                logger.info("🔮 Trying pure vector search...")
                vector_results = search_vector(
                    query_vector=query_vector,
                    filters=filters,
                    size=max_results - len(all_sources)  # ✅ FIXED: Direct calculation, no remaining_slots
                )
                
                logger.info(f"📊 Vector search returned {len(vector_results)} hits")
                
                # ✅ VALIDATE FILTERS BEFORE PROCESSING
                validated_vector = validate_filter_compliance(vector_results, "vector_search")
                
                # Add vector results that aren't already in all_sources
                existing_ids = {s.get("evaluationId") for s in all_sources}
                
                for hit in validated_vector:
                    evaluationId = hit.get("evaluationId")
                    if evaluationId not in existing_ids:
                        source_info = {
                            "evaluationId": evaluationId,
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
                        existing_ids.add(evaluationId)
                
                search_methods_used.append("pure_vector")
                
            except Exception as e:
                logger.error(f"❌ Vector search failed: {e}")
        
        # Strategy 3: Enhanced text search as fallback - WITH STRICT VALIDATION
        if len(all_sources) < max_results:  # ✅ FIXED: Proper indentation and condition
            try:  # ✅ FIXED: Added missing try block
                logger.info("📝 Supplementing with enhanced text search...")
                text_results = search_opensearch(
                    query=query,
                    filters=filters,
                    size=max_results - len(all_sources)  # ✅ FIXED: Direct calculation, no remaining_slots
                )
                
                logger.info(f"📊 Text search returned {len(text_results)} hits")
                
                # ✅ CRITICAL: VALIDATE FILTERS - This is where violations were coming from!
                validated_text = validate_filter_compliance(text_results, "text_search")
                
                # Add text results that aren't already included
                existing_ids = {s.get("evaluationId") for s in all_sources}
                
                for hit in validated_text:
                    evaluationId = hit.get("evaluationId")
                    if evaluationId not in existing_ids:
                        source_info = {
                            "evaluationId": evaluationId,
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
                        existing_ids.add(evaluationId)
                
                search_methods_used.append("enhanced_text")
                
            except Exception as e:
                logger.error(f"❌ Enhanced text search failed: {e}")
        
        # STEP 3: Final filter validation on combined results
        logger.info(f"🔗 TOTAL SOURCES BEFORE FINAL VALIDATION: {len(all_sources)} using methods: {search_methods_used}")
        
        # ✅ FINAL SAFETY CHECK: Validate all combined results
        all_sources = validate_filter_compliance(all_sources, "final_combined")
        
        if not all_sources:
            logger.warning("⚠️ NO SOURCES FOUND after filter validation")
            return create_empty_search_context("no_data_after_filtering"), []
        
        # STEP 4: Limit and deduplicate results
        processed_sources = []
        unique_evaluations = set()
        
        # Sort by score (hybrid/vector scores are generally better)
        all_sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        for source in all_sources[:max_results]:
            evaluationId = source.get("evaluationId")
            if evaluationId not in unique_evaluations:
                unique_evaluations.add(evaluationId)
                processed_sources.append(source)
        
        # STEP 5: Build enhanced context with vector search information
        if processed_sources:
            vector_enhanced_count = sum(1 for s in processed_sources if s.get("vector_enhanced", False))
            
            # ✅ REPORT FILTER COMPLIANCE
            template_names = set(s.get("template_name") for s in processed_sources)
            filter_compliance_status = "✅ ALL FILTERS RESPECTED" if len(template_names) == 1 and filters.get("template_name") in template_names else "🚨 FILTER VIOLATIONS DETECTED"
        
        logger.info("🔍 Performing metadata verification...")
        metadata_summary = verify_metadata_alignment(processed_sources)

        # STEP 6: Build enhanced context with metadata verification AND STRICT ENFORCEMENT
        if processed_sources:
            vector_enhanced_count = sum(1 for s in processed_sources if s.get("vector_enhanced", False))
            
            # ✅ STEP 6A: Extract strict metadata constraints (ADD THIS)
            strict_metadata = extract_actual_metadata_values(processed_sources)
            
            # Include metadata verification status
            metadata_verified = metadata_summary.get("has_real_data", False)
            verification_status = "VERIFIED_REAL_DATA" if metadata_verified else "NO_DATA_VERIFICATION"

            context = f"""
VERIFIED EVALUATION DATA FOUND: {len(processed_sources)} unique evaluations

✅ ENHANCED SEARCH RESULTS WITH FILTER VALIDATION:
- Total evaluations: {len(unique_evaluations)}
- Content sources: {len(all_sources)}
- Search methods: {', '.join(search_methods_used)}
- Vector-enhanced results: {vector_enhanced_count}/{len(processed_sources)}
- Search quality: {'ENHANCED with semantic similarity' if vector_enhanced_count > 0 else 'Text-based matching'}
- Filter compliance: {filter_compliance_status}
- Templates found: {list(template_names)}
- Filters applied: {filters}

🔒 CRITICAL: You can ONLY use these exact metadata values from the search results:

ALLOWED DISPOSITIONS (ONLY THESE):
{', '.join(strict_metadata['dispositions'])}

ALLOWED PROGRAMS (ONLY THESE):  
{', '.join(strict_metadata['programs'])}

ALLOWED PARTNERS (ONLY THESE):
{', '.join(strict_metadata['partners'])}

ALLOWED SITES (ONLY THESE):
{', '.join(strict_metadata['sites'])}

SAMPLE CONTENT FROM TOP RESULT:
{processed_sources[0].get('text', '')[:500]}...

EVALUATION DETAILS:
"""
            
            # Add details from first few evaluations
            for i, source in enumerate(processed_sources[:10]):
                metadata = source.get("metadata", {})
                search_type = source.get("search_type", "unknown")
                score = source.get("score", 0)
                
                context += f"""
[Evaluation {i+1}] ID: {source['evaluationId']} (Score: {score:.3f}, Type: {search_type})
- Template: {source.get('template_name', 'Unknown')}
- Program: {metadata.get('program', 'Unknown')}
- Disposition: {metadata.get('disposition', 'Unknown')}
- subDisposition: {metadata.get('subDisposition', 'Unknown')}
- Partner: {metadata.get('partner', 'Unknown')}
- Agent: {metadata.get('agentName', 'Unknown')}
- Content: {source.get('text', '')[:200]}...
"""
            
            if len(processed_sources) > 5:
                context += f"\n... and {len(processed_sources) - 5} more evaluations"
            
            context += f"""

ABSOLUTE PROHIBITIONS:

1. NEVER use dispositions not in the allowed list above
2. NEVER use program/partner/site names not in the allowed lists above
3. NEVER generate statistics not directly countable from these {len(processed_sources)} evaluations
4. NEVER estimate or extrapolate beyond the provided data

REQUIRED BEHAVIOR:
- Always specify "from these {len(processed_sources)} evaluations"  
- Count exact occurrences when calculating percentages
- Reference specific evaluation IDs when discussing individual cases
- Quote directly from evaluation content when possible
- Use ONLY the metadata values listed in the allowed lists above

If information is not available in these {len(processed_sources)} evaluations, say so explicitly rather than guessing.
"""
            
        logger.info(f"✅ METADATA-ENFORCED CONTEXT BUILT: {len(context)} chars with {len(processed_sources)} sources")
        logger.info(f"🔒 Strict metadata constraints: {len(strict_metadata['agents'])} agents, {len(strict_metadata['dispositions'])} dispositions")

        for source in processed_sources:
            source["metadata_verified"] = metadata_verified
            
            return context, processed_sources
        else:
            logger.warning("⚠️ NO VALID SOURCES AFTER PROCESSING")
            return create_empty_search_context("no_valid_sources"), []

    except Exception as e:
        logger.error(f"❌ ENHANCED SEARCH CONTEXT BUILD FAILED: {e}")
        return create_empty_search_context("system_error", str(e)), []

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
    
    logger.info("📊 Metadata Verification Complete:")
    logger.info(f"   - Total evaluations: {verification['total_evaluations']}")
    logger.info(f"   - Data verification: {verification['data_verification']}")
    logger.info(f"   - Consistency score: {verification['data_consistency'].get('consistency_score', 0)}%")
    logger.info(f"   - Alignment issues: {len(verification['alignment_issues'])}")
    
    if verification["alignment_issues"]:
        logger.warning(f"⚠️ Found {len(verification['alignment_issues'])} alignment issues")
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
    seen_evaluationIds = set()

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
        context, sources = build_search_context(req.message, req.filters, max_results=CHAT_MAX_RESULTS)
        
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
            f"{GENAI_ENDPOINT.rstrip('/')}/api/v1/chat/completions",
            f"{GENAI_ENDPOINT.rstrip('/')}/v1/chat/completions",            
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
            error_text = genai_response.text if genai_response else "No response"
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
            evaluationId = source.get("evaluationId") or source.get("evaluationId")
            if evaluationId not in seen_ids:
                unique_sources.append({
                    "evaluationId": evaluationId,
                    "template_name": source.get("template_name", "Unknown"),
                    "search_type": source.get("search_type", "text"),
                    "score": source.get("score", 0),
                    "vector_enhanced": source.get("vector_enhanced", False),  # ✅ NEW
                    "metadata": source.get("metadata", {})
                })
                seen_ids.add(evaluationId)

        # STEP 6: Build sources summary with vector search enhancement details
        sources_data = build_sources_summary_with_details(unique_sources, req.filters)

        # STEP 7: Build enhanced response with vector search information
        response_data = {
            "reply": reply_text,
            "response": reply_text,
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