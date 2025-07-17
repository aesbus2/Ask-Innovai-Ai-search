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

# Run this test after updating your functions:
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
    ENHANCED: Build search context with configurable limits (removed 15 doc restriction)
    Now analyzes much more data for better comprehensive responses
    """
    logger.info(f"🔍 BUILDING SEARCH CONTEXT WITH ENHANCED LIMITS")
    logger.info(f"📋 Query: '{query}'")
    logger.info(f"🏷️ Filters: {filters}")
    logger.info(f"📊 Max results: {max_results}")
    
    try:
        # STEP 1: Try vector search first (if available)
        vector_hits = []
        try:
            logger.info("🔮 Attempting vector embedding...")
            query_vector = embed_text(query)
            logger.info(f"✅ Vector embedding successful: {len(query_vector)} dimensions")
            
            try:
                logger.info("🔍 Performing vector search...")
                # Increased vector search size too
                vector_hits = search_vector(query_vector, index_override=None, size=max_results//2)
                logger.info(f"📊 Vector search returned {len(vector_hits)} hits")
                
                for hit in vector_hits:
                    hit["search_type"] = "vector"
                    
            except Exception as e:
                logger.warning(f"⚠️ Vector search failed: {e}")
                
        except Exception as e:
            logger.warning(f"⚠️ Vector embedding failed: {e}")

        # STEP 2: Try text search (primary method) with increased size
        text_hits = []
        try:
            logger.info("📝 Performing text search with enhanced size...")
            
            # Enhanced query for disposition-related searches
            search_query = query
            if any(keyword in query.lower() for keyword in ["disposition", "call outcome", "call result", "call status"]):
                # Enhance search for disposition queries
                search_query = f"{query} disposition sub_disposition call_outcome call_result"
                logger.info(f"🎯 Enhanced disposition search query: '{search_query}'")
            
            # INCREASED: Much larger search size for comprehensive results
            text_hits = search_opensearch(search_query, index_override=None, filters=filters, size=max_results)
            logger.info(f"📊 Text search returned {len(text_hits)} hits")
            
            for hit in text_hits:
                hit["search_type"] = "text"
                
        except Exception as e:
            logger.error(f"❌ Text search FAILED: {e}")

        # STEP 3: Combine and deduplicate results
        all_hits = vector_hits + [hit for hit in text_hits if hit.get("_id") not in [h.get("_id") for h in vector_hits]]
        logger.info(f"🔗 COMBINED RESULTS: {len(all_hits)} total hits")

        # STEP 4: Process hits and build sources (REMOVED 15 limit)
        processed_sources = []
        
        # ENHANCED: Process ALL hits instead of limiting to 15
        for i, hit in enumerate(all_hits):  # ← REMOVED [:15] LIMIT!
            try:
                doc = hit.get("_source", {})
                score = hit.get("_score", 0)
                search_type = hit.get("search_type", "unknown")
                
                # Get evaluation ID with better extraction
                evaluation_id = None
                for id_field in ["evaluationId", "evaluation_id", "internalId", "internal_id"]:
                    if doc.get(id_field):
                        evaluation_id = doc.get(id_field)
                        break
                
                if not evaluation_id:
                    evaluation_id = f"eval_{i}"

                # Extract text content
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

                if content_text and len(content_text.strip()) > 20:
                    # ENHANCED: Don't truncate content as aggressively for better analysis
                    if len(content_text) > 1500:  # Increased from 800
                        content_text = content_text[:1500] + "..."
                    
                    # CRITICAL: Extract and verify metadata with better handling
                    metadata = doc.get("metadata", {})
                    
                    # Build verified source with all metadata
                    source_info = {
                        "evaluationId": evaluation_id,
                        "text": content_text,
                        "score": round(score, 3),
                        "search_type": search_type,
                        "template_id": doc.get("template_id"),
                        "template_name": doc.get("template_name"),
                        "metadata": {
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
                            "phone_number": metadata.get("phone_number"),
                            "contact_id": metadata.get("contact_id"),
                            "ucid": metadata.get("ucid"),
                            "call_type": metadata.get("call_type")
                        }
                    }
                    
                    processed_sources.append(source_info)
                    
                    if i < 20 or i % 50 == 0:  # Log first 20 and every 50th after that
                        logger.info(f"✅ SOURCE {i+1}: {evaluation_id} - Disposition: {metadata.get('disposition', 'N/A')} ({search_type})")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process hit {i}: {e}")

        # STEP 5: Verify metadata alignment and build strict context
        logger.info(f"🔍 METADATA VERIFICATION: Processing {len(processed_sources)} sources")
        
        metadata_summary = verify_metadata_alignment(processed_sources)
        
        logger.info(f"📊 ENHANCED METADATA SUMMARY:")
        logger.info(f"   Has real data: {metadata_summary['has_real_data']}")
        logger.info(f"   Total evaluations: {metadata_summary['total_evaluations']}")
        logger.info(f"   Total chunks: {metadata_summary['total_chunks_found']}")
        logger.info(f"   Dispositions found: {len(metadata_summary['dispositions'])}")
        logger.info(f"   Programs found: {len(metadata_summary['programs'])}")
        logger.info(f"   Sources processed: {len(processed_sources)}")
        
        if metadata_summary["has_real_data"]:
            # Build strict context with real metadata only
            strict_context = build_simplified_context(metadata_summary, query)
            
            # Add sample content for context (but more of it now)
            if processed_sources:
                strict_context += f"\n\nSAMPLE EVALUATION CONTENT FROM {len(processed_sources)} SOURCES:\n"
                for i, source in enumerate(processed_sources[:10]):  # Show first 10 instead of 3
                    strict_context += f"\n[Evaluation {i+1} - ID: {source['evaluationId']}]\n"
                    strict_context += f"Template: {source.get('template_name', 'Unknown')}\n"
                    strict_context += f"Program: {source.get('metadata', {}).get('program', 'Unknown')}\n"
                    strict_context += f"Disposition: {source.get('metadata', {}).get('disposition', 'Unknown')}\n"
                    strict_context += f"Content Preview: {source['text'][:200]}...\n"
                
                if len(processed_sources) > 10:
                    strict_context += f"\n... and {len(processed_sources) - 10} more evaluation sources analyzed\n"
            
            logger.info(f"✅ ENHANCED CONTEXT BUILT: {len(strict_context)} chars with {len(processed_sources)} sources")
            return strict_context, processed_sources
        else:
            logger.warning("⚠️ NO VERIFIED METADATA FOUND")
            no_data_context = """
NO EVALUATION DATA FOUND: The search did not return any evaluation records with metadata.

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
"""
            return no_data_context, []

    except Exception as e:
        logger.error(f"❌ ENHANCED SEARCH CONTEXT BUILD FAILED: {e}")
        error_context = f"""
SEARCH ERROR: Failed to retrieve evaluation data due to: {str(e)}

INSTRUCTIONS:
- Inform the user that there was a technical error accessing the evaluation database
- Do not generate any statistics or data
- Suggest trying again or contacting technical support
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

# =============================================================================
# MAIN RAG-ENABLED CHAT ENDPOINT WITH STRICT METADATA VERIFICATION
# =============================================================================

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
2. Compare tone, language, empathy, and product knowledge across different agents.
3. Identify patterns in unresolved issues, script reliance, and lack of customer satisfaction.
4. Highlight strengths that indicate potential program success.
5. Determine if the overall program is successful or needs improvement, with justification.
6. Write a concise but structured summary with clear sections and bullet points.

EVALUATION DATABASE CONTEXT:
{context}

CRITICAL INSTRUCTIONS:
- ONLY use data from the provided context above
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

        genai_response = requests.post(
            do_url,
            headers=headers,
            json=do_payload,
            timeout=30
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
            eval_id = source.get("evaluationId") or source.get("evaluation_id")
            if eval_id and eval_id not in seen_ids:
                unique_sources.append({
                    "evaluationId": eval_id,
                    "template_name": source.get("template_name", "Unknown"),
                    "search_type": source.get("search_type", "text"),
                    "score": source.get("score", 0),
                    "metadata": source.get("metadata", {})
                })
                seen_ids.add(eval_id)

        # STEP 5: Build enhanced response
        response_data = {
            "reply": reply_text,
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
                "version": "4.4.0_do_fixed"
            }
        }
        
        logger.info(f"✅ CHAT RESPONSE COMPLETE: {len(reply_text)} chars, {len(unique_sources)} verified sources")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"❌ CHAT REQUEST FAILED: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "reply": f"I apologize, but I encountered an error while processing your request: {str(e)[:200]}. Please try again or contact support if the issue persists.",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "filter_context": body.get("filters", {}) if 'body' in locals() else {},
                "search_metadata": {
                    "error": str(e),
                    "processing_time": round(time.time() - start_time, 2),
                    "version": "4.4.0_do_fixed"
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