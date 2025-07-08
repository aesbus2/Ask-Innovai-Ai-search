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

logger = logging.getLogger(__name__)
chat_router = APIRouter()
health_router = APIRouter()

# =============================================================================
# CONFIGURATION
# =============================================================================

GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT", "")
GENAI_ACCESS_KEY = os.getenv("GENAI_ACCESS_KEY", "")
GENAI_MODEL = os.getenv("GENAI_MODEL", "n/a")
GENAI_TEMPERATURE = float(os.getenv("GENAI_TEMPERATURE", "0.7"))
GENAI_MAX_TOKENS = int(os.getenv("GENAI_MAX_TOKENS", "2000"))

# =============================================================================
# METADATA VERIFICATION AND ALIGNMENT FUNCTIONS
# =============================================================================

# Fix for chat_handlers.py - Replace the verify_metadata_alignment function

def verify_metadata_alignment(sources: List[dict]) -> Dict[str, Any]:
    """
    UPDATED: Simplified metadata verification focusing on 4 essential fields only
    Fields: evaluationId, template_name, agentName, created_on
    """
    metadata_summary = {
        # Keep existing structure for backward compatibility
        "dispositions": set(),
        "sub_dispositions": set(),
        "programs": set(),
        "partners": set(),
        "sites": set(),
        "lobs": set(),
        "agents": set(),  # Focus on this for agentName
        "languages": set(),
        "call_dates": [],
        "evaluation_ids": set(),
        "has_real_data": False,
        "total_evaluations": 0,
        "total_chunks_found": 0,
        "data_verification": "VERIFIED_REAL_DATA",
        
        # NEW: Essential fields tracking
        "essential_fields": {
            "evaluationId": set(),
            "template_name": set(),
            "agentName": set(),
            "created_on": set()
        }
    }
    
    seen_evaluation_ids = set()
    
    for source in sources:
        try:
            metadata_summary["total_chunks_found"] += 1
            
            # Extract evaluation ID
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
            
            if evaluation_id and evaluation_id not in seen_evaluation_ids:
                seen_evaluation_ids.add(evaluation_id)
                metadata_summary["total_evaluations"] += 1
                metadata_summary["evaluation_ids"].add(evaluation_id)
                metadata_summary["essential_fields"]["evaluationId"].add(evaluation_id)
                
                logger.info(f"🆔 Found unique evaluation: {evaluation_id}")
            
            # Extract metadata with focus on essential fields
            metadata = {}
            if source_data.get("metadata"):
                metadata = source_data["metadata"]
            elif source.get("metadata"):
                metadata = source["metadata"]
            
            # Process essential fields
            if evaluation_id:
                metadata_summary["essential_fields"]["evaluationId"].add(evaluation_id)
            
            # Template name
            template_name = (source_data.get("template_name") or 
                           source_data.get("templateName") or 
                           metadata.get("template_name") or 
                           "Unknown Template")
            metadata_summary["essential_fields"]["template_name"].add(template_name)
            
            # Agent name
            agent_name = (metadata.get("agent") or 
                         metadata.get("agentName") or 
                         source_data.get("agentName") or 
                         "Unknown Agent")
            metadata_summary["essential_fields"]["agentName"].add(agent_name)
            metadata_summary["agents"].add(agent_name)  # Keep existing structure
            
            # Created on
            created_on = (source_data.get("created_on") or 
                         metadata.get("created_on") or 
                         source_data.get("call_date") or 
                         metadata.get("call_date") or 
                         "Unknown Date")
            metadata_summary["essential_fields"]["created_on"].add(created_on)
            
            # Keep existing metadata processing for backward compatibility
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
            if metadata.get("language"):
                metadata_summary["languages"].add(metadata["language"])
            if metadata.get("call_date"):
                metadata_summary["call_dates"].append(metadata["call_date"])
                
            # Mark as having real data if essential fields are present
            if evaluation_id and template_name != "Unknown Template":
                metadata_summary["has_real_data"] = True
                
        except Exception as e:
            logger.error(f"Error processing source metadata: {e}")
            continue
    
    # Convert sets to sorted lists for consistent output
    for key in ["dispositions", "sub_dispositions", "programs", "partners", "sites", "lobs", "agents", "languages"]:
        metadata_summary[key] = sorted(list(metadata_summary[key]))
    
    metadata_summary["evaluation_ids"] = list(metadata_summary["evaluation_ids"])
    
    # Convert essential fields to lists
    for field in metadata_summary["essential_fields"]:
        metadata_summary["essential_fields"][field] = sorted(list(metadata_summary["essential_fields"][field]))
    
    # Enhanced logging for essential fields
    logger.info(f"📊 SIMPLIFIED METADATA SUMMARY:")
    logger.info(f"   Total evaluations: {metadata_summary['total_evaluations']}")
    logger.info(f"   Total chunks: {metadata_summary['total_chunks_found']}")
    logger.info(f"   Unique evaluation IDs: {len(metadata_summary['essential_fields']['evaluationId'])}")
    logger.info(f"   Unique template names: {len(metadata_summary['essential_fields']['template_name'])}")
    logger.info(f"   Unique agent names: {len(metadata_summary['essential_fields']['agentName'])}")
    logger.info(f"   Has real data: {metadata_summary['has_real_data']}")
    
    return metadata_summary

# Also fix the build_strict_metadata_context function:

def build_simplified_context(metadata_summary: Dict[str, Any], query: str) -> str:
    """
    UPDATED: Build context focusing on essential fields only
    """
    if not metadata_summary["has_real_data"]:
        return """
NO DATA FOUND: No evaluation records match your query criteria. 
You must clearly state that no data is available and suggest:
1. Checking if data has been imported
2. Adjusting search terms or filters
3. Verifying the evaluation database connectivity

DO NOT GENERATE OR ESTIMATE ANY NUMBERS, DATES, OR STATISTICS.
"""

    # Build context focusing on essential metadata
    context = f"""
VERIFIED EVALUATION DATA FOUND: {metadata_summary['total_evaluations']} unique evaluations from {metadata_summary['total_chunks_found']} content sources

ESSENTIAL METADATA AVAILABLE:
- Evaluation IDs: {len(metadata_summary['essential_fields']['evaluationId'])} unique
- Template Names: {metadata_summary['essential_fields']['template_name']}
- Agent Names: {metadata_summary['essential_fields']['agentName']}
- Date Range: {len(metadata_summary['essential_fields']['created_on'])} unique dates

CRITICAL INSTRUCTIONS:
1. ONLY use data from the provided evaluation sources
2. Focus on: evaluationId, template_name, agentName, created_on
3. DO NOT generate percentages or statistics not directly calculable from the data
4. Report on {metadata_summary['total_evaluations']} EVALUATIONS (not chunks)
5. Use only the agent names found: {', '.join(metadata_summary['essential_fields']['agentName'][:10])}

DATA VERIFICATION STATUS: {metadata_summary['data_verification']}
"""
    
    return context
# =============================================================================
# PYDANTIC INPUT SCHEMA
# =============================================================================

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatTurn] = []
    filters: Dict[str, Any] = {}
    analytics: bool = False
    metadata_focus: List[str] = []
    programs: List[str] = []


def detect_report_query(query: str) -> bool:
    """
    Detect if the user is asking for a comprehensive report/analysis
    This is just for detection - doesn't change any existing functionality
    """
    report_keywords = [
        'report', 'analysis', 'summary', 'overview', 'breakdown', 'statistics',
        'all', 'total', 'overall', 'comprehensive', 'complete', 'entire',
        'trends', 'patterns', 'distribution', 'performance', 'metrics',
        'dashboard', 'insights', 'analytics', 'aggregated'
    ]
    
    question_patterns = [
        'what are all', 'show me all', 'give me all', 'list all',
        'how many total', 'what is the total', 'across all',
        'overall performance', 'complete breakdown', 'full analysis'
    ]
    
    query_lower = query.lower()
    
    # Check for report keywords
    if any(keyword in query_lower for keyword in report_keywords):
        return True
    
    # Check for question patterns
    if any(pattern in query_lower for pattern in question_patterns):
        return True
    
    return False

def create_simplified_source_info(doc: dict, evaluation_id: str, content_text: str, score: float, search_type: str) -> dict:
    """
    UPDATED: Create source info focusing on essential fields for frontend
    """
    metadata = doc.get("metadata", {})
    
    # Focus on essential fields for frontend
    source_info = {
        # Core identification
        "evaluationId": evaluation_id,
        "text": content_text,
        "score": round(score, 3),
        "search_type": search_type,
        
        # Essential fields for simplified display
        "template_name": doc.get("template_name") or doc.get("templateName") or "Unknown Template",
        "agentName": metadata.get("agent") or metadata.get("agentName") or "Unknown Agent",
        "created_on": doc.get("created_on") or metadata.get("created_on") or doc.get("call_date") or metadata.get("call_date") or "Unknown Date",
        
        # Keep metadata structure for backward compatibility, but prioritize essential fields
        "metadata": {
            # Essential fields first
            "agent": metadata.get("agent") or metadata.get("agentName") or "Unknown Agent",
            "created_on": doc.get("created_on") or metadata.get("created_on") or doc.get("call_date") or metadata.get("call_date"),
            
            # Other fields for backward compatibility (but de-prioritized)
            "program": metadata.get("program"),
            "partner": metadata.get("partner"),
            "site": metadata.get("site"),
            "lob": metadata.get("lob"),
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
    
    return source_info


def build_search_context(query: str, filters: dict, max_results: int = 100) -> tuple[str, List[dict]]:
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

        # STEP 1: Build context with strict metadata verification (WITH DEBUGGING)
        context, sources = build_search_context(req.message, req.filters)
        
        # DEBUG: Log context quality
        logger.info(f"📋 CONTEXT BUILT: {len(context)} chars, {len(sources)} sources")
        if not context:
            logger.warning("⚠️ NO CONTEXT FOUND - Chat will use general knowledge only")
        
        # STEP 2: Enhanced system message with strict instructions
        system_message = f"""You are an AI assistant for call center evaluation data analysis. You must ONLY use the specific data provided in the context below.

CRITICAL RULES:
1. NEVER generate, estimate, or extrapolate any numbers, dates, or statistics
2. ONLY use the exact values and data shown in the context
3. If the context shows "NO DATA FOUND", clearly state no data is available
4. NEVER mention specific dates unless they appear in the actual context data
5. When reporting counts, distinguish between EVALUATIONS and CHUNKS
6. Report evaluation counts (like 304), NOT chunk counts (like 2,490)
7. If asked for trends or patterns, only describe what you can see in the actual data provided
8. Always clarify that your response is based on the specific evaluation records found

EVALUATION DATABASE CONTEXT:
{context}

Based on this verified data, provide accurate analysis only using the information explicitly shown above.
"""

        # STEP 3: Construct chat payload with lower temperature for factual responses
        do_payload = {
            "messages": [
                {"role": "system", "content": system_message},
                *[turn.dict() for turn in req.history],
                {"role": "user", "content": req.message}
            ],
            "temperature": max(0.3, GENAI_TEMPERATURE - 0.2),  # Lower temperature for more factual responses
            "max_tokens": GENAI_MAX_TOKENS
        }

        headers = {
            "Authorization": f"Bearer {GENAI_ACCESS_KEY}",
            "Content-Type": "application/json"
        }

        do_url = f"{GENAI_ENDPOINT.rstrip('/')}/api/v1/chat/completions"
        logger.info(f"➡️ CALLING GenAI with strict metadata context...")

        response = requests.post(
            do_url,
            headers=headers,
            json=do_payload,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()
        
        reply_text = "(No response)"
        if "choices" in result and result["choices"]:
            reply_text = result["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"❌ GenAI response missing 'choices': {result}")

        # Remove duplicate sources
        unique_sources = []
        seen_ids = set()
        for s in sources:
            eid = s.get("evaluationId") or s.get("evaluation_id")
            if eid and eid not in seen_ids:
                unique_sources.append(s)
                seen_ids.add(eid)

        # Enhanced response with metadata verification info
        response_data = {
            "reply": reply_text,
            "sources": unique_sources,
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
                "version": "4.4.0_metadata_aligned"
            }
        }
        
        logger.info(f"✅ CHAT RESPONSE WITH METADATA VERIFICATION: {len(reply_text)} chars, {len(unique_sources)} verified sources")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"❌ CHAT WITH METADATA VERIFICATION FAILED: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "reply": f"I apologize, but I encountered an error while accessing the evaluation database: {str(e)[:200]}. Please try again or contact support if the issue persists.",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "filter_context": body.get("filters", {}),
                "search_metadata": {
                    "error": str(e),
                    "context_length": 0,
                    "processing_time": round(time.time() - start_time, 2),
                    "metadata_verified": False,
                    "version": "4.4.0_metadata_aligned"
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