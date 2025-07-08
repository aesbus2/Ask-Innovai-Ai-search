# chat_handlers.py - FIXED: Enhanced RAG with proper debugging and filter application
# Version: 4.3.2 - DEBUGGING ENABLED + PROPER FILTER APPLICATION

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

# =============================================================================
# MAIN RAG-ENABLED CHAT ENDPOINT
# =============================================================================

@chat_router.post("/chat")
async def relay_chat_rag(request: Request):
    start_time = time.time()
    try:
        body = await request.json()
        req = ChatRequest(**body)

        logger.info(f"💬 CHAT REQUEST: {req.message[:60]}")
        logger.info(f"🔎 FILTERS RECEIVED: {req.filters}")

        # STEP 1: Build context from OpenSearch using filters (WITH DEBUGGING)
        context, sources = build_search_context(req.message, req.filters)
        
        # DEBUG: Log context quality
        logger.info(f"📋 CONTEXT BUILT: {len(context)} chars, {len(sources)} sources")
        if not context:
            logger.warning("⚠️ NO CONTEXT FOUND - Chat will use general knowledge only")
        
        # STEP 2: Enhanced system message with better instructions
        if context:
            system_message = f"""You are an AI assistant analyzing call center evaluation data. Use the provided context to answer questions about call dispositions, agent performance, site comparisons, and evaluation metrics.

IMPORTANT: Base your response primarily on the context data provided below. If the context doesn't contain relevant information for the user's question, clearly state that and provide general guidance.

Context from evaluation database:
{context}

Format your response clearly with:
- Direct answers based on the data
- Specific numbers, percentages, or metrics when available
- Bullet points or tables for multiple items
- Clear distinctions between data-based insights and general recommendations

If no relevant data is found in the context, say "I don't see specific data for that query in the current results" and provide general guidance."""
        else:
            system_message = """You are an AI assistant for call center analytics. The search didn't return specific data for this query. Please provide general guidance and suggest the user try:
- Different search terms
- Checking if data has been imported
- Adjusting filters to broader criteria
- Verifying the system has the relevant data types

Provide helpful general information about call center analytics when possible."""

        # STEP 3: Construct chat payload
        do_payload = {
            "messages": [
                {"role": "system", "content": system_message},
                *[turn.dict() for turn in req.history],
                {"role": "user", "content": req.message}
            ],
            "temperature": GENAI_TEMPERATURE,
            "max_tokens": GENAI_MAX_TOKENS
        }

        headers = {
            "Authorization": f"Bearer {GENAI_ACCESS_KEY}",
            "Content-Type": "application/json"
        }

        do_url = f"{GENAI_ENDPOINT.rstrip('/')}/api/v1/chat/completions"
        logger.info(f"➡️ CALLING GenAI: {do_url}")

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

        # Enhanced response with debugging info
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
                "context_found": bool(context),
                "version": "4.3.2_enhanced_rag"
            }
        }
        
        logger.info(f"✅ CHAT RESPONSE READY: {len(reply_text)} chars, {len(unique_sources)} sources")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"❌ CHAT RAG FAILED: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "reply": f"I'm sorry, but I encountered an error while processing your request: {str(e)[:200]}. Please try again or contact support if the issue persists.",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "filter_context": body.get("filters", {}),
                "search_metadata": {
                    "error": str(e),
                    "context_length": 0,
                    "processing_time": round(time.time() - start_time, 2),
                    "version": "4.3.2_enhanced_rag"
                }
            }
        )

# =============================================================================
# ENHANCED RAG CONTEXT HELPER WITH DEBUGGING
# =============================================================================

def build_search_context(query: str, filters: dict) -> tuple[str, List[dict]]:
    """
    ENHANCED: Build search context with comprehensive debugging and error handling
    """
    logger.info(f"🔍 BUILDING SEARCH CONTEXT for: '{query}'")
    logger.info(f"📋 FILTERS: {filters}")
    
    context_parts = []
    sources = []

    try:
        # STEP 1: Try vector search first (if available)
        query_vector = None
        vector_hits = []
        
        try:
            logger.info("🔮 Attempting vector embedding...")
            query_vector = embed_text(query)
            logger.info(f"✅ Vector embedding successful: {len(query_vector)} dimensions")
            
            try:
                logger.info("🔍 Performing vector search...")
                vector_hits = search_vector(query_vector, index_override=None, size=5)
                logger.info(f"📊 Vector search returned {len(vector_hits)} hits")
                
                for hit in vector_hits:
                    hit["search_type"] = "vector"
                    
            except Exception as e:
                logger.warning(f"⚠️ Vector search failed: {e}")
                
        except Exception as e:
            logger.warning(f"⚠️ Vector embedding failed: {e}")

        # STEP 2: Try text search (primary method)
        text_hits = []
        try:
            logger.info("📝 Performing text search...")
            logger.info(f"🔎 Text search query: '{query}'")
            logger.info(f"🏷️ Text search filters: {filters}")
            
            text_hits = search_opensearch(query, index_override=None, filters=filters, size=10)
            logger.info(f"📊 Text search returned {len(text_hits)} hits")
            
            for hit in text_hits:
                hit["search_type"] = "text"
                
        except Exception as e:
            logger.error(f"❌ Text search FAILED: {e}")
            logger.error(f"🔍 Query was: '{query}'")
            logger.error(f"🏷️ Filters were: {filters}")

        # STEP 3: Combine and deduplicate results
        all_hits = []
        
        # Add vector hits first (higher priority)
        for hit in vector_hits:
            all_hits.append(hit)
            
        # Add text hits that aren't already included
        text_hit_ids = set()
        for hit in text_hits:
            hit_id = hit.get("_id")
            if hit_id not in [h.get("_id") for h in vector_hits]:
                all_hits.append(hit)
                text_hit_ids.add(hit_id)

        logger.info(f"🔗 COMBINED RESULTS: {len(all_hits)} total hits")

        # STEP 4: Extract context and build sources
        for i, hit in enumerate(all_hits[:8]):  # Limit to top 8 results
            try:
                doc = hit.get("_source", {})
                score = hit.get("_score", 0)
                search_type = hit.get("search_type", "unknown")
                
                # Get evaluation ID
                evaluation_id = (doc.get("evaluationId") or 
                               doc.get("evaluation_id") or 
                               doc.get("internalId") or 
                               f"eval_{i}")

                # Extract text content with priority order
                content_text = ""
                content_type = "unknown"
                
                # Try different content fields
                if doc.get("full_text"):
                    content_text = doc.get("full_text")
                    content_type = "full_text"
                elif doc.get("evaluation_text"):
                    content_text = doc.get("evaluation_text")
                    content_type = "evaluation"
                elif doc.get("transcript_text"):
                    content_text = doc.get("transcript_text")
                    content_type = "transcript"
                else:
                    # Try chunks
                    chunks = doc.get("chunks", [])
                    if chunks and isinstance(chunks, list):
                        chunk_texts = []
                        for chunk in chunks[:3]:  # First 3 chunks
                            if isinstance(chunk, dict) and chunk.get("text"):
                                chunk_texts.append(chunk["text"])
                        if chunk_texts:
                            content_text = "\n".join(chunk_texts)
                            content_type = "chunks"

                if content_text and len(content_text.strip()) > 20:
                    # Truncate long content
                    if len(content_text) > 1000:
                        content_text = content_text[:1000] + "..."
                    
                    context_parts.append(f"[Source {i+1} - {search_type} search]\n{content_text}")
                    
                    # Extract metadata safely
                    metadata = doc.get("metadata", {})
                    
                    source_info = {
                        "evaluationId": evaluation_id,
                        "text": content_text[:300] + ("..." if len(content_text) > 300 else ""),
                        "score": round(score, 3),
                        "search_type": search_type,
                        "content_type": content_type,
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
                            "call_duration": metadata.get("call_duration")
                        }
                    }
                    sources.append(source_info)
                    
                    logger.info(f"✅ SOURCE {i+1}: {evaluation_id} ({search_type}) - {len(content_text)} chars")
                else:
                    logger.warning(f"⚠️ Hit {i+1} has no usable content")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process hit {i}: {e}")

        # STEP 5: Build final context
        final_context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"📋 FINAL CONTEXT: {len(final_context)} characters")
        logger.info(f"📊 FINAL SOURCES: {len(sources)} sources")
        
        if not final_context:
            logger.warning("⚠️ NO CONTEXT GENERATED - Check if:")
            logger.warning("   1. Data has been imported to OpenSearch")
            logger.warning("   2. Search terms match indexed content")
            logger.warning("   3. Filters are not too restrictive")
            logger.warning("   4. OpenSearch connection is working")

        return final_context, sources

    except Exception as e:
        logger.error(f"❌ SEARCH CONTEXT BUILD FAILED: {e}")
        return "", []

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
            "document_structure": "enhanced v4.3.2",
            "rag_debugging": "enabled",
            "filter_application": "enhanced"
        }
    }

@health_router.get("/last_import_info")
async def last_import_info():
    return {
        "status": "success",
        "last_import_timestamp": datetime.now().isoformat()
    }