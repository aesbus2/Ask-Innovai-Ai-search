# chat_handlers.py - PRODUCTION FastAPI Chat Router with Efficient OpenSearch 2.x Vector Support
# Version: 4.3.1 - FIXED history schema to accept full chat message dicts

import os
import logging
import requests
import time
from datetime import datetime
from typing import Dict, Any, List, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from opensearch_client import search_opensearch, search_vector
from embedder import embed_text

logger = logging.getLogger(__name__)
chat_router = APIRouter()

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

        logger.info(f"💬 Chat received: {req.message[:60]}")
        logger.info(f"🔎 Filters: {list(req.filters.keys()) if req.filters else 'None'}")

        # Step 1: Build context from OpenSearch
        context, sources = build_search_context(req.message, req.filters)

        # Step 2: Inject RAG context
        system_message = f"""You are a helpful assistant. Use the following context to answer the user's question.\n\n{context}\n\nOnly answer based on the context above if relevant. Otherwise, use general knowledge."""

        # Step 3: Construct DO Agent payload
        do_payload = {
            "messages": [
                {"role": "system", "content": system_message},
                *[turn.dict() for turn in req.history],
                {"role": "user", "content": req.message}
            ],
            "stream": False,
            "include_functions_info": False,
            "include_retrieval_info": False,
            "include_guardrails_info": False
        }

        headers = {
            "Authorization": f"Bearer {GENAI_ACCESS_KEY}",
            "Content-Type": "application/json"
        }

        do_url = f"{GENAI_ENDPOINT.rstrip('/')}/chat/completions"
        logger.info(f"➡️ Forwarding to: {do_url}")

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
            reply_text = result["choices"][0]["message"]["content"]
        else:
            logger.error(f"❌ GenAI response missing 'choices': {result}")

        return JSONResponse(content={
            "reply": result["choices"][0]["message"]["content"],
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "filter_context": req.filters,
            "search_metadata": {
                "vector_sources": len([s for s in sources if s.get("search_type") == "vector"]),
                "text_sources": len([s for s in sources if s.get("search_type") == "text"]),
                "context_length": len(context),
                "processing_time": round(time.time() - start_time, 2),
                "version": "4.3.1_rag"
            }
        })

    except Exception as e:
        logger.error(f"❌ Chat relay RAG failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "reply": f"I'm sorry, but I couldn't process your request due to an error: {str(e)[:200]}",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "filter_context": body.get("filters", {}),
                "search_metadata": {
                    "error": str(e),
                    "context_length": 0,
                    "version": "4.3.1_rag"
                }
            }
        )

# =============================================================================
# RAG CONTEXT HELPER FUNCTION
# =============================================================================

def build_search_context(query: str, filters: dict) -> tuple[str, List[dict]]:
    context_parts = []
    sources = []

    try:
        query_vector = None
        try:
            query_vector = embed_text(query)
        except Exception as e:
            logger.warning(f"Vector embedding failed: {e}")

        vector_hits = []
        if query_vector:
            try:
                vector_hits = search_vector(query_vector, filters=filters, size=5)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        text_hits = []
        try:
            text_hits = search_opensearch(query, filters=filters, size=5)
        except Exception as e:
            logger.warning(f"Text search failed: {e}")

        all_hits = vector_hits + [h for h in text_hits if h not in vector_hits]

        for hit in all_hits[:5]:
            doc = hit.get("_source", {})
            score = hit.get("_score", 0)
            context_piece = doc.get("evaluation_text") or doc.get("transcript_text") or doc.get("full_text", "")
            if context_piece:
                context_parts.append(context_piece[:700])
                sources.append({
                    "text": context_piece[:300],
                    "template_id": doc.get("template_id"),
                    "template_name": doc.get("template_name"),
                    "program": doc.get("metadata", {}).get("program"),
                    "evaluation_id": doc.get("evaluationId"),
                    "score": score,
                    "search_type": "vector" if hit in vector_hits else "text",
                    "source_id": hit.get("_id"),
                    "collection": hit.get("_index")
                })

    except Exception as e:
        logger.error(f"Failed to build search context: {e}")

    final_context = "\n\n---\n\n".join(context_parts)
    return final_context, sources
