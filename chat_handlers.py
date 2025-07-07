# chat_handlers.py - PRODUCTION FastAPI Chat Router with Efficient OpenSearch 2.x Vector Support
# Version: 4.3.0 - RAG-enhanced with DO Agent Relay

import os
import logging
import requests
import time
from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from opensearch_client import search_opensearch, search_vector
from chat_handlers import determine_target_indices, build_search_context

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

class ChatRequest(BaseModel):
    message: str
    history: List[str] = []
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

        # Step 2: Convert history to OpenAI-style format
        history_messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
            for i, msg in enumerate(req.history)
        ]

        # Step 3: Inject RAG context
        system_message = f"""You are a helpful assistant. Use the following context to answer the user's question.\n\n{context}\n\nOnly answer based on the context above if relevant. Otherwise, use general knowledge."""

        # Step 4: Construct DO Agent payload
        do_payload = {
            "messages": [
                {"role": "system", "content": system_message},
                *history_messages,
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
                "version": "4.3.0_rag"
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
                    "version": "4.3.0_rag"
                }
            }
        )
