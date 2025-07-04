# chat_handlers.py - Production Digital Ocean AI Agent Integration
# Version: 1.1.0 - Production Ready with Clean Logging

import os
import logging
import asyncio
import aiohttp
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Setup production logging
logger = logging.getLogger("ask-innovai.chat")

# Environment configuration for Digital Ocean AI Agent (Llama 3.1 Instruct 8B)
GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT", "")
GENAI_ACCESS_KEY = os.getenv("GENAI_ACCESS_KEY", "")
GENAI_MODEL = os.getenv("GENAI_MODEL", "llama-3.1-8b-instruct")
GENAI_TIMEOUT = int(os.getenv("GENAI_TIMEOUT", "120"))
GENAI_MAX_RETRIES = int(os.getenv("GENAI_MAX_RETRIES", "3"))

# Llama 3.1 specific configuration
LLAMA_MAX_TOKENS = 4096
LLAMA_TEMPERATURE = 0.7
LLAMA_TOP_P = 0.9
LLAMA_REPETITION_PENALTY = 1.1

# Create FastAPI router
chat_router = APIRouter(prefix="", tags=["chat"])

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    history: list = []
    filters: dict = {}
    analytics: bool = False
    metadata_focus: list = []
    programs: list = []

class ChatResponse(BaseModel):
    reply: str
    sources: list = []
    filters_applied: dict = {}
    context_found: bool = False
    search_results_count: int = 0
    response_time_ms: int = 0
    ai_agent_status: str = "success"

# ============================================================================
# DIGITAL OCEAN AI AGENT CLIENT
# ============================================================================

async def call_digital_ocean_ai_agent(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Production Digital Ocean AI agent client for Llama 3.1 Instruct"""
    if not GENAI_ENDPOINT or not GENAI_ACCESS_KEY:
        return {
            "success": False, 
            "error": "AI agent not configured. Check GENAI_ENDPOINT and GENAI_ACCESS_KEY."
        }
    
    # Prepare payload for Llama 3.1 Instruct
    payload = {
        "model": GENAI_MODEL,
        "messages": format_messages_for_llama(messages),
        "max_tokens": LLAMA_MAX_TOKENS,
        "temperature": LLAMA_TEMPERATURE,
        "top_p": LLAMA_TOP_P,
        "repetition_penalty": LLAMA_REPETITION_PENALTY,
        "stream": False,
        "stop": ["<|eot_id|>", "<|end_of_text|>"]
    }
    
    headers = {
        "Authorization": f"Bearer {GENAI_ACCESS_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "InnovAI-ChatBot/4.1.0-Production"
    }
    
    for attempt in range(GENAI_MAX_RETRIES):
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GENAI_ENDPOINT,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=GENAI_TIMEOUT)
                ) as response:
                    
                    response_time = round((time.time() - start_time) * 1000)
                    
                    if response.status == 200:
                        data = await response.json()
                        reply = extract_llama_reply(data)
                        
                        if reply:
                            return {
                                "success": True, 
                                "reply": reply,
                                "response_time_ms": response_time,
                                "attempt": attempt + 1,
                                "model": "llama-3.1-8b-instruct"
                            }
                        else:
                            logger.warning("Empty response from Llama 3.1")
                            return {"success": False, "error": "Empty response from AI agent"}
                    
                    elif response.status == 401:
                        error_msg = "Authentication failed - check GENAI_ACCESS_KEY"
                        logger.error(error_msg)
                        return {"success": False, "error": error_msg}
                    
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    elif response.status == 404:
                        error_msg = f"AI endpoint not found: {GENAI_ENDPOINT}"
                        logger.error(error_msg)
                        return {"success": False, "error": error_msg}
                    
                    else:
                        error_text = await response.text()
                        error_msg = f"HTTP {response.status}: {error_text[:200]}"
                        
                        if attempt < GENAI_MAX_RETRIES - 1:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            return {"success": False, "error": error_msg}
        
        except asyncio.TimeoutError:
            error_msg = f"AI agent timeout after {GENAI_TIMEOUT}s"
            
            if attempt < GENAI_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                return {"success": False, "error": error_msg}
        
        except aiohttp.ClientError as e:
            error_msg = f"Network error: {str(e)}"
            
            if attempt < GENAI_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                return {"success": False, "error": error_msg}
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            
            if attempt < GENAI_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                return {"success": False, "error": error_msg}
    
    return {"success": False, "error": "All retry attempts failed"}

def format_messages_for_llama(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Format messages optimally for Llama 3.1 Instruct"""
    formatted_messages = []
    
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        if role == "system":
            formatted_content = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "user":
            formatted_content = f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "assistant":
            formatted_content = f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        else:
            formatted_content = content
        
        formatted_messages.append({
            "role": role,
            "content": formatted_content
        })
    
    return formatted_messages

def extract_llama_reply(response_data: Dict[str, Any]) -> Optional[str]:
    """Extract Llama 3.1 reply from response"""
    try:
        # OpenAI-compatible format
        if "choices" in response_data and len(response_data["choices"]) > 0:
            choice = response_data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                reply = choice["message"]["content"].strip()
                return clean_llama_response(reply)
        
        # Alternative formats
        for field in ["reply", "response", "text", "generated_text"]:
            if field in response_data:
                reply = str(response_data[field]).strip()
                return clean_llama_response(reply)
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting Llama reply: {e}")
        return None

def clean_llama_response(response: str) -> str:
    """Clean Llama 3.1 response by removing formatting tokens"""
    # Remove Llama 3.1 special tokens
    tokens_to_remove = [
        "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|start_header_id|>system<|end_header_id|>",
        "<|start_header_id|>user<|end_header_id|>"
    ]
    
    for token in tokens_to_remove:
        response = response.replace(token, "")
    
    # Clean up and filter system prompt leakage
    lines = response.strip().split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip lines that look like system prompt leakage
        if not (line.startswith("You are MetroAI") or 
                line.startswith("Your capabilities include") or
                line.startswith("Available Metadata Fields")):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

# ============================================================================
# CONTEXT BUILDING AND SEARCH
# ============================================================================

def build_search_context(message: str, filters: Dict[str, Any]) -> tuple[str, List[Dict]]:
    """Build context from OpenSearch with filter support"""
    context = ""
    sources = []
    
    try:
        from opensearch_client import get_connection_status, search_opensearch
        
        conn_status = get_connection_status()
        
        if not conn_status.get("connected", False):
            return context, sources
        
        # Use cleaned collection name
        index_name = "ai-corporate-sptr-test"
        
        # Perform search
        search_results = search_opensearch_with_filters(
            message, 
            filters=filters,
            index_override=index_name,
            size=8
        )
        
        if search_results:
            context_parts = []
            
            for result in search_results[:5]:
                source_data = result.get('_source', {})
                text = source_data.get('text', '')
                metadata = source_data.get('metadata', {})
                
                if text:
                    # Create rich context with metadata
                    context_piece = f"Evaluation: {text[:300]}"
                    if metadata:
                        context_piece += f"\nMetadata: Agent={metadata.get('agent', 'Unknown')}, "
                        context_piece += f"Disposition={metadata.get('disposition', 'Unknown')}, "
                        context_piece += f"Duration={metadata.get('call_duration', 'Unknown')}s, "
                        context_piece += f"Site={metadata.get('site', 'Unknown')}, "
                        context_piece += f"Partner={metadata.get('program', 'Unknown')}"
                    
                    context_parts.append(context_piece)
                    
                    sources.append({
                        'text': text,
                        'metadata': metadata,
                        'score': result.get('_score', 0),
                        'source_id': result.get('_id', ''),
                        'collection': source_data.get('collection_cleaned', 'unknown')
                    })
            
            context = "\n---\n".join(context_parts)
    
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Context search failed: {e}")
    
    return context, sources

def search_opensearch_with_filters(query: str, filters: Dict[str, Any] = None, 
                                 index_override: str = None, size: int = 10) -> List[Dict]:
    """Enhanced OpenSearch with filter support"""
    try:
        from opensearch_client import search_opensearch
        
        # Use basic search - can be enhanced with actual filter logic
        results = search_opensearch(query, index_override=index_override, size=size)
        
        return results
    
    except Exception as e:
        logger.error(f"Filtered search failed: {e}")
        return []

def build_system_message(is_analytics: bool, filters: Dict[str, Any], context: str) -> str:
    """Build system message optimized for Llama 3.1 Instruct"""
    if is_analytics:
        system_msg = """You are MetroAI Analytics, an expert call center analytics specialist for Metro by T-Mobile. You analyze call center evaluation data to provide actionable business insights.

## Your Expertise:
- Performance Analysis: Call metrics, resolution rates, quality scores
- Agent Performance: Individual evaluation, training needs, coaching
- Customer Experience: Satisfaction patterns, complaint analysis
- Operational Intelligence: Resource optimization, workflow efficiency
- Quality Assurance: Evaluation scoring, compliance monitoring
- Business Intelligence: Partner/site comparisons, trend identification

## Data Structure:
- **evaluationId**: Unique evaluation identifier
- **partner**: Call center partner (iQor, Teleperformance)
- **site**: Location (Dasma, Manila, Cebu)
- **lob**: Line of Business (WNP, Prepaid, Postpaid)
- **agentName**: Agent being evaluated
- **disposition**: Call category (Account, Technical Support, Billing)
- **subDisposition**: Specific call reason details
- **call_date**: Customer call timestamp
- **call_duration**: Call length in seconds
- **language**: Call language

## Analysis Instructions:
1. Provide specific, data-driven insights with examples
2. Include quantitative metrics when possible
3. Offer actionable recommendations for improvement
4. Reference actual evaluation content when making observations
5. Compare performance across agents, sites, or time periods when relevant

## Response Format:
- Start with key findings
- Provide supporting data and examples
- End with specific recommendations"""
    else:
        system_msg = """You are MetroAI, an intelligent assistant for Metro by T-Mobile customer service operations. 

You help with:
- General Metro by T-Mobile service inquiries
- Customer service best practices
- Call center operational guidance
- Policy and procedure questions

Provide helpful, accurate, and professional responses."""
    
    # Add filter context
    if filters:
        system_msg += f"\n\n## Current Analysis Filters:\n"
        for key, value in filters.items():
            if isinstance(value, list):
                system_msg += f"- **{key}**: {', '.join(map(str, value))}\n"
            else:
                system_msg += f"- **{key}**: {value}\n"
    
    # Add search context
    if context:
        system_msg += f"\n\n## Relevant Evaluation Data:\n{context}"
    
    return system_msg

# ============================================================================
# CHAT ENDPOINTS
# ============================================================================

@chat_router.post("/chat")
async def chat_handler(request: ChatRequest):
    """Production chat endpoint with AI agent integration"""
    start_time = time.time()
    
    try:
        # Extract filters and metadata focus
        filters = request.filters or {}
        is_analytics = request.analytics or False
        metadata_focus = request.metadata_focus or []
        
        # Build search context from OpenSearch
        context, sources = build_search_context(request.message, filters)
        
        # Build system message
        system_msg = build_system_message(is_analytics, filters, context)
        
        # Prepare messages for AI agent
        messages = [{"role": "system", "content": system_msg}]
        messages.extend([{"role": m["role"], "content": m["content"]} for m in request.history])
        messages.append({"role": "user", "content": request.message})
        
        # Call Digital Ocean AI agent
        ai_response = await call_digital_ocean_ai_agent(messages)
        
        # Calculate response time
        total_response_time = round((time.time() - start_time) * 1000)
        
        if ai_response.get("success"):
            reply = ai_response.get("reply", "")
            
            return ChatResponse(
                reply=reply.strip() if reply else "Sorry, I couldn't generate a response.",
                sources=sources[:5] if sources else [],
                filters_applied=filters,
                context_found=bool(context),
                search_results_count=len(sources),
                response_time_ms=total_response_time,
                ai_agent_status="success"
            )
        else:
            error_msg = ai_response.get("error", "Unknown error")
            logger.error(f"AI agent error: {error_msg}")
            
            return ChatResponse(
                reply="I'm sorry, but I'm having trouble connecting to the AI service right now. Please try again in a moment.",
                sources=[],
                filters_applied=filters,
                context_found=bool(context),
                search_results_count=len(sources),
                response_time_ms=total_response_time,
                ai_agent_status="error"
            )
            
    except Exception as e:
        total_response_time = round((time.time() - start_time) * 1000)
        logger.error(f"Chat handler error: {e}")
        
        return ChatResponse(
            reply="Sorry, there was an error processing your request. Please try again.",
            sources=[],
            filters_applied=request.filters or {},
            context_found=False,
            search_results_count=0,
            response_time_ms=total_response_time,
            ai_agent_status="error"
        )

# ============================================================================
# TESTING AND HEALTH ENDPOINTS
# ============================================================================

@chat_router.get("/test_ai_agent")
async def test_ai_agent():
    """Test Digital Ocean AI agent connectivity"""
    try:
        if not GENAI_ACCESS_KEY or not GENAI_ENDPOINT:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "AI agent not configured. Set GENAI_ENDPOINT and GENAI_ACCESS_KEY."
                }
            )
        
        # Test basic connectivity
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond clearly and concisely."},
            {"role": "user", "content": "Hello! Please respond with exactly: 'AI connection test successful'"}
        ]
        
        basic_response = await call_digital_ocean_ai_agent(test_messages)
        
        # Test analytics capability
        analytics_messages = [
            {
                "role": "system", 
                "content": """You are MetroAI Analytics, an expert call center analytics specialist for Metro by T-Mobile.

Sample Data: Agent: Rey Mendoza, Disposition: Port Out, Duration: 491s, Site: Dasma, Partner: iQor

Analyze this briefly."""
            },
            {
                "role": "user", 
                "content": "Based on the sample data, what insights can you provide about this port out interaction?"
            }
        ]
        
        analytics_response = await call_digital_ocean_ai_agent(analytics_messages)
        
        return {
            "status": "success",
            "basic_connectivity": {
                "success": basic_response.get("success", False),
                "response_time_ms": basic_response.get("response_time_ms", 0),
                "response_preview": basic_response.get("reply", "")[:100],
                "error": basic_response.get("error")
            },
            "analytics_capability": {
                "success": analytics_response.get("success", False),
                "response_time_ms": analytics_response.get("response_time_ms", 0),
                "response_length": len(analytics_response.get("reply", "")),
                "error": analytics_response.get("error")
            },
            "configuration": {
                "endpoint": GENAI_ENDPOINT,
                "has_access_key": bool(GENAI_ACCESS_KEY),
                "model": GENAI_MODEL,
                "timeout": GENAI_TIMEOUT,
                "max_retries": GENAI_MAX_RETRIES
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"AI agent test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"AI agent test failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@chat_router.get("/chat/health")
async def chat_health():
    """Chat-specific health check"""
    try:
        if GENAI_ACCESS_KEY and GENAI_ENDPOINT:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Respond with 'Health check OK'"}
            ]
            
            response = await call_digital_ocean_ai_agent(test_messages)
            
            return {
                "status": "healthy" if response.get("success") else "degraded",
                "ai_agent": {
                    "configured": True,
                    "connected": response.get("success", False),
                    "response_time_ms": response.get("response_time_ms", 0),
                    "error": response.get("error")
                },
                "configuration": {
                    "endpoint": GENAI_ENDPOINT,
                    "model": GENAI_MODEL,
                    "timeout": GENAI_TIMEOUT
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "not_configured",
                "ai_agent": {
                    "configured": False,
                    "connected": False,
                    "missing": [
                        var for var in ["GENAI_ENDPOINT", "GENAI_ACCESS_KEY"] 
                        if not os.getenv(var)
                    ]
                },
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Chat health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# ============================================================================
# INITIALIZATION
# ============================================================================

logger.info("🚀 Production Chat handlers module loaded")
logger.info(f"   AI Model: {GENAI_MODEL}")
logger.info(f"   Endpoint: {'✅ Configured' if GENAI_ENDPOINT else '❌ Missing'}")
logger.info(f"   Access Key: {'✅ Configured' if GENAI_ACCESS_KEY else '❌ Missing'}")

if GENAI_ENDPOINT and GENAI_ACCESS_KEY:
    logger.info("🤖 Production AI integration ready")
else:
    logger.warning("⚠️ AI agent not fully configured - check environment variables")