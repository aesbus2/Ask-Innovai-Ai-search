# chat_handlers.py - Digital Ocean AI Agent Integration
# Version: 1.0.0 - Separated Chat Logic with Enhanced AI Integration

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

# Setup logging
logger = logging.getLogger("ask-innovai.chat")

# Environment configuration for Digital Ocean AI Agent (Llama 3.1 Instruct 8B)
GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT", "")
GENAI_ACCESS_KEY = os.getenv("GENAI_ACCESS_KEY", "")
GENAI_MODEL = os.getenv("GENAI_MODEL", "llama-3.1-8b-instruct")
GENAI_TIMEOUT = int(os.getenv("GENAI_TIMEOUT", "120"))  # Llama may need more time
GENAI_MAX_RETRIES = int(os.getenv("GENAI_MAX_RETRIES", "3"))

# Llama 3.1 specific configuration (optimized defaults)
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
    programs: list = []  # Keep for backward compatibility

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
    """
    Enhanced Digital Ocean AI agent client optimized for Llama 3.1 Instruct (8B)
    """
    if not GENAI_ENDPOINT or not GENAI_ACCESS_KEY:
        return {
            "success": False, 
            "error": "AI agent not configured. Check GENAI_ENDPOINT and GENAI_ACCESS_KEY."
        }
    
    # Prepare payload optimized for Llama 3.1 Instruct
    payload = {
        "model": GENAI_MODEL,
        "messages": format_messages_for_llama(messages),
        "max_tokens": LLAMA_MAX_TOKENS,
        "temperature": LLAMA_TEMPERATURE,
        "top_p": LLAMA_TOP_P,
        "repetition_penalty": LLAMA_REPETITION_PENALTY,
        "stream": False,
        "stop": ["<|eot_id|>", "<|end_of_text|>"]  # Llama 3.1 stop tokens
    }
    
    headers = {
        "Authorization": f"Bearer {GENAI_ACCESS_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "InnovAI-ChatBot/3.3.0-Llama"
    }
    
    for attempt in range(GENAI_MAX_RETRIES):
        try:
            logger.info(f"🦙 Calling Llama 3.1 Instruct on Digital Ocean (attempt {attempt + 1}/{GENAI_MAX_RETRIES})")
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GENAI_ENDPOINT,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=GENAI_TIMEOUT)
                ) as response:
                    
                    response_time = round((time.time() - start_time) * 1000)  # milliseconds
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Handle different response formats
                        reply = extract_llama_reply(data)
                        
                        if reply:
                            logger.info(f"✅ Llama 3.1 response received ({len(reply)} chars, {response_time}ms)")
                            return {
                                "success": True, 
                                "reply": reply,
                                "response_time_ms": response_time,
                                "attempt": attempt + 1,
                                "model": "llama-3.1-8b-instruct"
                            }
                        else:
                            logger.warning(f"⚠️ Llama 3.1 returned empty response: {data}")
                            return {"success": False, "error": "Empty response from Llama 3.1"}
                    
                    elif response.status == 401:
                        error_msg = "Authentication failed - check GENAI_ACCESS_KEY"
                        logger.error(f"❌ {error_msg}")
                        return {"success": False, "error": error_msg}
                    
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        wait_time = 2 ** attempt
                        logger.warning(f"⚠️ Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    elif response.status == 404:
                        error_msg = f"Llama 3.1 endpoint not found: {GENAI_ENDPOINT}"
                        logger.error(f"❌ {error_msg}")
                        return {"success": False, "error": error_msg}
                    
                    else:
                        error_text = await response.text()
                        error_msg = f"HTTP {response.status}: {error_text[:200]}"
                        logger.error(f"❌ Llama 3.1 error: {error_msg}")
                        
                        if attempt < GENAI_MAX_RETRIES - 1:
                            wait_time = 2 ** attempt
                            logger.info(f"🔄 Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            return {"success": False, "error": error_msg}
        
        except asyncio.TimeoutError:
            error_msg = f"Llama 3.1 timeout after {GENAI_TIMEOUT}s"
            logger.warning(f"⚠️ {error_msg}, attempt {attempt + 1}/{GENAI_MAX_RETRIES}")
            
            if attempt < GENAI_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                return {"success": False, "error": error_msg}
        
        except aiohttp.ClientError as e:
            error_msg = f"Network error: {str(e)}"
            logger.warning(f"⚠️ {error_msg}, attempt {attempt + 1}/{GENAI_MAX_RETRIES}")
            
            if attempt < GENAI_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                return {"success": False, "error": error_msg}
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"❌ {error_msg}, attempt {attempt + 1}/{GENAI_MAX_RETRIES}")
            
            if attempt < GENAI_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                return {"success": False, "error": error_msg}
    
    return {"success": False, "error": "All retry attempts failed"}

def format_messages_for_llama(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Format messages optimally for Llama 3.1 Instruct
    """
    formatted_messages = []
    
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        # Llama 3.1 instruction format optimization
        if role == "system":
            # Llama responds well to clear, structured system prompts
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
    """
    Extract Llama 3.1 reply from different response formats
    """
    try:
        # OpenAI-compatible format
        if "choices" in response_data and len(response_data["choices"]) > 0:
            choice = response_data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                reply = choice["message"]["content"].strip()
                return clean_llama_response(reply)
        
        # Direct reply format
        if "reply" in response_data:
            reply = str(response_data["reply"]).strip()
            return clean_llama_response(reply)
        
        # Response field format
        if "response" in response_data:
            reply = str(response_data["response"]).strip()
            return clean_llama_response(reply)
        
        # Text field format
        if "text" in response_data:
            reply = str(response_data["text"]).strip()
            return clean_llama_response(reply)
        
        # Generated text format (common for Llama)
        if "generated_text" in response_data:
            reply = str(response_data["generated_text"]).strip()
            return clean_llama_response(reply)
        
        logger.warning(f"⚠️ Unknown Llama response format: {list(response_data.keys())}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error extracting Llama reply: {e}")
        return None

def clean_llama_response(response: str) -> str:
    """
    Clean Llama 3.1 response by removing formatting tokens and artifacts
    """
    # Remove Llama 3.1 special tokens
    response = response.replace("<|begin_of_text|>", "")
    response = response.replace("<|end_of_text|>", "")
    response = response.replace("<|eot_id|>", "")
    response = response.replace("<|start_header_id|>assistant<|end_header_id|>", "")
    response = response.replace("<|start_header_id|>system<|end_header_id|>", "")
    response = response.replace("<|start_header_id|>user<|end_header_id|>", "")
    
    # Clean up any remaining artifacts
    response = response.strip()
    
    # Remove any repeated system prompts that might leak through
    lines = response.split('\n')
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
    """
    Build context from OpenSearch with comprehensive filter support
    """
    context = ""
    sources = []
    
    try:
        # Import here to avoid circular imports
        from opensearch_client import get_connection_status, search_opensearch
        
        conn_status = get_connection_status()
        
        if not conn_status.get("connected", False):
            logger.warning("⚠️ OpenSearch not available for context search")
            return context, sources
        
        # Determine index based on filters or use default
        index_name = "ai-corporate-sptr-test"  # Your main cleaned collection name
        
        # Perform search with filters
        search_results = search_opensearch_with_filters(
            message, 
            filters=filters,
            index_override=index_name,
            size=8
        )
        
        if search_results:
            context_parts = []
            
            for result in search_results[:5]:  # Use top 5 results
                source_data = result.get('_source', {})
                text = source_data.get('text', '')
                metadata = source_data.get('metadata', {})
                
                if text:
                    # Create rich context with metadata
                    context_piece = f"Evaluation: {text[:300]}"
                    if metadata:
                        context_piece += f"\nMetadata: Agent={metadata.get('agentName', 'Unknown')}, "
                        context_piece += f"Disposition={metadata.get('disposition', 'Unknown')}, "
                        context_piece += f"SubDisposition={metadata.get('subDisposition', 'None')}, "
                        context_piece += f"Duration={metadata.get('call_duration', 'Unknown')}s, "
                        context_piece += f"Site={metadata.get('site', 'Unknown')}, "
                        context_piece += f"Partner={metadata.get('partner', 'Unknown')}"
                    
                    context_parts.append(context_piece)
                    
                    # Add to sources for frontend display
                    sources.append({
                        'text': text,
                        'metadata': metadata,
                        'score': result.get('_score', 0),
                        'source_id': result.get('_id', ''),
                        'collection': source_data.get('collection_cleaned', 'unknown')
                    })
            
            context = "\n---\n".join(context_parts)
            logger.info(f"📊 Built context from {len(sources)} search results")
        
        else:
            logger.info("📊 No search results found for context")
    
    except ImportError:
        logger.warning("⚠️ OpenSearch client not available")
    except Exception as e:
        logger.error(f"❌ Context search failed: {e}")
    
    return context, sources

def search_opensearch_with_filters(query: str, filters: Dict[str, Any] = None, 
                                 index_override: str = None, size: int = 10) -> List[Dict]:
    """
    Enhanced OpenSearch with filter support (placeholder for now)
    This would be implemented based on your OpenSearch schema
    """
    try:
        # Import here to avoid circular imports
        from opensearch_client import search_opensearch
        
        # For now, use basic search - you can enhance this with actual filter logic
        results = search_opensearch(query, index_override=index_override, size=size)
        
        # Future enhancement: Apply filters to search query
        # This is where you'd build OpenSearch query with filters
        if filters:
            logger.info(f"🔍 Search with filters: {list(filters.keys())}")
            # TODO: Implement actual filter application to OpenSearch query
        
        return results
    
    except Exception as e:
        logger.error(f"❌ Filtered search failed: {e}")
        return []

def build_system_message(is_analytics: bool, filters: Dict[str, Any], context: str) -> str:
    """
    Build comprehensive system message optimized for Llama 3.1 Instruct
    """
    if is_analytics:
        system_msg = """You are MetroAI Analytics, an expert call center analytics specialist for Metro by T-Mobile. You analyze call center evaluation data to provide actionable business insights.

## Your Expertise:
- Performance Analysis: Call metrics, resolution rates, quality scores, efficiency analysis
- Agent Performance: Individual evaluation, training needs, performance coaching
- Customer Experience: Satisfaction patterns, complaint analysis, service quality metrics
- Operational Intelligence: Resource optimization, workflow efficiency, peak time analysis
- Quality Assurance: Evaluation scoring, compliance monitoring, improvement recommendations
- Business Intelligence: Partner/site comparisons, LOB analysis, trend identification

## Data Structure You Analyze:
- **internalId**: Primary evaluation reference ID
- **evaluationId**: Unique evaluation identifier  
- **template_id**: Evaluation template (use this over template_name)
- **partner**: Call center partner (iQor, Teleperformance, etc.)
- **site**: Location (Dasma, Manila, Cebu, etc.)
- **lob**: Line of Business (WNP, Prepaid, Postpaid, Business)
- **agentName**: Agent being evaluated
- **agentId**: Agent identifier for searches
- **disposition**: Call category (Account, Technical Support, Billing, Port Out)
- **subDisposition**: Specific call reason details
- **call_date**: Customer call timestamp (use for temporal analysis)
- **call_duration**: Call length in seconds
- **language**: Call language (english, spanish, etc.)

## Analysis Instructions:
1. Always provide specific, data-driven insights with examples
2. Include quantitative metrics when possible
3. Offer actionable recommendations for improvement
4. Reference actual evaluation content when making observations
5. Compare performance across agents, sites, or time periods when relevant
6. Use call_date for temporal analysis, created_on for workflow analysis
7. Always reference template_id for consistency over template_name

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
    
    # Add filter context with clear formatting for Llama
    if filters:
        system_msg += f"\n\n## Current Analysis Filters:\n"
        
        for key, value in filters.items():
            if isinstance(value, list):
                system_msg += f"- **{key}**: {', '.join(map(str, value))}\n"
            elif key in ['startCallDate', 'endCallDate', 'startCreatedDate', 'endCreatedDate']:
                system_msg += f"- **{key}**: {value}\n"
            else:
                system_msg += f"- **{key}**: {value}\n"
    
    # Add search context with clear formatting for Llama
    if context:
        system_msg += f"\n\n## Relevant Evaluation Data:\n{context}"
    
    return system_msg

# ============================================================================
# CHAT ENDPOINTS
# ============================================================================

@chat_router.post("/chat")
async def chat_handler(request: ChatRequest):
    """
    Enhanced chat endpoint with Digital Ocean AI agent integration
    """
    start_time = time.time()
    
    try:
        logger.info(f"💬 Processing chat request: {request.message[:50]}...")
        
        # Extract comprehensive filters and metadata focus
        filters = request.filters or {}
        is_analytics = request.analytics or False
        metadata_focus = request.metadata_focus or []
        
        # Build search context from OpenSearch
        context, sources = build_search_context(request.message, filters)
        
        # Build enhanced system message
        system_msg = build_system_message(is_analytics, filters, context)
        
        # Prepare messages for AI agent
        messages = [{"role": "system", "content": system_msg}]
        messages.extend([{"role": m["role"], "content": m["content"]} for m in request.history])
        messages.append({"role": "user", "content": request.message})
        
        logger.info(f"🤖 Sending {len(messages)} messages to Digital Ocean AI agent")
        
        # Call Digital Ocean AI agent
        ai_response = await call_digital_ocean_ai_agent(messages)
        
        # Calculate total response time
        total_response_time = round((time.time() - start_time) * 1000)
        
        if ai_response.get("success"):
            reply = ai_response.get("reply", "")
            ai_response_time = ai_response.get("response_time_ms", 0)
            
            logger.info(f"✅ Chat completed successfully ({total_response_time}ms total)")
            
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
            logger.error(f"❌ AI agent error: {error_msg}")
            
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
        logger.error(f"❌ Chat handler error: {e}")
        
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
    """Test Digital Ocean AI agent connectivity and functionality"""
    try:
        if not GENAI_ACCESS_KEY or not GENAI_ENDPOINT:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "AI agent not configured. Set GENAI_ENDPOINT and GENAI_ACCESS_KEY."
                }
            )
        
        # Test basic connectivity with Llama-optimized prompt
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond clearly and concisely."},
            {"role": "user", "content": "Hello! Please respond with exactly: 'Llama 3.1 connection test successful'"}
        ]
        
        basic_response = await call_digital_ocean_ai_agent(test_messages)
        
        # Test analytics capability with Llama-optimized analytics prompt
        analytics_messages = [
            {
                "role": "system", 
                "content": """You are MetroAI Analytics, an expert call center analytics specialist for Metro by T-Mobile.

## Sample Data:
Agent: Rey Mendoza, Disposition: Port Out, Duration: 491s, Site: Dasma, Partner: iQor

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
        logger.error(f"❌ AI agent test failed: {e}")
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
        # Quick connectivity test optimized for Llama 3.1
        if GENAI_ACCESS_KEY and GENAI_ENDPOINT:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Respond with 'Llama health check OK'"}
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
        logger.error(f"❌ Chat health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# ============================================================================
# INITIALIZATION LOG
# ============================================================================

logger.info("🚀 Chat handlers module loaded - Optimized for Llama 3.1 Instruct (8B)")
logger.info(f"   AI Model: {GENAI_MODEL}")
logger.info(f"   Endpoint: {'✅ Configured' if GENAI_ENDPOINT else '❌ Missing'}")
logger.info(f"   Access Key: {'✅ Configured' if GENAI_ACCESS_KEY else '❌ Missing'}")
logger.info(f"   Max Tokens: {LLAMA_MAX_TOKENS}")
logger.info(f"   Temperature: {LLAMA_TEMPERATURE}")
logger.info(f"   Top P: {LLAMA_TOP_P}")
logger.info(f"   Repetition Penalty: {LLAMA_REPETITION_PENALTY}")
logger.info(f"   Timeout: {GENAI_TIMEOUT}s")
logger.info(f"   Max Retries: {GENAI_MAX_RETRIES}")

if GENAI_ENDPOINT and GENAI_ACCESS_KEY:
    logger.info("🦙 Llama 3.1 Instruct (8B) integration ready")
    logger.info("   ✅ Optimized instruction formatting")
    logger.info("   ✅ Enhanced response cleaning")
    logger.info("   ✅ Llama-specific parameters")
else:
    logger.warning("⚠️ Llama 3.1 not fully configured - check environment variables")