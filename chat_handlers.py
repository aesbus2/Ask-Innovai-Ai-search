# chat_handlers.py - Digital Ocean AI Agent Integration with Proper Authentication
# Version: 2.0.0 - Updated with Digital Ocean OpenAI Client Format

import os
import logging
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger("ask-innovai.chat")

# Environment configuration for Digital Ocean AI Agent
GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT", "")
GENAI_ACCESS_KEY = os.getenv("GENAI_ACCESS_KEY", "")
GENAI_MODEL = os.getenv("GENAI_MODEL", "llama-3.1-8b-instruct")
GENAI_TIMEOUT = int(os.getenv("GENAI_TIMEOUT", "120"))
GENAI_MAX_RETRIES = int(os.getenv("GENAI_MAX_RETRIES", "3"))

# Enhanced configuration for Digital Ocean format
AGENT_ENDPOINT = GENAI_ENDPOINT
AGENT_ACCESS_KEY = GENAI_ACCESS_KEY

# Digital Ocean specific settings
DO_INCLUDE_RETRIEVAL_INFO = os.getenv("DO_INCLUDE_RETRIEVAL_INFO", "true").lower() == "true"

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
    retrieval_info: dict = {}  # Add retrieval info for DO format

# ============================================================================
# DIGITAL OCEAN AI AGENT CLIENT (Updated Format)
# ============================================================================

async def call_digital_ocean_ai_agent(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Digital Ocean AI agent client using OpenAI format with proper authentication
    """
    if not AGENT_ENDPOINT or not AGENT_ACCESS_KEY:
        return {
            "success": False, 
            "error": "AI agent not configured. Check GENAI_ENDPOINT and GENAI_ACCESS_KEY."
        }
    
    try:
        # Import OpenAI client
        from openai import AsyncOpenAI
    except ImportError:
        return {
            "success": False,
            "error": "OpenAI client not installed. Run: pip install openai"
        }
    
    # Prepare the base URL in Digital Ocean format
    base_url = AGENT_ENDPOINT
    if not base_url.endswith('/'):
        base_url += '/'
    if not base_url.endswith('api/v1/'):
        base_url += 'api/v1/'
    
    logger.info(f"🔗 Using Digital Ocean base URL: {base_url}")
    
    for attempt in range(GENAI_MAX_RETRIES):
        try:
            logger.info(f"🦙 Calling Digital Ocean AI Agent (attempt {attempt + 1}/{GENAI_MAX_RETRIES})")
            start_time = time.time()
            
            # Create OpenAI client with Digital Ocean configuration
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=AGENT_ACCESS_KEY,
                timeout=GENAI_TIMEOUT
            )
            
            # Prepare messages for Digital Ocean format
            formatted_messages = format_messages_for_digital_ocean(messages)
            
            # Create completion with Digital Ocean specific parameters
            response = await client.chat.completions.create(
                model="n/a",  # Digital Ocean uses "n/a" as model
                messages=formatted_messages,
                extra_body={
                    "include_retrieval_info": DO_INCLUDE_RETRIEVAL_INFO,
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "top_p": 0.9
                }
            )
            
            response_time = round((time.time() - start_time) * 1000)  # milliseconds
            
            # Extract response content
            if response.choices and len(response.choices) > 0:
                reply = response.choices[0].message.content
                
                # Extract retrieval information if available
                retrieval_info = {}
                try:
                    response_dict = response.to_dict()
                    if "retrieval" in response_dict:
                        retrieval_info = response_dict["retrieval"]
                        logger.info(f"📚 Retrieved info from {len(retrieval_info.get('documents', []))} documents")
                except Exception as e:
                    logger.warning(f"⚠️ Could not extract retrieval info: {e}")
                
                if reply:
                    logger.info(f"✅ Digital Ocean response received ({len(reply)} chars, {response_time}ms)")
                    return {
                        "success": True,
                        "reply": reply,
                        "response_time_ms": response_time,
                        "attempt": attempt + 1,
                        "model": "digital-ocean-agent",
                        "retrieval_info": retrieval_info
                    }
                else:
                    logger.warning(f"⚠️ Digital Ocean returned empty response")
                    if attempt < GENAI_MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        return {"success": False, "error": "Empty response from Digital Ocean agent"}
            else:
                logger.warning(f"⚠️ No choices in Digital Ocean response")
                if attempt < GENAI_MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return {"success": False, "error": "No response choices from Digital Ocean agent"}
        
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific error types
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                logger.error(f"❌ Authentication failed - check GENAI_ACCESS_KEY")
                return {"success": False, "error": "Authentication failed - check GENAI_ACCESS_KEY"}
            
            elif "404" in error_msg or "not found" in error_msg.lower():
                logger.error(f"❌ Endpoint not found: {base_url}")
                return {"success": False, "error": f"Endpoint not found: {base_url}"}
            
            elif "timeout" in error_msg.lower():
                logger.warning(f"⚠️ Timeout on attempt {attempt + 1}: {error_msg}")
                if attempt < GENAI_MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return {"success": False, "error": f"Timeout after {GENAI_MAX_RETRIES} attempts"}
            
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                logger.warning(f"⚠️ Rate limited on attempt {attempt + 1}")
                if attempt < GENAI_MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"🔄 Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return {"success": False, "error": "Rate limited - too many requests"}
            
            else:
                logger.error(f"❌ Digital Ocean error on attempt {attempt + 1}: {error_msg}")
                if attempt < GENAI_MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return {"success": False, "error": f"Digital Ocean error: {error_msg}"}
    
    return {"success": False, "error": "All retry attempts failed"}

def format_messages_for_digital_ocean(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Format messages for Digital Ocean AI agent (keeping standard OpenAI format)
    """
    formatted_messages = []
    
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        # Digital Ocean uses standard OpenAI message format
        formatted_messages.append({
            "role": role,
            "content": content
        })
    
    return formatted_messages

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
    Enhanced OpenSearch with filter support
    """
    try:
        # Import here to avoid circular imports
        from opensearch_client import search_opensearch
        
        # For now, use basic search - you can enhance this with actual filter logic
        results = search_opensearch(query, index_override=index_override, size=size)
        
        # Future enhancement: Apply filters to search query
        if filters:
            logger.info(f"🔍 Search with filters: {list(filters.keys())}")
            # TODO: Implement actual filter application to OpenSearch query
        
        return results
    
    except Exception as e:
        logger.error(f"❌ Filtered search failed: {e}")
        return []

def build_system_message(is_analytics: bool, filters: Dict[str, Any], context: str) -> str:
    """
    Build comprehensive system message optimized for Digital Ocean AI Agent
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
    
    # Add filter context
    if filters:
        system_msg += f"\n\n## Current Analysis Filters:\n"
        
        for key, value in filters.items():
            if isinstance(value, list):
                system_msg += f"- **{key}**: {', '.join(map(str, value))}\n"
            elif key in ['startCallDate', 'endCallDate', 'startCreatedDate', 'endCreatedDate']:
                system_msg += f"- **{key}**: {value}\n"
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
    """
    Enhanced chat endpoint with Digital Ocean AI agent integration using proper authentication
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
        
        # Call Digital Ocean AI agent with proper authentication
        ai_response = await call_digital_ocean_ai_agent(messages)
        
        # Calculate total response time
        total_response_time = round((time.time() - start_time) * 1000)
        
        if ai_response.get("success"):
            reply = ai_response.get("reply", "")
            ai_response_time = ai_response.get("response_time_ms", 0)
            retrieval_info = ai_response.get("retrieval_info", {})
            
            logger.info(f"✅ Chat completed successfully ({total_response_time}ms total)")
            
            return ChatResponse(
                reply=reply.strip() if reply else "Sorry, I couldn't generate a response.",
                sources=sources[:5] if sources else [],
                filters_applied=filters,
                context_found=bool(context),
                search_results_count=len(sources),
                response_time_ms=total_response_time,
                ai_agent_status="success",
                retrieval_info=retrieval_info
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
                ai_agent_status="error",
                retrieval_info={}
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
            ai_agent_status="error",
            retrieval_info={}
        )

# ============================================================================
# TESTING AND HEALTH ENDPOINTS
# ============================================================================

@chat_router.get("/test_ai_agent")
async def test_ai_agent():
    """Test Digital Ocean AI agent connectivity with proper authentication"""
    try:
        if not AGENT_ACCESS_KEY or not AGENT_ENDPOINT:
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
            {"role": "user", "content": "Hello! Please respond with exactly: 'Digital Ocean connection test successful'"}
        ]
        
        basic_response = await call_digital_ocean_ai_agent(test_messages)
        
        # Test analytics capability
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
        
        # Prepare base URL for display
        base_url = AGENT_ENDPOINT
        if not base_url.endswith('/'):
            base_url += '/'
        if not base_url.endswith('api/v1/'):
            base_url += 'api/v1/'
        
        return {
            "status": "success",
            "basic_connectivity": {
                "success": basic_response.get("success", False),
                "response_time_ms": basic_response.get("response_time_ms", 0),
                "response_preview": basic_response.get("reply", "")[:100],
                "error": basic_response.get("error"),
                "retrieval_info": basic_response.get("retrieval_info", {})
            },
            "analytics_capability": {
                "success": analytics_response.get("success", False),
                "response_time_ms": analytics_response.get("response_time_ms", 0),
                "response_length": len(analytics_response.get("reply", "")),
                "error": analytics_response.get("error"),
                "retrieval_info": analytics_response.get("retrieval_info", {})
            },
            "configuration": {
                "endpoint": AGENT_ENDPOINT,
                "base_url": base_url,
                "has_access_key": bool(AGENT_ACCESS_KEY),
                "include_retrieval_info": DO_INCLUDE_RETRIEVAL_INFO,
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
    """Chat-specific health check with Digital Ocean format"""
    try:
        # Quick connectivity test
        if AGENT_ACCESS_KEY and AGENT_ENDPOINT:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Respond with 'Digital Ocean health check OK'"}
            ]
            
            response = await call_digital_ocean_ai_agent(test_messages)
            
            # Prepare base URL for display
            base_url = AGENT_ENDPOINT
            if not base_url.endswith('/'):
                base_url += '/'
            if not base_url.endswith('api/v1/'):
                base_url += 'api/v1/'
            
            return {
                "status": "healthy" if response.get("success") else "degraded",
                "ai_agent": {
                    "configured": True,
                    "connected": response.get("success", False),
                    "response_time_ms": response.get("response_time_ms", 0),
                    "error": response.get("error"),
                    "retrieval_available": bool(response.get("retrieval_info"))
                },
                "configuration": {
                    "endpoint": AGENT_ENDPOINT,
                    "base_url": base_url,
                    "authentication_format": "Digital Ocean OpenAI Client",
                    "include_retrieval_info": DO_INCLUDE_RETRIEVAL_INFO,
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

logger.info("🚀 Chat handlers module loaded - Digital Ocean OpenAI Client Format")
logger.info(f"   Endpoint: {'✅ Configured' if AGENT_ENDPOINT else '❌ Missing'}")
logger.info(f"   Access Key: {'✅ Configured' if AGENT_ACCESS_KEY else '❌ Missing'}")
logger.info(f"   Include Retrieval: {DO_INCLUDE_RETRIEVAL_INFO}")
logger.info(f"   Timeout: {GENAI_TIMEOUT}s")
logger.info(f"   Max Retries: {GENAI_MAX_RETRIES}")

if AGENT_ENDPOINT and AGENT_ACCESS_KEY:
    # Prepare base URL for logging
    base_url = AGENT_ENDPOINT
    if not base_url.endswith('/'):
        base_url += '/'
    if not base_url.endswith('api/v1/'):
        base_url += 'api/v1/'
    
    logger.info("🦙 Digital Ocean AI Agent integration ready")
    logger.info(f"   ✅ Base URL: {base_url}")
    logger.info("   ✅ OpenAI client format")
    logger.info("   ✅ Proper authentication")
    logger.info("   ✅ Retrieval info support")
else:
    logger.warning("⚠️ Digital Ocean AI Agent not fully configured - check environment variables")
    logger.warning("   Required: GENAI_ENDPOINT, GENAI_ACCESS_KEY")