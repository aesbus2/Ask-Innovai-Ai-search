# Enhanced chat_handlers.py - Evaluation Grouped Search Support
# Version: 4.0.0 - Updated for template_ID collections and evaluation grouping

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
    evaluation_sources: int = 0  # NEW: Count of evaluation documents found

# ============================================================================
# DIGITAL OCEAN AI AGENT CLIENT (Keeping existing implementation)
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
# ENHANCED CONTEXT BUILDING AND SEARCH - EVALUATION GROUPED
# ============================================================================

def build_search_context(message: str, filters: Dict[str, Any]) -> tuple[str, List[Dict]]:
    """
    ENHANCED: Build context from OpenSearch with evaluation-grouped structure
    Now searches evaluation documents instead of individual chunks
    """
    context = ""
    sources = []
    
    try:
        # Import here to avoid circular imports
        from opensearch_client import get_connection_status, search_opensearch, search_evaluation_chunks
        
        conn_status = get_connection_status()
        
        if not conn_status.get("connected", False):
            logger.warning("⚠️ OpenSearch not available for context search")
            return context, sources
        
        # ENHANCED: Search evaluation documents (not individual chunks)
        search_results = search_opensearch_with_filters(
            message, 
            filters=filters,
            index_override=None,  # Search all template collections
            size=8
        )
        
        if search_results:
            context_parts = []
            evaluation_count = 0
            
            for result in search_results[:5]:  # Use top 5 evaluation results
                source_data = result.get('_source', {})
                
                # Extract evaluation-level information
                evaluation_id = source_data.get('evaluationId', 'Unknown')
                template_name = source_data.get('template_name', 'Unknown Template')
                total_chunks = source_data.get('total_chunks', 0)
                metadata = source_data.get('metadata', {})
                
                # Get text content (full_text is the combined text of all chunks)
                full_text = source_data.get('full_text', '')
                evaluation_text = source_data.get('evaluation_text', '')
                transcript_text = source_data.get('transcript_text', '')
                
                if full_text:
                    evaluation_count += 1
                    
                    # Create rich context with evaluation-level metadata
                    context_piece = f"Evaluation ID: {evaluation_id}\n"
                    context_piece += f"Template: {template_name}\n"
                    context_piece += f"Agent: {metadata.get('agent', 'Unknown')}\n"
                    context_piece += f"Call Details: {metadata.get('disposition', 'Unknown')} - {metadata.get('sub_disposition', 'None')}\n"
                    context_piece += f"Duration: {metadata.get('call_duration', 'Unknown')}s, "
                    context_piece += f"Site: {metadata.get('site', 'Unknown')}, "
                    context_piece += f"Partner: {metadata.get('partner', 'Unknown')}, "
                    context_piece += f"LOB: {metadata.get('lob', 'Unknown')}\n"
                    context_piece += f"Call Date: {metadata.get('call_date', 'Unknown')}\n"
                    
                    # Add relevant content (limit to prevent token overflow)
                    if evaluation_text and len(evaluation_text) > 0:
                        context_piece += f"Evaluation Content: {evaluation_text[:400]}...\n"
                    
                    if transcript_text and len(transcript_text) > 0:
                        context_piece += f"Call Transcript: {transcript_text[:300]}...\n"
                    
                    context_piece += f"Total Chunks: {total_chunks}\n"
                    context_piece += "---"
                    
                    context_parts.append(context_piece)
                    
                    # Add to sources for frontend display (enhanced with evaluation info)
                    sources.append({
                        'text': full_text[:500] if full_text else 'No content available',  # Preview text
                        'evaluation_id': evaluation_id,
                        'template_id': source_data.get('template_id', ''),
                        'template_name': template_name,
                        'total_chunks': total_chunks,
                        'metadata': metadata,
                        'score': result.get('_score', 0),
                        'source_id': result.get('_id', ''),
                        'collection': result.get('_index', 'unknown'),
                        'document_type': 'evaluation',
                        'evaluation_chunks_count': source_data.get('evaluation_chunks_count', 0),
                        'transcript_chunks_count': source_data.get('transcript_chunks_count', 0)
                    })
            
            context = "\n\n".join(context_parts)
            logger.info(f"📊 Built context from {evaluation_count} evaluation documents ({len(search_results)} total results)")
        
        else:
            logger.info("📊 No evaluation search results found for context")
    
    except ImportError:
        logger.warning("⚠️ OpenSearch client not available")
    except Exception as e:
        logger.error(f"❌ Context search failed: {e}")
    
    return context, sources

def search_opensearch_with_filters(query: str, filters: Dict[str, Any] = None, 
                                 index_override: str = None, size: int = 10) -> List[Dict]:
    """
    ENHANCED: OpenSearch with filter support for evaluation documents
    """
    try:
        # Import here to avoid circular imports
        from opensearch_client import search_opensearch
        
        # Use the enhanced search that returns evaluation documents
        results = search_opensearch(query, index_override=index_override, 
                                  filters=filters, size=size)
        
        # Log search details
        if filters:
            logger.info(f"🔍 Evaluation search with filters: {list(filters.keys())}")
        
        logger.info(f"🔍 Found {len(results)} evaluation documents")
        
        return results
    
    except Exception as e:
        logger.error(f"❌ Filtered evaluation search failed: {e}")
        return []

def search_specific_evaluation_chunks(query: str, evaluation_id: str) -> List[Dict]:
    """
    NEW: Search within chunks of a specific evaluation for detailed analysis
    """
    try:
        from opensearch_client import search_evaluation_chunks
        
        # Search within chunks of specific evaluation
        chunk_results = search_evaluation_chunks(query, evaluation_id=evaluation_id)
        
        logger.info(f"🔍 Found {len(chunk_results)} chunk matches in evaluation {evaluation_id}")
        
        return chunk_results
    
    except Exception as e:
        logger.error(f"❌ Evaluation chunk search failed: {e}")
        return []

def build_system_message(is_analytics: bool, filters: Dict[str, Any], context: str) -> str:
    """
    ENHANCED: Build comprehensive system message for evaluation-grouped structure
    """
    if is_analytics:
        system_msg = """You are MetroAI Analytics, an expert call center analytics specialist for Metro by T-Mobile. You analyze call center evaluation data to provide actionable business insights.

## ENHANCED DATA STRUCTURE (v4.0):
**Document Organization**: Each evaluation is now stored as a single document containing all its chunks, grouped by template_ID-based collections.

**Key Identifiers**:
- **evaluationId**: Primary evaluation reference (used as document ID)
- **template_id**: Template identifier (determines collection/index)
- **template_name**: Human-readable template name (use for display)
- **internalId**: Internal system reference

**Evaluation Document Structure**:
- **Full Content**: Combined evaluation and transcript text
- **Chunk Array**: Individual Q&A pairs and transcript segments within the evaluation
- **Metadata**: Complete call details and agent information

**Organizational Hierarchy**:
- **template_id** → **partner** → **site** → **lob** → **agent**

**Enhanced Analysis Capabilities**:
1. **Evaluation-Level Analysis**: Analyze complete evaluations as units
2. **Template-Based Grouping**: Compare evaluations within template categories
3. **Agent Performance**: Track agent performance across evaluations
4. **Call Pattern Analysis**: Identify patterns in call dispositions and outcomes
5. **Quality Trends**: Monitor evaluation scores and feedback over time

## Your Expertise:
- **Performance Analysis**: Evaluation-level metrics, agent scoring, quality trends
- **Template Analysis**: Compare performance across evaluation templates
- **Call Analysis**: Disposition patterns, duration analysis, outcome tracking
- **Agent Development**: Individual coaching opportunities, skill gaps, strengths
- **Operational Intelligence**: Partner/site comparisons, resource optimization
- **Quality Assurance**: Evaluation consistency, scoring patterns, improvement areas

## Analysis Instructions:
1. **Think in Evaluations**: Each search result represents a complete evaluation
2. **Use Template Context**: Reference template_name for clarity, template_id for precision
3. **Provide Evaluation-Level Insights**: Analyze whole interactions, not just fragments
4. **Reference Specific Evaluations**: Use evaluationId when citing examples
5. **Consider Chunk Details**: Reference specific Q&A or transcript segments when relevant
6. **Track Patterns**: Look for trends across multiple evaluations

## Response Format:
- Start with key evaluation-level findings
- Provide specific evaluation examples with IDs
- Reference template names for context
- End with actionable recommendations based on evaluation patterns"""
    else:
        system_msg = """You are MetroAI, an intelligent assistant for Metro by T-Mobile customer service operations. 

You help with:
- General Metro by T-Mobile service inquiries
- Customer service best practices
- Call center operational guidance
- Policy and procedure questions
- Evaluation and quality assurance guidance

ENHANCED SYSTEM (v4.0): You now work with evaluation-grouped documents where each search result represents a complete call evaluation with all its components.

Provide helpful, accurate, and professional responses based on complete evaluation contexts."""
    
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
    
    # Add search context with evaluation information
    if context:
        system_msg += f"\n\n## Relevant Evaluation Data (Enhanced Structure):\n{context}"
        
        # Add guidance for using the enhanced context
        system_msg += f"\n\n## Context Usage Notes:"
        system_msg += f"\n- Each section above represents a complete evaluation document"
        system_msg += f"\n- Evaluation IDs can be referenced for specific examples"
        system_msg += f"\n- Template names indicate the evaluation framework used"
        system_msg += f"\n- Total chunks show the depth of each evaluation"
        system_msg += f"\n- Use this comprehensive view for thorough analysis"
    
    return system_msg

# ============================================================================
# ENHANCED CHAT ENDPOINTS
# ============================================================================

@chat_router.post("/chat")
async def chat_handler(request: ChatRequest):
    """
    ENHANCED: Chat endpoint with evaluation-grouped document support
    """
    start_time = time.time()
    
    try:
        logger.info(f"💬 Processing enhanced chat request: {request.message[:50]}...")
        
        # Extract comprehensive filters and metadata focus
        filters = request.filters or {}
        is_analytics = request.analytics or False
        metadata_focus = request.metadata_focus or []
        
        # Build search context from evaluation documents
        context, sources = build_search_context(request.message, filters)
        
        # Build enhanced system message for evaluation structure
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
            
            # Count evaluation sources (unique evaluations found)
            evaluation_sources = len(set(source.get('evaluation_id') for source in sources if source.get('evaluation_id')))
            
            logger.info(f"✅ Enhanced chat completed ({total_response_time}ms total)")
            logger.info(f"📊 Context: {evaluation_sources} evaluations, {len(sources)} total sources")
            
            return ChatResponse(
                reply=reply.strip() if reply else "Sorry, I couldn't generate a response.",
                sources=sources[:5] if sources else [],
                filters_applied=filters,
                context_found=bool(context),
                search_results_count=len(sources),
                evaluation_sources=evaluation_sources,  # NEW: evaluation count
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
                evaluation_sources=0,
                response_time_ms=total_response_time,
                ai_agent_status="error",
                retrieval_info={}
            )
            
    except Exception as e:
        total_response_time = round((time.time() - start_time) * 1000)
        logger.error(f"❌ Enhanced chat handler error: {e}")
        
        return ChatResponse(
            reply="Sorry, there was an error processing your request. Please try again.",
            sources=[],
            filters_applied=request.filters or {},
            context_found=False,
            search_results_count=0,
            evaluation_sources=0,
            response_time_ms=total_response_time,
            ai_agent_status="error",
            retrieval_info={}
        )

# ============================================================================
# ENHANCED TESTING AND HEALTH ENDPOINTS
# ============================================================================

@chat_router.get("/test_ai_agent")
async def test_ai_agent():
    """Enhanced AI agent test with evaluation structure information"""
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
        
        # Test enhanced analytics capability with evaluation structure
        analytics_messages = [
            {
                "role": "system", 
                "content": """You are MetroAI Analytics, an expert call center analytics specialist for Metro by T-Mobile.

## ENHANCED DATA STRUCTURE (v4.0):
Document Organization: Each evaluation is stored as a single document containing all chunks, grouped by template_ID.

## Sample Evaluation Data:
Evaluation ID: 14
Template: Ai Corporate SPTR - TEST  
Agent: Rey Mendoza
Call Details: Port Out - Questions/pin/acct #
Duration: 491s, Site: Dasma, Partner: iQor, LOB: WNP
Total Chunks: 6 (4 evaluation Q&A, 2 transcript segments)

Analyze this evaluation structure briefly."""
            },
            {
                "role": "user", 
                "content": "Based on the enhanced evaluation structure, what insights can you provide about this port out interaction and the new document organization?"
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
            "enhanced_analytics_capability": {
                "success": analytics_response.get("success", False),
                "response_time_ms": analytics_response.get("response_time_ms", 0),
                "response_length": len(analytics_response.get("reply", "")),
                "supports_evaluation_structure": True,
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
            "enhancements": {
                "structure_version": "4.0.0",
                "document_type": "evaluation_grouped",
                "collection_strategy": "template_id_based",
                "search_returns": "complete_evaluations"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Enhanced AI agent test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Enhanced AI agent test failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@chat_router.get("/test_enhanced_search")
async def test_enhanced_search():
    """Test the enhanced evaluation-grouped search functionality"""
    try:
        # Test evaluation search
        test_query = "customer service evaluation"
        
        context, sources = build_search_context(test_query, {})
        
        # Test specific evaluation chunk search (if we have evaluation IDs)
        chunk_results = []
        if sources and len(sources) > 0:
            first_eval_id = sources[0].get('evaluation_id')
            if first_eval_id:
                chunk_results = search_specific_evaluation_chunks("question", first_eval_id)
        
        return {
            "status": "success",
            "test_query": test_query,
            "evaluation_search": {
                "context_built": bool(context),
                "context_length": len(context),
                "sources_found": len(sources),
                "unique_evaluations": len(set(source.get('evaluation_id') for source in sources if source.get('evaluation_id'))),
                "templates_found": list(set(source.get('template_name') for source in sources if source.get('template_name')))
            },
            "chunk_search": {
                "tested": len(chunk_results) > 0,
                "chunk_matches": len(chunk_results),
                "evaluation_tested": sources[0].get('evaluation_id') if sources else None
            },
            "structure_info": {
                "version": "4.0.0",
                "document_type": "evaluation_grouped",
                "search_level": "evaluation_documents",
                "chunk_search_available": True
            },
            "sample_source": sources[0] if sources else None,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Enhanced search test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Enhanced search test failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@chat_router.get("/chat/health")
async def chat_health():
    """Enhanced chat health check with evaluation structure support"""
    try:
        # Quick connectivity test
        if AGENT_ACCESS_KEY and AGENT_ENDPOINT:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Respond with 'Enhanced Digital Ocean health check OK'"}
            ]
            
            response = await call_digital_ocean_ai_agent(test_messages)
            
            # Test search functionality
            search_test = False
            try:
                context, sources = build_search_context("test health check", {})
                search_test = True
            except Exception as e:
                logger.warning(f"Search test failed: {e}")
            
            # Prepare base URL for display
            base_url = AGENT_ENDPOINT
            if not base_url.endswith('/'):
                base_url += '/'
            if not base_url.endswith('api/v1/'):
                base_url += 'api/v1/'
            
            return {
                "status": "healthy" if response.get("success") and search_test else "degraded",
                "ai_agent": {
                    "configured": True,
                    "connected": response.get("success", False),
                    "response_time_ms": response.get("response_time_ms", 0),
                    "error": response.get("error"),
                    "retrieval_available": bool(response.get("retrieval_info"))
                },
                "search_functionality": {
                    "evaluation_search": search_test,
                    "structure_version": "4.0.0",
                    "document_type": "evaluation_grouped"
                },
                "configuration": {
                    "endpoint": AGENT_ENDPOINT,
                    "base_url": base_url,
                    "authentication_format": "Digital Ocean OpenAI Client",
                    "include_retrieval_info": DO_INCLUDE_RETRIEVAL_INFO,
                    "timeout": GENAI_TIMEOUT
                },
                "enhancements": {
                    "evaluation_level_search": True,
                    "template_based_collections": True,
                    "chunk_level_search": True,
                    "comprehensive_context": True
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
        logger.error(f"❌ Enhanced chat health check failed: {e}")
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

logger.info("🚀 Enhanced chat handlers module loaded - Evaluation Grouped Structure")
logger.info(f"   Structure Version: 4.0.0")
logger.info(f"   Document Type: evaluation_grouped")
logger.info(f"   Search Level: evaluation_documents")
logger.info(f"   Collection Strategy: template_id_based")
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
    
    logger.info("🦙 Enhanced Digital Ocean AI Agent integration ready")
    logger.info(f"   ✅ Base URL: {base_url}")
    logger.info("   ✅ OpenAI client format")
    logger.info("   ✅ Proper authentication")
    logger.info("   ✅ Retrieval info support")
    logger.info("   ✅ Evaluation-grouped search")
    logger.info("   ✅ Template-based collections")
    logger.info("   ✅ Chunk-level analysis")
else:
    logger.warning("⚠️ Enhanced Digital Ocean AI Agent not fully configured - check environment variables")
    logger.warning("   Required: GENAI_ENDPOINT, GENAI_ACCESS_KEY")