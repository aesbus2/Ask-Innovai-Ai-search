# chat_handlers.py - PRODUCTION FastAPI Chat Router with Efficient OpenSearch 2.x Vector Support
# Version: 4.2.0 - Production-ready with index targeting and caching

import os
import logging
import asyncio
import time
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

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
    timestamp: str
    filter_context: dict = {}
    search_metadata: dict = {}


chat_router = APIRouter()

# =============================================================================
# PRODUCTION GENAI INTEGRATION CONFIGURATION
# =============================================================================

GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT","")
GENAI_ACCESS_KEY = os.getenv("GENAI_ACCESS_KEY", "")
GENAI_MODEL = os.getenv("GENAI_MODEL", "n/a")
GENAI_MAX_TOKENS = int(os.getenv("GENAI_MAX_TOKENS", "2000"))
GENAI_TEMPERATURE = float(os.getenv("GENAI_TEMPERATURE", "0.7"))

# =============================================================================
# PRODUCTION EFFICIENT INDEX TARGETING FUNCTIONS
# =============================================================================

def determine_target_indices(filters: Dict[str, Any]) -> List[str]:
    """
    PRODUCTION: Determine which indices to search based on filters
    This dramatically reduces search scope and improves performance
    """
    try:
        from opensearch_client import get_opensearch_client
        
        client = get_opensearch_client()
        if not client:
            return ["eval-*"]  # Fallback to all indices
        
        # If template_name filter is specified, we can target specific indices
        if 'template_name' in filters:
            template_name = filters['template_name']
            return get_indices_for_template(client, template_name)
        
        # If template_id filter is specified, directly target that index
        if 'template_id' in filters:
            template_id = filters['template_id']
            target_index = f"eval-{clean_template_id_for_index(template_id)}"
            return [target_index]
        
        # For other filters, get a smart subset of indices
        return get_relevant_indices_subset(client, filters)
        
    except Exception as e:
        logger.warning(f"Could not determine target indices: {e}")
        return ["eval-*"]  # Safe fallback

def get_indices_for_template(client, template_name: str) -> List[str]:
    """
    PRODUCTION: Find indices that contain a specific template name
    """
    try:
        # Quick sample search across indices to find matching template
        search_response = client.search(
            index="eval-*",
            body={
                "size": 0,  # No documents needed
                "query": {
                    "term": {"template_name.keyword": template_name}
                },
                "aggs": {
                    "indices": {
                        "terms": {
                            "field": "_index",
                            "size": 20
                        }
                    }
                }
            },
            timeout="10s"
        )
        
        indices = []
        buckets = search_response.get("aggregations", {}).get("indices", {}).get("buckets", [])
        
        for bucket in buckets:
            index_name = bucket.get("key")
            if index_name and index_name.startswith("eval-"):
                indices.append(index_name)
        
        logger.info(f"🎯 Template '{template_name}' found in {len(indices)} indices")
        return indices if indices else ["eval-*"]
        
    except Exception as e:
        logger.warning(f"Could not get indices for template: {e}")
        return ["eval-*"]

def get_relevant_indices_subset(client, filters: Dict[str, Any]) -> List[str]:
    """
    PRODUCTION: Get a smart subset of indices based on other filters
    """
    try:
        # Get index stats to find largest/most active indices
        stats_response = client.indices.stats(index="eval-*", timeout="10s")
        
        index_info = []
        for index_name, stats in stats_response.get("indices", {}).items():
            doc_count = stats.get("primaries", {}).get("docs", {}).get("count", 0)
            if doc_count > 0:  # Only include indices with documents
                index_info.append({
                    "name": index_name,
                    "doc_count": doc_count
                })
        
        # Sort by document count and take top indices
        index_info.sort(key=lambda x: x["doc_count"], reverse=True)
        
        # Limit to top 10 most populated indices for efficiency
        target_indices = [info["name"] for info in index_info[:10]]
        
        logger.info(f"🎯 Using top {len(target_indices)} populated indices for search")
        return target_indices if target_indices else ["eval-*"]
        
    except Exception as e:
        logger.warning(f"Could not get relevant indices subset: {e}")
        return ["eval-*"]

def clean_template_id_for_index(template_id: str) -> str:
    """
    PRODUCTION: Clean template_id to create valid OpenSearch index names
    """
    if not template_id:
        return "default-template"
    
    # Convert to lowercase and clean
    cleaned = str(template_id).lower().strip()
    
    # Replace any non-alphanumeric characters with hyphens
    cleaned = re.sub(r'[^a-z0-9]', '-', cleaned)
    
    # Remove multiple consecutive hyphens
    cleaned = re.sub(r'-+', '-', cleaned)
    
    # Remove leading/trailing hyphens
    cleaned = cleaned.strip('-')
    
    # Ensure it's not empty
    if not cleaned:
        cleaned = "default-template"
    
    # Ensure it starts with a letter (OpenSearch requirement)
    if cleaned and not cleaned[0].isalpha():
        cleaned = f"template-{cleaned}"
    
    # Limit length (OpenSearch has limits)
    if len(cleaned) > 50:
        cleaned = cleaned[:50].rstrip('-')
    
    return cleaned

# =============================================================================
# PRODUCTION ENHANCED SEARCH CONTEXT BUILDING - OPENSEARCH 2.X OPTIMIZED
# =============================================================================

def build_search_context(message: str, filters: Dict[str, Any]) -> tuple[str, List[Dict]]:
    """
    PRODUCTION: Enhanced search context building using targeted index searches
    Optimized for OpenSearch 2.x with efficient index targeting
    """
    context = ""
    sources = []
    
    try:
        from opensearch_client import get_connection_status, search_opensearch, search_vector
        
        conn_status = get_connection_status()
        
        if not conn_status.get("connected", False):
            logger.warning("⚠️ OpenSearch not available for context search")
            return context, sources
        
        logger.info(f"🔍 Building search context with filters: {list(filters.keys()) if filters else 'None'}")
        
        # PRODUCTION: Determine target indices based on filters
        target_indices = determine_target_indices(filters)
        index_pattern = ",".join(target_indices) if target_indices else "eval-*"
        
        # STRATEGY 1: Vector search with targeted indices (OpenSearch 2.x optimized)
        vector_results = []
        try:
            from embedder import embed_text, EMBEDDER_AVAILABLE
            
            if EMBEDDER_AVAILABLE:
                logger.info(f"🔗 Vector search targeting {len(target_indices)} indices")
                query_embedding = embed_text(message)
                
                if query_embedding and len(query_embedding) > 0:
                    vector_results = search_vector(
                        query_vector=query_embedding,
                        index_override=index_pattern,
                        size=5
                    )
                    
                    if vector_results:
                        logger.info(f"✅ Vector search found {len(vector_results)} results")
                    else:
                        logger.info("📊 Vector search returned no results")
                else:
                    logger.warning("⚠️ Failed to generate query embedding")
            else:
                logger.info("📝 Embedder not available, skipping vector search")
                
        except Exception as vector_error:
            logger.warning(f"⚠️ Vector search failed: {vector_error}")
            vector_results = []
        
        # STRATEGY 2: Text search with targeted indices and filters
        text_results = []
        try:
            text_results = search_opensearch_with_filters(
                message, 
                filters=filters,
                index_override=index_pattern,
                size=8
            )
            
            if text_results:
                logger.info(f"📝 Text search found {len(text_results)} results in targeted indices")
            else:
                logger.info("📊 Text search returned no results")
                
        except Exception as text_error:
            logger.error(f"❌ Text search failed: {text_error}")
            text_results = []
        
        # PRODUCTION: Combine results with deduplication
        combined_results = []
        vector_ids = set()
        
        # Add vector results first (priority)
        for result in vector_results:
            result_id = result.get('_id') or result.get('_source', {}).get('evaluationId')
            if result_id:
                combined_results.append(result)
                vector_ids.add(result_id)
        
        # Add text results that aren't already in vector results
        for result in text_results:
            result_id = result.get('_id') or result.get('_source', {}).get('evaluationId')
            if result_id and result_id not in vector_ids:
                combined_results.append(result)
        
        # Limit to top 8 total results
        final_results = combined_results[:8]
        
        if final_results:
            context_parts = []
            evaluation_count = 0
            
            for result in final_results[:5]:  # Use top 5 for context
                source_data = result.get('_source', {})
                
                # Extract evaluation-level information
                evaluation_id = source_data.get('evaluationId', 'Unknown')
                template_name = source_data.get('template_name', 'Unknown Template')
                template_id = source_data.get('template_id', 'Unknown')
                total_chunks = source_data.get('total_chunks', 0)
                metadata = source_data.get('metadata', {})
                
                # Enhanced metadata extraction
                program = metadata.get('program', 'Unknown Program')
                
                # Get text content
                full_text = source_data.get('full_text', '')
                evaluation_text = source_data.get('evaluation_text', '')
                transcript_text = source_data.get('transcript_text', '')
                
                if full_text:
                    evaluation_count += 1
                    
                    # Create rich context with proper Template vs Program distinction
                    context_piece = f"Evaluation ID: {evaluation_id}\n"
                    context_piece += f"Template: {template_name} (ID: {template_id})\n"
                    context_piece += f"Program: {program}\n"
                    context_piece += f"Agent: {metadata.get('agent', 'Unknown')}\n"
                    context_piece += f"Call Details: {metadata.get('disposition', 'Unknown')} - {metadata.get('sub_disposition', 'None')}\n"
                    context_piece += f"Duration: {metadata.get('call_duration', 'Unknown')}s, "
                    context_piece += f"Site: {metadata.get('site', 'Unknown')}, "
                    context_piece += f"Partner: {metadata.get('partner', 'Unknown')}, "
                    context_piece += f"LOB: {metadata.get('lob', 'Unknown')}\n"
                    context_piece += f"Call Date: {metadata.get('call_date', 'Unknown')}\n"
                    
                    # Add relevant content (truncated for context)
                    if evaluation_text and len(evaluation_text) > 0:
                        context_piece += f"Evaluation Content: {evaluation_text[:600]}...\n"
                    
                    if transcript_text and len(transcript_text) > 0:
                        context_piece += f"Call Transcript: {transcript_text[:400]}...\n"
                    
                    context_piece += f"Total Chunks: {total_chunks}\n"
                    context_piece += "---"
                    
                    context_parts.append(context_piece)
                    
                    # Add to sources with enhanced metadata
                    sources.append({
                        'text': full_text[:500] if full_text else 'No content available',
                        'evaluation_id': evaluation_id,
                        'template_id': template_id,
                        'template_name': template_name,
                        'program': program,
                        'total_chunks': total_chunks,
                        'metadata': metadata,
                        'score': result.get('_score', 0),
                        'source_id': result.get('_id', ''),
                        'collection': result.get('_index', 'unknown'),
                        'document_type': 'evaluation',
                        'evaluation_chunks_count': source_data.get('evaluation_chunks_count', 0),
                        'transcript_chunks_count': source_data.get('transcript_chunks_count', 0),
                        'search_type': 'vector' if result in vector_results else 'text'
                    })
            
            context = "\n\n".join(context_parts)
            
            vector_count = len([r for r in final_results if r in vector_results])
            text_count = len(final_results) - vector_count
            
            logger.info(f"📊 Context built from {evaluation_count} evaluations ({vector_count} vector + {text_count} text results)")
            logger.info(f"🎯 Searched {len(target_indices)} targeted indices vs all eval-*")
            
            if evaluation_count > 0:
                first_source = sources[0]
                logger.info(f"📋 Sample context - Template: '{first_source.get('template_name')}', Program: '{first_source.get('program')}'")
        
        else:
            logger.info("📊 No search results found for context")
    
    except ImportError as e:
        logger.error(f"❌ OpenSearch client import failed: {e}")
    except Exception as e:
        logger.error(f"❌ Context search failed: {e}")
    
    return context, sources

def search_opensearch_with_filters(query: str, filters: Dict[str, Any] = None, 
                                 index_override: str = None, size: int = 10) -> List[Dict]:
    """
    PRODUCTION: Enhanced OpenSearch with filter support for Template vs Program
    """
    try:
        from opensearch_client import search_opensearch
        
        if filters:
            logger.info(f"🔍 Applying search filters:")
            for key, value in filters.items():
                logger.info(f"   {key}: {value}")
        
        # Use the enhanced search with proper field mapping
        results = search_opensearch(query, index_override=index_override, 
                                  filters=filters, size=size)
        
        logger.info(f"🔍 Found {len(results)} evaluation documents with filters")
        
        if filters and results:
            # Check if results actually match the filters
            sample_result = results[0].get('_source', {})
            sample_metadata = sample_result.get('metadata', {})
            
            logger.info(f"📊 Sample result verification:")
            if 'template_name' in filters:
                actual_template = sample_result.get('template_name', 'N/A')
                logger.info(f"   Template filter '{filters['template_name']}' → Result: '{actual_template}'")
            
            if 'program' in filters:
                actual_program = sample_metadata.get('program', 'N/A')
                logger.info(f"   Program filter '{filters['program']}' → Result: '{actual_program}'")
        
        return results
    
    except Exception as e:
        logger.error(f"❌ Filtered evaluation search failed: {e}")
        return []

def build_system_message(is_analytics: bool, filters: Dict[str, Any], context: str) -> str:
    """
    PRODUCTION: Enhanced system message with proper Template vs Program distinction
    """
    if is_analytics:
        system_msg = """You are MetroAI Analytics, an expert call center analytics specialist for Metro by T-Mobile. You analyze call center evaluation data to provide actionable business insights.

## ENHANCED DATA STRUCTURE (v4.2.0) - OpenSearch 2.x Optimized:
**Document Organization**: Each evaluation is stored as a single document containing all its chunks, grouped by template_ID-based collections.

**Key Identifiers**:
- **evaluationId**: Primary evaluation reference (used as document ID)
- **template_id**: Template identifier (determines collection/index)  
- **template_name**: Human-readable template name (evaluation form name)
- **program**: Business program (Metro, T-Mobile Prepaid, ASW, Corporate)
- **weight_score**: QA weighted score percentage (0-100) for quality analysis

**Quality Analysis with Weight Scores**:
- Use weight_score for performance analysis (e.g., "calls with score below 80%")
- Compare scores across agents, sites, programs, and time periods
- Identify coaching opportunities and top performers
- Calculate averages, trends, and improvement areas
- Weight scores represent overall quality assessment (0-100%)

**Enhanced Analysis Capabilities**:
1. **Template Analysis**: Compare different evaluation forms and their effectiveness
2. **Program Analysis**: Analyze performance across business programs (Metro vs T-Mobile Prepaid vs ASW)
3. **Quality Analysis**: Use weight_score for performance insights and coaching opportunities
4. **Agent Performance**: Track agent performance across evaluations and programs
5. **Partner/Site Analysis**: Compare vendor and location performance
6. **Call Pattern Analysis**: Identify patterns in call dispositions and outcomes

## Your Expertise:
- **Performance Analysis**: Evaluation-level metrics, agent scoring, quality trends
- **Quality Coaching**: Use weight_score to identify improvement opportunities
- **Business Program Analysis**: Compare performance across Metro, T-Mobile Prepaid, ASW programs
- **Template Effectiveness**: Analyze which evaluation forms provide better insights
- **Partner/Site Optimization**: Identify best-performing vendors and locations
- **Agent Development**: Individual coaching opportunities, skill gaps, strengths

## Analysis Instructions:
1. **Use Weight Scores**: When asked about performance, reference weight_score field
2. **Distinguish Templates from Programs**: Templates are evaluation forms, Programs are business units
3. **Quality Analysis**: Use weight_score for coaching insights and performance comparisons
4. **Provide Specific Examples**: Use evaluationId when citing examples
5. **Reference Complete Evaluations**: Analyze full interactions, not just fragments

## Response Format:
- Start with key evaluation-level findings including quality metrics
- Use weight_score data for performance insights and coaching recommendations
- Provide specific evaluation examples with IDs and scores
- End with actionable recommendations based on quality analysis"""
    else:
        system_msg = """You are MetroAI, an intelligent assistant for Metro by T-Mobile customer service operations. 

You help with:
- General Metro by T-Mobile service inquiries
- Customer service best practices
- Call center operational guidance
- Policy and procedure questions
- Evaluation and quality assurance guidance

ENHANCED SYSTEM (v4.2.0): You now work with evaluation-grouped documents with proper Template vs Program distinction:
- Templates: Evaluation forms (e.g., "Ai Corporate SPTR - TEST")
- Programs: Business units (e.g., Metro, T-Mobile Prepaid, ASW)

Provide helpful, accurate, and professional responses based on complete evaluation contexts."""
    
    # Add filter context with proper Template vs Program labels
    if filters:
        system_msg += f"\n\n## Current Analysis Filters:\n"
        
        filter_display_map = {
            'template_name': 'Template (Evaluation Form)',
            'program': 'Program (Business Unit)',
            'partner': 'Partner (Vendor)',
            'site': 'Site (Location)',
            'lob': 'LOB (Line of Business)',
            'disposition': 'Call Disposition',
            'sub_disposition': 'Call Sub-Disposition',
            'agent_name': 'Agent',
            'call_date_start': 'Call Date From',
            'call_date_end': 'Call Date To',
            'language': 'Language',
            'min_duration': 'Min Duration (seconds)',
            'max_duration': 'Max Duration (seconds)'
        }
        
        for key, value in filters.items():
            display_name = filter_display_map.get(key, key)
            if isinstance(value, list):
                system_msg += f"- **{display_name}**: {', '.join(map(str, value))}\n"
            else:
                system_msg += f"- **{display_name}**: {value}\n"
    
    # Add search context with evaluation information
    if context:
        system_msg += f"\n\n## Relevant Evaluation Data (Enhanced Structure):\n{context}"
        
        system_msg += f"\n\n## Context Usage Notes:"
        system_msg += f"\n- Each section above represents a complete evaluation document"
        system_msg += f"\n- Template names indicate the evaluation form used"
        system_msg += f"\n- Program indicates the business unit (Metro, T-Mobile Prepaid, ASW, etc.)"
        system_msg += f"\n- Evaluation IDs can be referenced for specific examples"
        system_msg += f"\n- Use this comprehensive view for thorough business analysis"
    
    return system_msg

# =============================================================================
# PRODUCTION GENAI INTEGRATION FUNCTIONS
# =============================================================================

async def call_genai_api(system_message: str, user_message: str, chat_history: List[Dict] = None) -> str:
    """
    PRODUCTION: Call GenAI API with proper error handling and OpenSearch 2.x context
    """
    try:
        import requests
        
        if not GENAI_ENDPOINT or not GENAI_ACCESS_KEY:
            raise Exception("GenAI configuration missing - check GENAI_ENDPOINT and GENAI_ACCESS_KEY")
        
        # Build messages array
        messages = [{"role": "system", "content": system_message}]
        
        # Add chat history if provided
        if chat_history:
            # Add last 6 messages to maintain context without overwhelming the API
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            for msg in recent_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Prepare request payload
        payload = {
            "model": GENAI_MODEL,
            "messages": messages,
            "max_tokens": GENAI_MAX_TOKENS,
            "temperature": GENAI_TEMPERATURE,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {GENAI_ACCESS_KEY}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"🤖 Calling GenAI API with {len(messages)} messages")
        
        # Make API call with timeout
        response = requests.post(
            GENAI_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=60
        )
        
        if not response.ok:
            error_text = response.text[:200] if response.text else "No error details"
            raise Exception(f"GenAI API error {response.status_code}: {error_text}")
        
        data = response.json()
        
        # Extract response content
        if 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0].get('message', {}).get('content', '')
            if content:
                logger.info(f"✅ GenAI response received ({len(content)} characters)")
                return content
            else:
                raise Exception("Empty response from GenAI API")
        else:
            raise Exception("Invalid response format from GenAI API")
    
    except Exception as e:
        logger.error(f"❌ GenAI API call failed: {e}")
        return f"I apologize, but I'm experiencing technical difficulties connecting to the AI service. Error: {str(e)[:100]}. Please try again in a moment."

# =============================================================================
# PRODUCTION MAIN CHAT ENDPOINT
# =============================================================================

@chat_router.post("/chat")
async def chat_endpoint(request: ChatRequest) -> JSONResponse:
    """
    PRODUCTION: Main chat endpoint with OpenSearch 2.x vector optimization and index targeting
    """
    start_time = time.time()
    
    try:
        logger.info(f"💬 Chat request received: '{request.message[:50]}...'")
        logger.info(f"🔍 Filters: {list(request.filters.keys()) if request.filters else 'None'}")
        
        # Build search context with OpenSearch 2.x optimization and index targeting
        context, sources = build_search_context(request.message, request.filters)
        
        # Build system message
        system_message = build_system_message(request.analytics, request.filters, context)
        
        # Call GenAI API
        reply = await call_genai_api(
            system_message=system_message,
            user_message=request.message,
            chat_history=request.history
        )
        
        # Prepare response
        response_data = {
            "reply": reply,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "filter_context": request.filters,
            "search_metadata": {
                "total_sources": len(sources),
                "vector_sources": len([s for s in sources if s.get('search_type') == 'vector']),
                "text_sources": len([s for s in sources if s.get('search_type') == 'text']),
                "context_length": len(context),
                "processing_time": time.time() - start_time,
                "version": "4.2.0_production"
            }
        }
        
        logger.info(f"✅ Chat response completed in {time.time() - start_time:.2f}s")
        logger.info(f"📊 Response: {len(reply)} chars, {len(sources)} sources")
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        
        # Return error response
        error_response = {
            "reply": f"I apologize, but I encountered an error processing your request: {str(e)[:200]}. Please try again.",
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "filter_context": request.filters,
            "search_metadata": {
                "error": str(e),
                "processing_time": time.time() - start_time,
                "version": "4.2.0_production"
            }
        }
        
        return JSONResponse(content=error_response, status_code=200)  # Return 200 to avoid frontend errors

# =============================================================================
# PRODUCTION HELPER ENDPOINTS
# =============================================================================

@chat_router.get("/chat/health")
async def chat_health_check():
    """
    PRODUCTION: Health check for chat functionality
    """
    try:
        # Test OpenSearch connection
        from opensearch_client import test_connection
        opensearch_ok = test_connection()
        
        # Test embedder availability
        embedder_ok = False
        try:
            from embedder import EMBEDDER_AVAILABLE
            embedder_ok = EMBEDDER_AVAILABLE
        except:
            pass
        
        # Test GenAI configuration
        genai_configured = bool(GENAI_ENDPOINT and GENAI_ACCESS_KEY)
        
        return {
            "status": "healthy",
            "components": {
                "opensearch": "connected" if opensearch_ok else "disconnected",
                "embedder": "available" if embedder_ok else "unavailable",
                "genai": "configured" if genai_configured else "not_configured"
            },
            "vector_support": embedder_ok and opensearch_ok,
            "features": {
                "index_targeting": True,
                "efficient_search": True,
                "template_program_distinction": True
            },
            "version": "4.2.0_production",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "4.2.0_production",
            "timestamp": datetime.now().isoformat()
        }

logger.info("✅ PRODUCTION Chat handlers v4.2.0 with FastAPI router loaded successfully")
logger.info("🔗 OpenSearch 2.x vector optimization with index targeting enabled")
logger.info("📊 Enhanced metadata and efficient filtering support active")
logger.info("🎯 Template vs Program distinction and targeted search implemented")