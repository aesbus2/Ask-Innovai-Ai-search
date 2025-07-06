# FIXED: Chat handlers search context with proper Template vs Program field mapping
# Chat_handlers.py version 2.1.0
# Replace the build_search_context function in chat_handlers.py

import logging
from typing import Dict, Any, List
logger = logging.getLogger(__name__)

def build_search_context(message: str, filters: Dict[str, Any]) -> tuple[str, List[Dict]]:
    """
    ENHANCED: Build context from OpenSearch with FIXED Template vs Program mapping
    """
    context = ""
    sources = []
    
    try:
        from opensearch_client import get_connection_status, search_opensearch, search_evaluation_chunks
        
        conn_status = get_connection_status()
        
        if not conn_status.get("connected", False):
            logger.warning("⚠️ OpenSearch not available for context search")
            return context, sources
        
        # FIXED: Log the incoming filters for debugging
        logger.info(f"🔍 Building search context with filters: {list(filters.keys()) if filters else 'None'}")
        
        # ENHANCED: Search evaluation documents with FIXED filter mapping
        search_results = search_opensearch_with_filters(
            message, 
            filters=filters,
            index_override=None,
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
                template_id = source_data.get('template_id', 'Unknown')
                total_chunks = source_data.get('total_chunks', 0)
                metadata = source_data.get('metadata', {})
                
                # FIXED: Extract Program field properly
                program = metadata.get('program', 'Unknown Program')
                
                # Get text content
                full_text = source_data.get('full_text', '')
                evaluation_text = source_data.get('evaluation_text', '')
                transcript_text = source_data.get('transcript_text', '')
                
                if full_text:
                    evaluation_count += 1
                    
                    # ENHANCED: Create rich context with proper Template vs Program distinction
                    context_piece = f"Evaluation ID: {evaluation_id}\n"
                    context_piece += f"Template: {template_name} (ID: {template_id})\n"
                    context_piece += f"Program: {program}\n"  # FIXED: Show Program separately
                    context_piece += f"Agent: {metadata.get('agent', 'Unknown')}\n"
                    context_piece += f"Call Details: {metadata.get('disposition', 'Unknown')} - {metadata.get('sub_disposition', 'None')}\n"
                    context_piece += f"Duration: {metadata.get('call_duration', 'Unknown')}s, "
                    context_piece += f"Site: {metadata.get('site', 'Unknown')}, "
                    context_piece += f"Partner: {metadata.get('partner', 'Unknown')}, "
                    context_piece += f"LOB: {metadata.get('lob', 'Unknown')}\n"
                    context_piece += f"Call Date: {metadata.get('call_date', 'Unknown')}\n"
                    
                    # Add relevant content
                    if evaluation_text and len(evaluation_text) > 0:
                        context_piece += f"Evaluation Content: {evaluation_text[:400]}...\n"
                    
                    if transcript_text and len(transcript_text) > 0:
                        context_piece += f"Call Transcript: {transcript_text[:300]}...\n"
                    
                    context_piece += f"Total Chunks: {total_chunks}\n"
                    context_piece += "---"
                    
                    context_parts.append(context_piece)
                    
                    # ENHANCED: Add to sources with Template vs Program distinction
                    sources.append({
                        'text': full_text[:500] if full_text else 'No content available',
                        'evaluation_id': evaluation_id,
                        'template_id': template_id,
                        'template_name': template_name,
                        'program': program,  # FIXED: Include program field
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
            
            # FIXED: Log sample context for debugging
            if evaluation_count > 0:
                first_source = sources[0]
                logger.info(f"📋 Sample context - Template: '{first_source.get('template_name')}', Program: '{first_source.get('program')}'")
        
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
    ENHANCED: OpenSearch with FIXED filter support for Template vs Program
    """
    try:
        from opensearch_client import search_opensearch
        
        # FIXED: Log filter details for debugging
        if filters:
            logger.info(f"🔍 Applying search filters:")
            for key, value in filters.items():
                logger.info(f"   {key}: {value}")
        
        # Use the enhanced search with proper field mapping
        results = search_opensearch(query, index_override=index_override, 
                                  filters=filters, size=size)
        
        logger.info(f"🔍 Found {len(results)} evaluation documents with filters")
        
        # FIXED: Log filter effectiveness
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
    ENHANCED: Build system message with FIXED Template vs Program distinction
    """
    if is_analytics:
        system_msg = """You are MetroAI Analytics, an expert call center analytics specialist for Metro by T-Mobile. You analyze call center evaluation data to provide actionable business insights.

## ENHANCED DATA STRUCTURE (v4.0+) - FIXED HIERARCHY:
**Document Organization**: Each evaluation is stored as a single document containing all its chunks, grouped by template_ID-based collections.

**Key Identifiers**:
- **evaluationId**: Primary evaluation reference (used as document ID)
- **template_id**: Template identifier (determines collection/index)  
- **template_name**: Human-readable template name (evaluation form name)
- **program**: Business program (Metro, T-Mobile Prepaid, ASW, Corporate)

**FIXED Organizational Hierarchy**:
- **Template** (evaluation form) → **Program** (business unit) → **Partner** (vendor) → **Site** (location) → **LOB** (line of business)

**Evaluation Document Structure**:
- **Full Content**: Combined evaluation and transcript text
- **Chunk Array**: Individual Q&A pairs and transcript segments within the evaluation
- **Enhanced Metadata**: Complete call details, agent information, and business hierarchy

**Enhanced Analysis Capabilities**:
1. **Template Analysis**: Compare different evaluation forms and their effectiveness
2. **Program Analysis**: Analyze performance across business programs (Metro vs T-Mobile Prepaid vs ASW)
3. **Agent Performance**: Track agent performance across evaluations and programs
4. **Partner/Site Analysis**: Compare vendor and location performance
5. **Call Pattern Analysis**: Identify patterns in call dispositions and outcomes
6. **Quality Trends**: Monitor evaluation scores and feedback over time

## Your Expertise:
- **Performance Analysis**: Evaluation-level metrics, agent scoring, quality trends
- **Business Program Analysis**: Compare performance across Metro, T-Mobile Prepaid, ASW programs
- **Template Effectiveness**: Analyze which evaluation forms provide better insights
- **Partner/Site Optimization**: Identify best-performing vendors and locations
- **Agent Development**: Individual coaching opportunities, skill gaps, strengths
- **Operational Intelligence**: Resource optimization, process improvements

## Analysis Instructions:
1. **Distinguish Templates from Programs**: Templates are evaluation forms, Programs are business units
2. **Use Template Context**: Reference template_name for evaluation form clarity
3. **Analyze by Program**: Compare Metro vs T-Mobile Prepaid vs ASW performance
4. **Provide Evaluation-Level Insights**: Analyze complete interactions, not just fragments
5. **Reference Specific Evaluations**: Use evaluationId when citing examples
6. **Consider Business Hierarchy**: Template → Program → Partner → Site → LOB

## Response Format:
- Start with key evaluation-level findings across business programs
- Provide specific evaluation examples with IDs and programs
- Reference template names and programs for context
- End with actionable recommendations based on business program performance"""
    else:
        system_msg = """You are MetroAI, an intelligent assistant for Metro by T-Mobile customer service operations. 

You help with:
- General Metro by T-Mobile service inquiries
- Customer service best practices
- Call center operational guidance
- Policy and procedure questions
- Evaluation and quality assurance guidance

ENHANCED SYSTEM (v4.0+): You now work with evaluation-grouped documents with proper Template vs Program distinction:
- Templates: Evaluation forms (e.g., "Ai Corporate SPTR - TEST")
- Programs: Business units (e.g., Metro, T-Mobile Prepaid, ASW)

Provide helpful, accurate, and professional responses based on complete evaluation contexts."""
    
    # FIXED: Add filter context with proper Template vs Program labels
    if filters:
        system_msg += f"\n\n## Current Analysis Filters:\n"
        
        # FIXED: Proper filter display with Template vs Program distinction
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