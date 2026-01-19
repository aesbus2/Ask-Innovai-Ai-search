# This is the KEY SECTION that needs to be updated in your opensearch_client.py
# Replace lines approximately 1411-1445 with this fixed version:

        # Split query into words for validation
        search_words = clean_query.lower().split()
        
        # FIXED: Detect analytical queries ONCE before the loop
        analytical_terms = ["analysis", "analyze", "report", "summary", "summarize", "review",
                           "performance", "quality", "metrics", "statistics", "trends", "weekly",
                           "monthly", "quarterly", "overview", "assessment", "evaluation"]
        is_analytical_query = any(term in clean_query.lower() for term in analytical_terms)
        
        # CRITICAL FIX: Process ALL hits for analytical queries, not just display_size
        hits_to_process = all_hits if is_analytical_query else all_hits[:display_size]
        
        logger.info(f"🔍 Processing {len(hits_to_process)} hits (analytical_query={is_analytical_query}, display_size={display_size})")
        
        for hit in hits_to_process:
            source = hit.get("_source", {})
            transcript_content = source.get("transcript_text", "")
            
            if not transcript_content or len(transcript_content.strip()) < 10:
                continue
            
            content_lower = transcript_content.lower()
            
            # FIXED: Skip validation for analytical queries (use variable from outside loop)
            if not is_analytical_query:
                if is_quoted_phrase:
                    # For quoted phrases, require exact phrase
                    if clean_query.lower() not in content_lower:
                        continue
                else:
                    # For regular queries, require at least one search word
                    words_found = [word for word in search_words if word in content_lower]
                    if not words_found:
                        continue

            # This is a valid match - rest of your existing code continues here...
