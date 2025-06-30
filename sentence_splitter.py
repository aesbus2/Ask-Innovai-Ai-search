import re
from typing import List, Dict

def split_into_chunks(text: str, max_chars: int = 1100, overlap: int = 150, min_chunk_size: int = 100) -> List[Dict]:
    """
    Enhanced text chunking that respects sentence boundaries for better embedding quality.
    
    Args:
        text (str): The text to split
        max_chars (int): Maximum characters per chunk
        overlap (int): Number of characters to overlap between chunks
        min_chunk_size (int): Minimum chunk size to avoid tiny fragments
        
    Returns:
        list: List of dictionaries with 'text', 'offset', 'length', and 'chunk_index' keys
    """
    if not text or not text.strip():
        return []
    
    # Clean up the text first
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        # Calculate the ideal end position
        ideal_end = min(start + max_chars, len(text))
        
        # If we're at the end of the text, just take what's left
        if ideal_end >= len(text):
            remaining_text = text[start:].strip()
            if remaining_text and len(remaining_text) >= min_chunk_size:
                chunks.append({
                    "text": remaining_text,
                    "offset": start,
                    "length": len(remaining_text),
                    "chunk_index": chunk_index
                })
            break
        
        # Find the best break point using smart boundary detection
        chunk_end = find_best_break_point(text, start, ideal_end, max_chars)
        
        # Extract the chunk
        chunk_text = text[start:chunk_end].strip()
        
        # Only add non-empty chunks that meet minimum size
        if chunk_text and len(chunk_text) >= min_chunk_size:
            chunks.append({
                "text": chunk_text,
                "offset": start,
                "length": len(chunk_text),
                "chunk_index": chunk_index,
                "break_type": get_break_type(text, chunk_end)  # For debugging
            })
            chunk_index += 1
        
        # Calculate next start position with overlap
        # But make sure we don't create duplicate content
        next_start = max(start + 1, chunk_end - overlap)
        
        # Avoid infinite loops
        if next_start <= start:
            next_start = start + max(1, max_chars // 2)
        
        start = next_start
    
    return chunks

def find_best_break_point(text: str, start: int, ideal_end: int, max_chars: int) -> int:
    """
    Find the best place to break text, prioritizing sentence boundaries.
    
    Args:
        text: Full text
        start: Start position of current chunk
        ideal_end: Ideal end position 
        max_chars: Maximum characters allowed
        
    Returns:
        int: Best break position
    """
    # If ideal_end is beyond text, return text length
    if ideal_end >= len(text):
        return len(text)
    
    # Define our search window - look back from ideal end to find good break points
    search_start = max(start, ideal_end - max_chars // 3)  # Don't look too far back
    search_text = text[search_start:ideal_end + 200]  # Look a bit ahead too
    
    # Strategy 1: Look for paragraph breaks (double newlines)
    paragraph_breaks = []
    for match in re.finditer(r'\n\s*\n', search_text):
        pos = search_start + match.end()
        if start + 100 <= pos <= ideal_end:  # Must be reasonable distance from start
            paragraph_breaks.append(pos)
    
    if paragraph_breaks:
        return max(paragraph_breaks)  # Take the latest paragraph break
    
    # Strategy 2: Look for sentence endings
    sentence_endings = []
    
    # Enhanced sentence detection patterns
    sentence_patterns = [
        r'[.!?]+\s+[A-Z]',  # Period/!/?  followed by space and capital letter
        r'[.!?]+\s*\n',     # Period/!/? followed by newline
        r'[.!?]+\s*$',      # Period/!/? at end of text
        r'[.!?]+\s+["\'"]', # Period/!/? followed by quote
    ]
    
    for pattern in sentence_patterns:
        for match in re.finditer(pattern, search_text):
            # Position right after the punctuation, before the next sentence
            pos = search_start + match.start() + len(match.group().rstrip())
            pos = search_start + match.end() - len(match.group().split()[-1]) if match.group().split() else search_start + match.end()
            
            # Adjust to end right after punctuation
            punctuation_end = search_start + match.start()
            while punctuation_end < len(text) and text[punctuation_end] in '.!?':
                punctuation_end += 1
            
            if start + 100 <= punctuation_end <= ideal_end:
                sentence_endings.append(punctuation_end)
    
    if sentence_endings:
        return max(sentence_endings)
    
    # Strategy 3: Look for other natural break points
    natural_breaks = []
    
    # Look for clause breaks (commas, semicolons, colons)
    clause_patterns = [
        r'[;:]\s+',
        r',\s+(?:and|but|or|however|therefore|meanwhile|furthermore)\s+',
        r'\.\s*\n',  # Period followed by newline (list items, etc.)
    ]
    
    for pattern in clause_patterns:
        for match in re.finditer(pattern, search_text):
            pos = search_start + match.end()
            if start + 50 <= pos <= ideal_end - 50:  # Not too close to start or end
                natural_breaks.append(pos)
    
    if natural_breaks:
        return max(natural_breaks)
    
    # Strategy 4: Look for word boundaries
    word_breaks = []
    for match in re.finditer(r'\s+', search_text):
        pos = search_start + match.start()
        if start + 50 <= pos <= ideal_end:
            word_breaks.append(pos)
    
    if word_breaks:
        # Prefer word breaks closer to ideal_end but not too close to avoid tiny chunks
        suitable_breaks = [pos for pos in word_breaks if pos <= ideal_end - 20]
        if suitable_breaks:
            return max(suitable_breaks)
        else:
            return max(word_breaks)
    
    # Strategy 5: Last resort - use ideal_end (character boundary)
    return ideal_end

def get_break_type(text: str, break_pos: int) -> str:
    """
    Determine what type of boundary we broke on (for debugging).
    """
    if break_pos >= len(text):
        return "end_of_text"
    
    # Look at a small window around the break
    window_start = max(0, break_pos - 5)
    window_end = min(len(text), break_pos + 5)
    window = text[window_start:window_end]
    
    if '\n\n' in window:
        return "paragraph"
    elif any(punct in window for punct in '.!?'):
        return "sentence"
    elif any(punct in window for punct in ';:'):
        return "clause"
    elif re.search(r'\s', window):
        return "word"
    else:
        return "character"

def preview_chunks(text: str, max_chars: int = 1200, overlap: int = 100) -> None:
    """
    Preview how text will be chunked (useful for debugging).
    """
    chunks = split_into_chunks(text, max_chars, overlap)
    
    print(f"Text length: {len(text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1} (offset {chunk['offset']}, length {chunk['length']}):")
        print(f"Break type: {chunk.get('break_type', 'unknown')}")
        
        # Show first and last few words to see boundaries
        words = chunk['text'].split()
        if len(words) <= 10:
            preview = chunk['text']
        else:
            preview = ' '.join(words[:5]) + ' ... ' + ' '.join(words[-5:])
        
        print(f"Preview: {preview}")
        print("-" * 40)

# Example usage and testing
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    Customer Account Protection is a comprehensive security system designed to protect Metro customers. 
    This system includes multiple layers of verification and authentication. When customers call to make 
    account changes, they must verify their identity through several methods.
    
    The primary verification methods include account PIN verification and security question authentication. 
    Account PINs are 4-digit codes that customers set up when they activate their service. Security questions 
    are personal questions that only the account holder should know the answers to.
    
    If a customer fails verification, additional steps may be required. These can include sending verification 
    codes to the phone number on the account or requiring the customer to visit a Metro store with photo ID. 
    This multi-layered approach helps prevent unauthorized access to customer accounts.
    """
    
    print("Enhanced Sentence-Aware Chunking Demo")
    print("=" * 50)
    
    # Test with different chunk sizes
    for chunk_size in [300, 500, 800]:
        print(f"\nChunk size: {chunk_size} characters")
        chunks = split_into_chunks(sample_text, max_chars=chunk_size, overlap=50)
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} ({chunk['length']} chars, break: {chunk.get('break_type', 'unknown')}):")
            print(f"  Starts: '{chunk['text'][:50]}...'")
            print(f"  Ends: '...{chunk['text'][-50:]}'")
            print()