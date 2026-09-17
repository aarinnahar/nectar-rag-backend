import tiktoken
from typing import List, Dict, Any

def _extract_text(chunk: Any) -> str:
    """Safely extracts text from strings, dicts, or framework objects."""
    if isinstance(chunk, str): return chunk
    if hasattr(chunk, "page_content"): return str(chunk.page_content)
    if hasattr(chunk, "text"): return str(chunk.text)
    if isinstance(chunk, dict): return str(chunk.get("page_content", chunk.get("text", "")))
    return str(chunk)

def compute_avg_tokens_per_query(
    strategy_queries_data: List[Dict[str, Any]], 
    model_name: str = "gpt-4o",
    template_overhead_tokens: int = 35 
) -> float:
    """
    Calculates the average number of tokens processed per query using batched encoding.
    
    :param strategy_queries_data: List of dicts containing 'query_text' and 'retrieved_chunks'.
    :param model_name: Model name for tiktoken (e.g., 'gpt-4o' uses 'o200k_base').
    :param template_overhead_tokens: Estimated tokens used by system prompts and formatting.
    """
    if not strategy_queries_data:
        return 0.0

    # 1. Initialize and cache the encoder outside the loop
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # 2. Prepare text safely for batching
    texts_to_encode = []
    
    for item in strategy_queries_data:
        query_text = str(item.get("query_text", "")).strip()
        
        raw_chunks = item.get("retrieved_chunks", [])
        if not isinstance(raw_chunks, list):
            raw_chunks = [raw_chunks]
            
        clean_chunks = [_extract_text(c).strip() for c in raw_chunks if c]
        
        # Combine query and chunks
        combined_text = f"{query_text}\n\n" + "\n\n".join(clean_chunks)
        texts_to_encode.append(combined_text)

    if not texts_to_encode:
        return 0.0

    # 3. BATCH ENCODE: Rust-optimized, multi-threaded encoding that ignores special tokens
    batched_tokens = encoding.encode_ordinary_batch(texts_to_encode)
    
    # Calculate totals
    total_payload_tokens = sum(len(tokens) for tokens in batched_tokens)
    total_overhead = len(strategy_queries_data) * template_overhead_tokens
    
    total_tokens = total_payload_tokens + total_overhead

    return round(float(total_tokens / len(strategy_queries_data)), 2)