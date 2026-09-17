import tiktoken
import numpy as np
from typing import List, Dict, Any, Optional

# Pre-load tokenizer encoding globally to avoid re-instantiation overhead
try:
    TOKENIZER_ENCODING = tiktoken.get_encoding("o200k_base")
except Exception:
    TOKENIZER_ENCODING = tiktoken.get_encoding("cl100k_base")

# Standard Model Tier Pricing Definitions per 1,000,000 Tokens (USD)
DEFAULT_PRICING_TIERS = {
    "premium": {"input_per_1m": 5.00, "output_per_1m": 15.00},  # e.g., GPT-4o
    "fast": {"input_per_1m": 0.15, "output_per_1m": 0.60},      # e.g., GPT-4o-mini
}

def compute_projected_costs(
    strategy_queries_data: List[Dict[str, Any]],
    pricing_tiers: Optional[Dict[str, Dict[str, float]]] = None,
    system_prompt_overhead: int = 50
) -> Dict[str, float]:
    """
    Calculates deterministic token usage and projected API costs per 1,000 queries.
    
    :param strategy_queries_data: Query evaluations containing 'question', 'answer', and 'retrieved_chunks'.
    :param pricing_tiers: Dictionary containing pricing rates per 1M tokens for custom models.
    :param system_prompt_overhead: Fixed token count reserved for RAG system instructions.
    """
    if not strategy_queries_data:
        return {
            "avg_input_tokens_per_query": 0.0,
            "avg_output_tokens_per_query": 0.0,
            "premium_cost_per_1k_usd": 0.0,
            "fast_cost_per_1k_usd": 0.0
        }

    tiers = pricing_tiers or DEFAULT_PRICING_TIERS
    input_token_counts = []
    output_token_counts = []

    for item in strategy_queries_data:
        query_text = item.get("question", "")
        raw_chunks = item.get("retrieved_chunks", [])
        
        # Extract text content from string or dictionary chunks
        chunk_texts = [
            c.get("page_content", c.get("text", "")) if isinstance(c, dict) else str(c)
            for c in raw_chunks
        ]
        
        # Calculate Input Tokens (Query + Context Chunks + Fixed Overhead)
        combined_input_text = f"{query_text} " + " ".join(chunk_texts)
        input_tokens = len(TOKENIZER_ENCODING.encode(combined_input_text)) + system_prompt_overhead
        input_token_counts.append(input_tokens)

        # Calculate Output Tokens (Golden Answer Proxy)
        golden_answer = item.get("answer", "")
        output_tokens = len(TOKENIZER_ENCODING.encode(golden_answer))
        output_token_counts.append(output_tokens)

    avg_input_tokens = float(np.mean(input_token_counts))
    avg_output_tokens = float(np.mean(output_token_counts))

    # Calculate Cost per 1,000 Queries: (Tokens * Price_Per_1M) / 1000
    prem_cfg = tiers.get("premium", DEFAULT_PRICING_TIERS["premium"])
    fast_cfg = tiers.get("fast", DEFAULT_PRICING_TIERS["fast"])

    premium_cost_per_1k = (
        (avg_input_tokens * prem_cfg["input_per_1m"] / 1000.0) +
        (avg_output_tokens * prem_cfg["output_per_1m"] / 1000.0)
    )

    fast_cost_per_1k = (
        (avg_input_tokens * fast_cfg["input_per_1m"] / 1000.0) +
        (avg_output_tokens * fast_cfg["output_per_1m"] / 1000.0)
    )

    return {
        "avg_input_tokens_per_query": round(avg_input_tokens, 1),
        "avg_output_tokens_per_query": round(avg_output_tokens, 1),
        "premium_cost_per_1k_usd": round(premium_cost_per_1k, 3),
        "fast_cost_per_1k_usd": round(fast_cost_per_1k, 3)
    }