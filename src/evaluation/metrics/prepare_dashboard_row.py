




def performance_metrics(strategy_name: str, raw_metrics: dict, latency: dict) -> dict:
    """Transforms raw evaluation metrics into a business-friendly dashboard row."""
    
    # 1. Retrieval Accuracy (Merge Recall & Precision)
    recall = raw_metrics.get("context_recall", 0.0)
    precision = raw_metrics.get("context_precision", 0.0)
    accuracy_pct = round(((recall + precision) / 2) * 100)
    
    # 2. Chunk Quality (Merge Boundary Health & Coherence)
    boundary = raw_metrics.get("boundary_health", 0.0)
    coherence = raw_metrics.get("intra_chunk_coherence", 0.0)
    quality_pct = round(((boundary + coherence) / 2) * 100)
    
    # 3. Format Cost
    cost = round(raw_metrics.get("projected_cost_per_1k_queries", 0.0), 2)
    
    return {
        "Strategy": strategy_name.replace("_", " ").title(),
        "System Grade": raw_metrics.get("system_grade", "N/A"),
        "Index Score": f"{raw_metrics.get('overall_index_score', 0.0)}%",
        "Retrieval Accuracy": f"{accuracy_pct}%",
        "Chunk Quality": f"{quality_pct}%",
        # NOTE: Replace avg_vector_search_latency_ms with the p95_ms we built earlier!
        "Cost / 1K": f"${cost}",
        "p50 Latency":latency.get('p50_ms', 0.0)
    }