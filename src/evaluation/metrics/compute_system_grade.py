from typing import Dict, Any

def compute_system_grade(
    strategy_metrics: Dict[str, float], 
    max_budget_per_1k: float = 10.0,
    max_p95_latency_ms: float = 1000.0) -> Dict[str, Any]:
    """
    Calculates the final composite score with Hard Veto Gates for production readiness.
    
    Weights (Updated to include Latency):
    - 35% Context Recall 
    - 20% Context Precision 
    - 15% Intra-Chunk Coherence 
    - 10% Boundary Health 
    - 10% Latency Efficiency (p95)
    - 10% Cost Efficiency
    """
    # 1. Extract core metrics
    recall = strategy_metrics.get("context_recall", 0.0)
    precision = strategy_metrics.get("context_precision", 0.0)
    boundary = strategy_metrics.get("boundary_health", 0.0)
    coherence = strategy_metrics.get("intra_chunk_coherence", 0.0)
    
    # 2. Extract and calculate operational efficiencies
    cost = strategy_metrics.get("premium_cost_per_1k_usd", max_budget_per_1k)
    cost_efficiency = max(0.0, (max_budget_per_1k - cost) / max_budget_per_1k)

    p95_latency = strategy_metrics.get("p95_ms", max_p95_latency_ms)
    latency_efficiency = max(0.0, (max_p95_latency_ms - p95_latency) / max_p95_latency_ms)

    # 3. Apply HARD VETO GATES (The Production Safety Net)
    # If the system cannot retrieve data or is catastrophically slow, it fails instantly.
    veto_reason = None
    if recall < 0.40:
        veto_reason = "VETO: Unacceptable Context Recall (< 40%)"
    elif p95_latency > max_p95_latency_ms:
        veto_reason = f"VETO: p95 Latency exceeds strict timeout limit (> {max_p95_latency_ms}ms)"

    if veto_reason:
        return {
            "overall_index_score_pct": 0.0,
            "system_grade": f"F (Failing - {veto_reason})",
            "cost_efficiency_score": round(cost_efficiency * 100, 1),
            "latency_efficiency_score": round(latency_efficiency * 100, 1)
        }

    # 4. Compute weighted composite score (Only runs if it passes the veto gates)
    composite_score = (
        (0.35 * recall) + 
        (0.20 * precision) + 
        (0.15 * coherence) + 
        (0.10 * boundary) + 
        (0.10 * latency_efficiency) +
        (0.10 * cost_efficiency)
    )
    
    final_score_pct = round(composite_score * 100, 1)
    
    # 5. Assign professional grade tier
    if final_score_pct >= 90:
        grade = "A+ (Production Ready - Outstanding)"
    elif final_score_pct >= 80:
        grade = "A (Production Ready - Strong)"
    elif final_score_pct >= 70:
        grade = "B (Acceptable - Minor Tuning Required)"
    elif final_score_pct >= 60:
        grade = "C (Marginal - High Risk of Noise or Waste)"
    else:
        grade = "F (Failing - Do Not Deploy)"
        
    return {
        "overall_index_score_pct": final_score_pct,
        "system_grade": grade,
        "cost_efficiency_score": round(cost_efficiency * 100, 1),
        "latency_efficiency_score": round(latency_efficiency * 100, 1)
    }