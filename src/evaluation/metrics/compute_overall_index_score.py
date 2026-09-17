import math
from typing import Dict, Any

def compute_overall_index_score(
    strategy_metrics: Dict[str, float],
    baseline_cost_per_1k: float = 10.0
) -> Dict[str, Any]:
    """
    Calculates a bounded, weighted composite index score and assigns letter grades.
    
    Weights:
    - 35% Context Recall
    - 25% Context Precision
    - 15% Boundary Health
    - 15% Intra-Chunk Coherence
    - 10% Cost Efficiency (Logarithmically scaled)
    """
    # Helper function to clamp raw inputs strictly between 0.0 and 1.0
    def clamp(val: float) -> float:
        return max(0.0, min(1.0, float(val)))

    recall = clamp(strategy_metrics.get("context_recall", 0.0))
    precision = clamp(strategy_metrics.get("context_precision", 0.0))
    boundary = clamp(strategy_metrics.get("boundary_health", 0.0))
    coherence = clamp(strategy_metrics.get("intra_chunk_coherence", 0.0))

    # Logarithmic cost efficiency scaling (prevents step-function cliff drops)
    raw_cost = float(strategy_metrics.get("premium_cost_per_1k_usd", baseline_cost_per_1k))
    if raw_cost <= 0.0:
        cost_efficiency = 1.0
    else:
        # Scale cost relative to baseline using log decay
        cost_ratio = raw_cost / baseline_cost_per_1k
        cost_efficiency = clamp(1.0 / (1.0 + math.log10(max(1.0, cost_ratio) + 0.1)))

    # Composite index calculation
    composite_score = (
        (0.35 * recall) +
        (0.25 * precision) +
        (0.15 * boundary) +
        (0.15 * coherence) +
        (0.10 * cost_efficiency)
    )

    final_score_pct = round(composite_score * 100, 1)

    # Standardized Production Grade Mapping
    if final_score_pct >= 85.0:
        grade = "A+ (Production Ready - Recommended)"
    elif final_score_pct >= 75.0:
        grade = "A (Production Ready - Strong)"
    elif final_score_pct >= 65.0:
        grade = "B (Acceptable - Minor Noise Risk)"
    elif final_score_pct >= 50.0:
        grade = "C (Marginal - High Optimization Required)"
    else:
        grade = "F (Failing - Unsuitable for Deployment)"

    return {
        "overall_index_score_pct": final_score_pct,
        "system_grade": grade,
        "cost_efficiency_score": round(cost_efficiency * 100, 1),
        "normalized_metrics": {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "boundary": round(boundary, 4),
            "coherence": round(coherence, 4)
        }
    }