import numpy as np
from typing import List, Dict, Any
from src.utils.save_state import save_state


def compute_vector_search_latency_metrics(strategy_queries_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates production-grade latency metrics (Mean, p50, p95, p99) for vector database searches.
    
    Filters out missing, failed, or invalid data points to prevent skewed averages.
    
    :param strategy_queries_data: List of query dicts.
    :return: A dictionary containing latency percentiles in milliseconds.
    """
    save_state(
            filename="strategy_queries_data",
            data=strategy_queries_data,
        )

    if not strategy_queries_data:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    valid_latencies_sec = []

    for item in strategy_queries_data:
        # Safely extract the value. Don't use .get(..., 0.0) as it masks failures!
        val = item.get("vector_db_search_time")
        
        # Ensure it's a valid, positive number (strips None, strings, or errors)
        if isinstance(val, (int, float)) and val > 0:
            valid_latencies_sec.append(float(val))

    if not valid_latencies_sec:
        # If all queries failed or lacked data, return zeros (or handle exception upstream)
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    # Convert all valid seconds to milliseconds for the calculation
    latencies_ms = np.array(valid_latencies_sec) * 1000

    save_state(
            filename="latencies_ms",
            data=latencies_ms,
        )

    save_state(
            filename="valid_latencies_sec",
            data=valid_latencies_sec,
        )

    # Calculate production percentiles using numpy
    return {
        "mean_ms": round(float(np.mean(latencies_ms)), 5),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 5),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 5),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 5),
        "valid_samples": len(latencies_ms) # Useful for your dashboard to track error rates
    }