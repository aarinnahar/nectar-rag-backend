import logging
from typing import Dict, Any
from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state

logger = logging.getLogger("app")


def generate_evaluation_insights(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph analysis node that ingests evaluation scores, ranks strategies deterministically,
    identifies the winner, runner-up, and loser, and compiles structured analytical insights
    via conditional rules for downstream LLM synthesis.
    """
    evaluation_scores = state.get("evaluation_scores", {})

    if not evaluation_scores:
        logger.warning("No evaluation scores found in state to analyze.")
        return {"evaluation_insights": {}}

    # 1. Sort strategies deterministically by overall_index_score descending
    sorted_strategies = sorted(
        evaluation_scores.items(),
        key=lambda item: item[1].get("overall_index_score", 0.0),
        reverse=True
    )

    strategies_ranked = [s[0] for s in sorted_strategies]
    
    winner_name = strategies_ranked[0] if len(strategies_ranked) > 0 else "None"
    runner_up_name = strategies_ranked[1] if len(strategies_ranked) > 1 else None
    loser_name = strategies_ranked[-1] if len(strategies_ranked) > 2 else None

    winner_metrics = evaluation_scores.get(winner_name, {})
    runner_up_metrics = evaluation_scores.get(runner_up_name, {}) if runner_up_name else {}
    loser_metrics = evaluation_scores.get(loser_name, {}) if loser_name else {}

    # 2. Construct the Structured Insights Payload
    insights_payload = {
        "rankings": strategies_ranked,
        "winner": {
            "name": winner_name,
            "score": winner_metrics.get("overall_index_score", 0.0),
            "grade": winner_metrics.get("system_grade", "N/A"),
        },
        "runner_up": {
            "name": runner_up_name,
            "score": runner_up_metrics.get("overall_index_score", 0.0) if runner_up_name else 0.0,
            "grade": runner_up_metrics.get("system_grade", "N/A") if runner_up_name else "N/A",
        },
        "loser": {
            "name": loser_name,
            "score": loser_metrics.get("overall_index_score", 0.0) if loser_name else 0.0,
            "grade": loser_metrics.get("system_grade", "N/A") if loser_name else "N/A",
        },
        "comparative_statements": {}
    }

    # 3. Deterministic If-Else Rule Engine for Explanatory Statements
    
    # --- Winner Rationale ---
    w_recall = winner_metrics.get("context_recall", 0.0)
    w_precision = winner_metrics.get("context_precision", 0.0)
    w_cost = winner_metrics.get("projected_cost_per_1k_queries", 0.0)
    w_boundary = winner_metrics.get("boundary_health", 0.0)

    winner_parts = [
        f"Strategy '{winner_name}' emerged as the definitive winner, claiming 1st place with an overall index score of {winner_metrics.get('overall_index_score', 0)}% and achieving a system grade of {winner_metrics.get('system_grade', 'N/A')}."
    ]
    
    if w_recall >= 0.80 and w_precision >= 0.75:
        winner_parts.append(f"It dominated accuracy metrics with a high context recall of {w_recall} and precision of {w_precision}, ensuring the generator receives precise, complete context without noise.")
    else:
        winner_parts.append(f"It maintained an acceptable balance with a recall of {w_recall} and strong structural integrity.")

    if w_cost <= 5.00:
        winner_parts.append(f"Furthermore, its cost efficiency is exceptional, projecting an affordable ${w_cost} per 1,000 queries.")
    else:
        winner_parts.append(f"Although its token cost runs slightly higher at ${w_cost} per 1k queries, its superior retrieval accuracy justifies the investment.")

    insights_payload["comparative_statements"]["winner_rationale"] = " ".join(winner_parts)

    # --- Runner-Up Rationale ---
    if runner_up_name:
        ru_score = runner_up_metrics.get("overall_index_score", 0.0)
        ru_recall = runner_up_metrics.get("context_recall", 0.0)
        ru_cost = runner_up_metrics.get("projected_cost_per_1k_queries", 0.0)
        
        insights_payload["comparative_statements"]["runner_up_rationale"] = (
            f"Strategy '{runner_up_name}' secured the runner-up position with an overall score of {ru_score}% "
            f"and a recall of {ru_recall}. While it offered competitive throughput and costed ${ru_cost} per 1k queries, "
            f"it fell short of the winner due to minor trade-offs in structural boundary health or precision."
        )
    else:
        insights_payload["comparative_statements"]["runner_up_rationale"] = "No runner-up strategy available for comparison."

    # --- Loser Rationale ---
    if loser_name and loser_name != winner_name:
        l_score = loser_metrics.get("overall_index_score", 0.0)
        l_recall = loser_metrics.get("context_recall", 0.0)
        l_redundancy = loser_metrics.get("inter_chunk_redundancy", 0.0)
        l_latency = loser_metrics.get("avg_vector_search_latency_ms", 0.0)

        loser_parts = [
            f"Strategy '{loser_name}' ranked last among the evaluated options with an overall score of {l_score}% ({loser_metrics.get('system_grade', 'N/A')})."
        ]
        
        if l_recall < 0.65:
            loser_parts.append(f"It suffered from critical retrieval failure with a low recall of {l_recall}, frequently missing crucial sentences required by the golden dataset.")
        if l_redundancy > 0.30:
            loser_parts.append(f"It exhibited high inter-chunk redundancy ({l_redundancy}), leading to significant index bloat and redundant token usage.")
        if l_latency['mean_ms'] > 40.0:
            loser_parts.append(f"Vector search latency also lagged at {l_latency}ms.")

        insights_payload["comparative_statements"]["loser_rationale"] = " ".join(loser_parts)
    else:
        insights_payload["comparative_statements"]["loser_rationale"] = "No distinct failing strategy identified."

    # Save payload locally and return state update
    save_state(filename="evaluation_insights", data=insights_payload)
    logger.info("Insights generator analysis node completed successfully.")

    return {"evaluation_insights": insights_payload}