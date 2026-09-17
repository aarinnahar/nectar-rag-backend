import re
import time
import logging
import numpy as np
from typing import Dict, Any, List, Union

# 1. IMPORT FASTEMBED (Replacing PyTorch-heavy SentenceTransformer)
from fastembed import TextEmbedding

from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state

# Import all individual modular scoring functions
from src.evaluation.metrics.compute_retrieval_metrics import compute_retrieval_metrics
from src.evaluation.metrics.compute_boundary_health import compute_boundary_health
from src.evaluation.metrics.compute_intra_chunk_coherence import compute_intra_chunk_coherence
from src.evaluation.metrics.compute_hit_rate import compute_hit_rate_at_k
from src.evaluation.metrics.compute_inter_chunk_redundancy import compute_inter_chunk_redundancy
from src.evaluation.metrics.compute_avg_vector_search_latency_ms import compute_vector_search_latency_metrics
from src.evaluation.metrics.compute_total_ingestion_time_s import compute_total_ingestion_time_s
from src.evaluation.metrics.compute_projected_costs import compute_projected_costs
from src.evaluation.metrics.compute_system_grade import compute_system_grade

logger = logging.getLogger("app")


# ==============================================================================
# FAST-EMBED WRAPPER (ONNX Drop-in Replacement for SentenceTransformer)
# ==============================================================================
class FastEmbedWrapper:
    """
    Acts exactly like a SentenceTransformer to downstream metric functions,
    but runs on the ultra-lightweight ONNX C++ runtime to save RAM.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        # FastEmbed requires a list of strings
        if isinstance(texts, str):
            texts = [texts]
        
        # model.embed returns a generator, so we immediately convert it to a 2D numpy array
        # This matches PyTorch's output format perfectly for your downstream math
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings, dtype=np.float32)

# Initialize the lightweight ONNX embedding model once globally
local_embedder = FastEmbedWrapper("BAAI/bge-small-en-v1.5")
# ==============================================================================


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
        "strategy": strategy_name.replace("_", " ").title(),
        "system_grade": raw_metrics.get("system_grade", "N/A"),
        "index_score": f"{raw_metrics.get('overall_index_score', 0.0)}%",
        "retrieval_accuracy": f"{accuracy_pct}",
        "chunk_quality": f"{quality_pct}",
        # NOTE: Replace avg_vector_search_latency_ms with the p95_ms we built earlier!
        "cost_per1K": f"${cost}",
        "p95_latency":latency.get('p95_ms', 0.0)
    }

def _clean_chunk_item(chunk_item: Union[str, dict]) -> dict:
    """
    Parses and cleans chunk representations whether they arrive as raw strings,
    repr strings (e.g. "page_content='...'"), or dictionaries.
    """
    if isinstance(chunk_item, dict):
        text = chunk_item.get("page_content", chunk_item.get("text", ""))
        chunk_id = chunk_item.get("chunk_id", None)
        return {"text": text, "chunk_id": chunk_id}

    if isinstance(chunk_item, str):
        raw_str = chunk_item.strip()
        # Clean string formats like: page_content='...' or page_content="..."
        if raw_str.startswith("page_content="):
            match = re.search(r"^page_content=['\"](.*)['\"]$", raw_str, re.DOTALL)
            if match:
                raw_str = match.group(1)
            else:
                raw_str = raw_str.replace("page_content=", "").strip("'\"")

        return {"text": raw_str, "chunk_id": None}

    return {"text": str(chunk_item), "chunk_id": None}


def evaluate_retrieved_chunks(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph evaluator node that processes retrieved chunks per strategy from AgentState,
    executes zero-LLM deterministic checks across all metric dimensions, and returns 
    comprehensive aggregated evaluation scores.
    """
    # LAZY LOAD: Initialize the wrapper strictly inside the node
    local_embedder = FastEmbedWrapper("BAAI/bge-small-en-v1.5")
    retrieved_chunks_by_strategy = state.get("retrieved_chunks", {})
    vectorstores = state.get("vectorstore", {})
    chunk_factory = state.get("chunk_factory", {})
    ingestion_time_dict = state.get("ingestion_time", {})

    if not retrieved_chunks_by_strategy:
        logger.warning("No retrieved chunks found in state for evaluation.")
        return {"evaluation_scores": {}}

    evaluation_scores = {}

    for strategy, query_items in retrieved_chunks_by_strategy.items():
        logger.info(f"EVALUATOR NODE: Evaluating strategy -> {strategy}")

        recall_list = []
        precision_list = []
        hit_rate_list = []
        boundary_list = []
        coherence_list = []
        search_latency_list = []

        # Fetch FAISS index if available in state for this strategy
        faiss_index = None
        if strategy in vectorstores and hasattr(vectorstores[strategy], "index"):
            faiss_index = vectorstores[strategy].index

        for item in query_items:
            question = item.get("question", "")
            golden_target = item.get("answer", "")
            search_time = item.get("vector_db_search", 0.0)
            raw_chunks = item.get("retrieved_chunks", [])

            search_latency_list.append(search_time)

            # Parse and normalize chunks
            cleaned_chunks = [_clean_chunk_item(c) for c in raw_chunks]
            chunk_texts = [c["text"] for c in cleaned_chunks]

            # -----------------------------------------------------------------
            # 1. RETRIEVAL METRICS (Context Recall & Precision)
            # -----------------------------------------------------------------
            # Embed golden target context ONCE per question
            golden_vector = local_embedder.encode([golden_target]).reshape(1, -1)

            retrieval_res = compute_retrieval_metrics(
                retrieved_chunks=cleaned_chunks,
                golden_target=golden_target,
                # faiss_index=faiss_index,
                # golden_vector=golden_vector,
                embedder=local_embedder
            )

            recall_list.append(retrieval_res["context_recall"])
            precision_list.append(retrieval_res["context_precision"])

            # -----------------------------------------------------------------
            # 2. HIT RATE @ K
            # -----------------------------------------------------------------
            hit = compute_hit_rate_at_k(
                retrieved_chunks=cleaned_chunks,
                golden_target=golden_target,
                faiss_index=faiss_index,
                golden_vector=golden_vector, #type:ignore
                embedder=local_embedder,
                k=4,
                threshold=0.65
            )
            hit_rate_list.append(hit)

            # -----------------------------------------------------------------
            # 3. BOUNDARY HEALTH (Structural Integrity)
            # -----------------------------------------------------------------
            boundary_score = compute_boundary_health(chunk_texts)
            boundary_list.append(boundary_score)

            # -----------------------------------------------------------------
            # 4. INTRA-CHUNK COHERENCE (Topic Unity)
            # -----------------------------------------------------------------
            coherence_score = compute_intra_chunk_coherence(
                chunk_texts, 
                embedder=local_embedder
            )
            coherence_list.append(coherence_score)

        # ---------------------------------------------------------------------
        # Strategy-Level Aggregations & Secondary Metric Module Calls
        # ---------------------------------------------------------------------
        avg_recall = round(float(np.mean(recall_list)), 3) if recall_list else 0.0
        avg_precision = round(float(np.mean(precision_list)), 3) if precision_list else 0.0
        avg_hit_rate = round(float(np.mean(hit_rate_list)), 3) if hit_rate_list else 0.0
        avg_boundary = round(float(np.mean(boundary_list)), 3) if boundary_list else 0.0
        avg_coherence = round(float(np.mean(coherence_list)), 3) if coherence_list else 0.0
        
        # Inter-Chunk Redundancy across the whole document chunks for this strategy
        doc_chunks = chunk_factory.get(strategy, [])
        inter_chunk_redundancy = compute_inter_chunk_redundancy(doc_chunks)

        # Search Latency (ms)
        avg_search_latency = compute_vector_search_latency_metrics(query_items)

        # Total Ingestion Time (s)
        total_ingestion_time = compute_total_ingestion_time_s(state, strategy) #type:ignore

        # Token Footprint & Cost Projections (Premium & Fast tiers)
        costs_data = compute_projected_costs(query_items)
        save_state(
                filename="costs_data",
                data=costs_data,
            )
        avg_tokens_per_query = costs_data.get("avg_input_tokens_per_query", 0.0)
        projected_cost_per_1k = costs_data.get("premium_cost_per_1k_usd", 0.0)

        # Temporary metrics bundle to pass into the System Grade calculator
        strategy_metrics_temp = {
            "context_recall": avg_recall,
            "context_precision": avg_precision,
            "boundary_health": avg_boundary,
            "intra_chunk_coherence": avg_coherence,
            "premium_cost_per_1k_usd": projected_cost_per_1k
        }

        # Compute Overall Composite Score & Letter Grade
        grade_result = compute_system_grade(strategy_metrics_temp)
        final_score_pct = grade_result.get("overall_index_score_pct", 0.0)
        letter_grade = grade_result.get("system_grade", "F")

        # ---------------------------------------------------------------------
        # Assemble Final Dictionary Structure per Strategy
        # ---------------------------------------------------------------------
        evaluation_scores[strategy] = {
            "context_recall": avg_recall,
            "context_precision": avg_precision,
            "hit_rate_at_k": avg_hit_rate,
            "boundary_health": avg_boundary,
            "intra_chunk_coherence": avg_coherence,
            "inter_chunk_redundancy": inter_chunk_redundancy,
            "avg_vector_search_latency_ms": avg_search_latency,
            "total_ingestion_time_s": total_ingestion_time,
            "avg_tokens_per_query": avg_tokens_per_query,
            "projected_cost_per_1k_queries": projected_cost_per_1k,
            "overall_index_score": final_score_pct,
            "system_grade": letter_grade,
            "total_queries_evaluated": len(query_items)
        }

        logger.info(
            f"STRATEGY [{strategy}] RESULTS -> "
            f"Grade: {letter_grade} ({final_score_pct}%) | "
            f"Recall: {avg_recall} | Precision: {avg_precision} | Hit Rate: {avg_hit_rate} | "
            f"Cost/1k: ${projected_cost_per_1k}"
        )

    save_state(filename="evaluation_scores_update", data=evaluation_scores)
    performance_metrics_scores = {} 

    for strategy, metrics in evaluation_scores.items():
        scores = performance_metrics(strategy_name=strategy, raw_metrics=metrics, latency=metrics['avg_vector_search_latency_ms'])
        performance_metrics_scores[strategy] = scores

    # Save state locally for debugging and persistence
    save_state(filename="evaluation_scores", data=evaluation_scores)
    save_state(filename="performance_metrics", data=performance_metrics_scores)
    logger.debug("Evaluation node completed successfully.")

    return {"evaluation_scores": evaluation_scores, "performance_metrics" : performance_metrics_scores}
