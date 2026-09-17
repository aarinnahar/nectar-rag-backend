import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def split_into_sentences(text: str) -> list[str]:
    """Splits text into non-empty sentences. Upgraded for better edge-case handling."""
    # Using a slightly more robust regex to avoid splitting on initials or decimals
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def compute_retrieval_metrics(
    retrieved_chunks: list[dict],
    golden_target: str,
    embedder,
    similarity_threshold: float = 0.55
) -> dict[str, float]:
    """
    Computes Context Recall and Context Precision deterministically.
    
    CRITICAL: `retrieved_chunks` MUST be ordered from rank 1 to K based on 
    the retriever's original search scores for Context Precision to be valid.
    """
    if not retrieved_chunks or not golden_target.strip():
        return {"context_recall": 0.0, "context_precision": 0.0}

    chunk_texts = [c.get("page_content", c.get("text", "")) for c in retrieved_chunks]
    target_sentences = split_into_sentences(golden_target)
    
    if not target_sentences or not chunk_texts:
        return {"context_recall": 0.0, "context_precision": 0.0}

    # -------------------------------------------------------------------------
    # 1. CONTEXT RECALL (Sentence-to-Sentence Semantic Coverage)
    # -------------------------------------------------------------------------
    # Break chunks down into sentences to avoid "embedding washout" of specific facts
    all_chunk_sentences = []
    for chunk in chunk_texts:
        all_chunk_sentences.extend(split_into_sentences(chunk))
        
    if not all_chunk_sentences:
        all_chunk_sentences = chunk_texts # Fallback if splitting fails
        
    chunk_sent_embeddings = embedder.encode(all_chunk_sentences, normalize_embeddings=True)
    target_sent_embeddings = embedder.encode(target_sentences, normalize_embeddings=True)
    
    # Compare every target sentence against every chunk sentence
    sentence_sims = cosine_similarity(target_sent_embeddings, chunk_sent_embeddings)
    
    # Max similarity per target sentence across all available chunk sentences
    max_sim_per_target = np.max(sentence_sims, axis=1)
    recalled_sentences = np.sum(max_sim_per_target >= similarity_threshold)
    
    context_recall = float(recalled_sentences / len(target_sentences))

    # -------------------------------------------------------------------------
    # 2. CONTEXT PRECISION (MAP@K / Rank-Weighted Precision)
    # -------------------------------------------------------------------------
    # For precision, we want to know if the chunk *as a whole* contains the answer.
    # Comparing the chunk to the whole golden target is appropriate here.
    chunk_embeddings = embedder.encode(chunk_texts, normalize_embeddings=True)
    golden_vector = embedder.encode([golden_target], normalize_embeddings=True)
    
    chunk_target_sims = cosine_similarity(chunk_embeddings, golden_vector).flatten()

    precision_scores = []
    relevant_chunks_found = 0

    # Iterating through ranks (assuming retrieved_chunks is pre-sorted!)
    for rank_idx, sim in enumerate(chunk_target_sims, start=1):
        if sim >= similarity_threshold:
            relevant_chunks_found += 1
            precision_scores.append(relevant_chunks_found / rank_idx)

    # Standard MAP logic
    context_precision = float(np.mean(precision_scores)) if precision_scores else 0.0

    return {
        "context_recall": round(context_recall, 4),
        "context_precision": round(context_precision, 4)
    }