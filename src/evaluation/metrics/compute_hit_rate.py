import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

def compute_hit_rate_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    golden_target: str,
    faiss_index: Any,
    golden_vector: Optional[np.ndarray] = None,
    embedder: Any = None,
    k: int = 4,
    threshold: float = 0.65
) -> int:
    """
    Calculates binary Hit Rate @ K using vectorized batch operations.
    """
    if not retrieved_chunks:
        return 0

    # 1. Resolve the golden vector safely
    if golden_vector is None:
        if embedder is None:
            raise ValueError("Either 'golden_vector' or 'embedder' must be provided.")
        # Ensure it is normalized for accurate similarity
        golden_vector = embedder.encode([golden_target], normalize_embeddings=True)
    else:
        golden_vector = np.array(golden_vector).reshape(1, -1)

    # 2. Slice top K and batch extract vectors
    top_k_chunks = retrieved_chunks[:k]
    chunk_vectors = []

    for chunk_info in top_k_chunks:
        chunk_id = chunk_info.get("chunk_id")
        if chunk_id is None:
            continue

        try:
            # FAISS usually throws a RuntimeError if the ID does not exist
            vec = faiss_index.reconstruct(int(chunk_id))
            chunk_vectors.append(vec)
        except RuntimeError as e:
            logging.warning(f"FAISS ID {chunk_id} missing or reconstruct unsupported: {e}")
            continue
        except Exception as e:
            logging.error(f"Unexpected error reconstructing FAISS ID {chunk_id}: {e}")
            continue

    if not chunk_vectors:
        return 0

    # 3. Vectorized Math (No loops)
    # Stack individual vectors into a matrix: shape (num_valid_chunks, embedding_dim)
    chunk_matrix = np.vstack(chunk_vectors)
    
    # Compute similarities for all K chunks in one highly optimized operation
    sims = cosine_similarity(golden_vector, chunk_matrix).flatten()

    # 4. Binary check: If any chunk crosses the threshold, it's a hit
    if np.any(sims >= threshold):
        return 1
        
    return 0