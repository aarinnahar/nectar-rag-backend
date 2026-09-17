import re
import numpy as np
from typing import List, Union, Any
from sklearn.metrics.pairwise import cosine_similarity

# 1. Compile regex globally to avoid loop overhead
SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+')

def _extract_text(chunk: Any) -> str:
    """Safely extracts text from strings, dicts, LangChain Documents, or LlamaIndex Nodes."""
    if isinstance(chunk, str): return chunk
    if hasattr(chunk, "page_content"): return str(chunk.page_content)
    if hasattr(chunk, "text"): return str(chunk.text)
    if isinstance(chunk, dict): return str(chunk.get("page_content", chunk.get("text", "")))
    return str(chunk)

def split_into_sentences(text: str) -> List[str]:
    """Fast regex sentence tokenizer."""
    sentences = SENTENCE_SPLIT_REGEX.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def compute_intra_chunk_coherence(
    chunk_input: Union[str, List[Any]],
    embedder=None
) -> float:
    """
    Computes Intra-Chunk Semantic Coherence (0.0 to 1.0) with Production Batching.
    """
    if not isinstance(chunk_input, list):
        chunk_input = [chunk_input]

    chunks = [_extract_text(c).strip() for c in chunk_input]
    chunks = [c for c in chunks if c]

    if not chunks:
        return 0.0

    # 1. Prepare all sentences across all chunks for BATCHING
    chunk_sentences_map = [] 
    flat_sentences = []      
    
    for text in chunks:
        sentences = split_into_sentences(text)
        chunk_sentences_map.append(sentences)
        flat_sentences.extend(sentences)

    chunk_scores = []
    
    # -------------------------------------------------------------------------
    # EMBEDDER PATH (Batched Execution)
    # -------------------------------------------------------------------------
    if embedder is not None and flat_sentences:
        # BATCH ENCODE: A single, highly optimized call to the model
        flat_embeddings = embedder.encode(flat_sentences, normalize_embeddings=True)
        
        current_idx = 0
        for sentences in chunk_sentences_map:
            num_sentences = len(sentences)
            
            if num_sentences <= 1:
                chunk_scores.append(0.70)
                current_idx += num_sentences
                continue
                
            # Slice the pre-computed embeddings for this specific chunk
            chunk_embs = flat_embeddings[current_idx : current_idx + num_sentences]
            current_idx += num_sentences
            
            chunk_centroid = np.mean(chunk_embs, axis=0, keepdims=True)
            sims = cosine_similarity(chunk_embs, chunk_centroid).flatten()
            coherence = float(np.mean(sims))
            chunk_scores.append(max(0.0, min(1.0, coherence)))
            
    # -------------------------------------------------------------------------
    # LEXICAL FALLBACK PATH
    # -------------------------------------------------------------------------
    else:
        stopwords = {"the", "a", "an", "is", "are", "and", "or", "in", "on", "at", "to", "for", "of", "with"}
        
        for sentences in chunk_sentences_map:
            num_sentences = len(sentences)
            if num_sentences <= 1:
                chunk_scores.append(0.70)
                continue
                
            words_per_sent = [
                set(re.findall(r'\w+', s.lower())) - stopwords 
                for s in sentences
            ]
            
            adjacent_jaccards = []
            for i in range(num_sentences - 1):
                w1, w2 = words_per_sent[i], words_per_sent[i + 1]
                union = w1.union(w2)
                adjacent_jaccards.append(len(w1.intersection(w2)) / len(union) if union else 0.0)
            
            coherence = min(1.0, float(np.mean(adjacent_jaccards)) * 2.5)
            chunk_scores.append(max(0.0, min(1.0, coherence)))

    return round(float(np.mean(chunk_scores)), 4) if chunk_scores else 0.0