import gc
import ctypes
import numpy as np
import re
import html
from bs4 import BeautifulSoup
import logging
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics.pairwise import cosine_similarity

from src.embed_and_store.shared_embedder import get_shared_embedder

logger = logging.getLogger("app") 

def release_system_memory():
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

def robust_clean(raw_text):
    unescaped_text = html.unescape(raw_text)
    soup = BeautifulSoup(unescaped_text, "html.parser")
    clean_text = soup.get_text()
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def get_embeddings_in_batches(texts: list[str], batch_size: int = 32) -> np.ndarray:
    embedder = get_shared_embedder()
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_vecs = list(embedder.embed_documents(batch))
        all_embeddings.extend(batch_vecs)
        
    release_system_memory()
    return np.array(all_embeddings, dtype=np.float32)

def semantic_chunking_pro(texts, window_size=3, percentile=10):
    text = robust_clean(texts)
    
    # 1. PURE PYTHON REGEX SPLITTING (Saves ~100MB of RAM by removing SpaCy)
    # Splits on standard punctuation (. ! ?) followed by whitespace and a capital letter
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
    
    if len(sentences) < window_size:
        return [" ".join(sentences)]

    # 2. Batch Embedding
    vectors = get_embeddings_in_batches(sentences, batch_size=32)

    # 3. Sliding Window Means 
    windows = sliding_window_view(vectors, window_shape=(window_size,), axis=0)
    mean_vectors = windows.mean(axis=1)

    # 4. Vectorized Cosine Similarity
    sim_matrix = cosine_similarity(mean_vectors)
    similarities = np.diagonal(sim_matrix, offset=1)

    # 5. Thresholding
    threshold = np.percentile(similarities, percentile)
    
    # 6. Find Breakpoints
    offset = window_size // 2
    breakpoints = [i + offset for i, s in enumerate(similarities) if s <= threshold]

    # 7. Slicing
    chunks = []
    start_idx = 0
    for bp in breakpoints:
        chunks.append(" ".join(sentences[start_idx:bp]))
        start_idx = bp
    
    chunks.append(" ".join(sentences[start_idx:]))
    
    # Cleanup
    del vectors
    del sim_matrix
    release_system_memory()
    
    return chunks
