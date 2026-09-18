import gc
import ctypes
import numpy as np
import re
import html
from bs4 import BeautifulSoup
import logging
from numpy.lib.stride_tricks import sliding_window_view
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

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

def get_micro_embeddings(texts: list[str]) -> np.ndarray:
    """Uses the 22MB micro-model strictly for boundary calculation."""
    # 1. Initialize the ultra-fast micro model locally
    micro_embedder = FastEmbedEmbeddings(
        model_name="taylorai/bge-micro-v2", 
        threads=2, 
        batch_size=32
    )
    
    all_embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        batch_vecs = list(micro_embedder.embed_documents(batch))
        all_embeddings.extend(batch_vecs)
        
    # 2. Destroy the micro model from RAM immediately to free space
    del micro_embedder
    release_system_memory()
    
    return np.array(all_embeddings, dtype=np.float32)

def semantic_chunking_pro(texts, window_size=3, percentile=10):
    text = robust_clean(texts)
    
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
    
    if len(sentences) < window_size:
        return [" ".join(sentences)]

    # Use the isolated micro-model instead of the shared embedder
    vectors = get_micro_embeddings(sentences)

    windows = sliding_window_view(vectors, window_shape=(window_size,), axis=0)
    mean_vectors = windows.mean(axis=1)

    # 3. HIGH-SPEED VECTOR MATH (Eliminates Sklearn bottleneck)
    # Normalize vectors
    norms = np.linalg.norm(mean_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normed_vectors = mean_vectors / norms
    
    # Calculate dot product only for adjacent windows
    similarities = np.sum(normed_vectors[:-1] * normed_vectors[1:], axis=1)

    threshold = np.percentile(similarities, percentile)
    
    offset = window_size // 2
    breakpoints = [i + offset for i, s in enumerate(similarities) if s <= threshold]

    chunks = []
    start_idx = 0
    for bp in breakpoints:
        chunks.append(" ".join(sentences[start_idx:bp]))
        start_idx = bp
    
    chunks.append(" ".join(sentences[start_idx:]))
    
    # Aggressive cleanup
    del vectors, mean_vectors, normed_vectors, similarities
    release_system_memory()
    
    return chunks
