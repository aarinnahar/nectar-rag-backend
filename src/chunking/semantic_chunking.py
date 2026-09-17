import gc
import ctypes
import numpy as np
import re
import html
from bs4 import BeautifulSoup
import logging
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.release_memory import release_system_memory
from src.config.settings import Settings
from src.embed_and_store.shared_embedder import get_shared_embedder

logger = logging.getLogger("app") 


def robust_clean(raw_text):
    unescaped_text = html.unescape(raw_text)
    soup = BeautifulSoup(unescaped_text, "html.parser")
    clean_text = soup.get_text()
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def get_embeddings_in_batches(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Processes embeddings in small chunks to prevent FastEmbed RAM spikes."""
    embedder = get_shared_embedder()
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # FastEmbed returns a generator, so we wrap it in list()
        batch_vecs = list(embedder.embed_documents(batch))
        all_embeddings.extend(batch_vecs)
        
    # Immediately release the memory used by the embedding engine
    release_system_memory()
    return np.array(all_embeddings, dtype=np.float32)

def semantic_chunking_pro(texts, window_size=3, percentile=10):
    text = robust_clean(texts)
    
    # 1. LAZY LOAD SPACY: Prevents 80MB memory hit during app startup
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer", "textcat", "attribute_ruler"])
    nlp.enable_pipe("senter")
    
    # 2. Split sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
    
    # Destroy the SpaCy objects immediately to reclaim RAM
    del doc
    del nlp
    release_system_memory()
    
    if len(sentences) < window_size:
        return [" ".join(sentences)]

    # 3. Batch Embedding (Replaces the single massive allocation)
    vectors = get_embeddings_in_batches(sentences, batch_size=32)

    # 4. Sliding Window Means 
    windows = sliding_window_view(vectors, window_shape=(window_size,), axis=0)
    mean_vectors = windows.mean(axis=1)

    # 5. Vectorized Cosine Similarity
    sim_matrix = cosine_similarity(mean_vectors)
    similarities = np.diagonal(sim_matrix, offset=1)

    # 6. Thresholding
    threshold = np.percentile(similarities, percentile)
    
    # 7. Find Breakpoints
    offset = window_size // 2
    breakpoints = [i + offset for i, s in enumerate(similarities) if s <= threshold]

    # 8. Slicing
    chunks = []
    start_idx = 0
    for bp in breakpoints:
        chunks.append(" ".join(sentences[start_idx:bp]))
        start_idx = bp
    
    chunks.append(" ".join(sentences[start_idx:]))
    
    # Final cleanup before returning the text strings
    del vectors
    del sim_matrix
    release_system_memory()
    
    return chunks
