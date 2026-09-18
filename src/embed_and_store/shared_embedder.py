import gc
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

_SHARED_EMBEDDER = None

def get_shared_embedder():
    """Returns a single, shared ONNX embedding session across all nodes."""
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        _SHARED_EMBEDDER = FastEmbedEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            threads=2,
            batch_size=32
        )
    return _SHARED_EMBEDDER
