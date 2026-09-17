import os 
import sys
# 1. Get the absolute path to the root of your project (two levels up)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# 2. Add the root directory to Python's system path
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.evaluation.metrics.compute_avg_tokens_per_query import compute_avg_tokens_per_query
from src.evaluation.metrics.compute_avg_vector_search_latency_ms import compute_vector_search_latency_metrics
from src.evaluation.metrics.compute_retrieval_metrics import compute_retrieval_metrics
from src.evaluation.metrics.compute_boundary_health import compute_boundary_health
from src.evaluation.metrics.compute_hit_rate import compute_hit_rate_at_k
from src.evaluation.metrics.compute_inter_chunk_redundancy import compute_inter_chunk_redundancy
from src.evaluation.metrics.compute_intra_chunk_coherence import compute_intra_chunk_coherence
import pytest
import numpy as np
from sentence_transformers import SentenceTransformer


# ===================================================================
# 1. FIXTURES & MOCKS (Test Setup)
# ===================================================================

@pytest.fixture(scope="module")
def embedder():
    """Loads the model into memory exactly once for the entire test suite."""
    print("\nLoading SentenceTransformer for tests...")
    return SentenceTransformer('all-MiniLM-L6-v2')

class MockFaissIndex:
    """A fake FAISS database that returns pre-computed vectors for testing."""
    def __init__(self, vector_map):
        self.vector_map = vector_map
        
    def reconstruct(self, chunk_id):
        if chunk_id not in self.vector_map:
            raise RuntimeError(f"ID {chunk_id} not found")
        return self.vector_map[chunk_id]


# ===================================================================
# 2. ACCURACY METRICS (Recall, Precision, MAP@K, Hit Rate)
# ===================================================================

def test_retrieval_perfect_match(embedder):
    golden_target = "The server is running on port 8000."
    retrieved_chunks = [
        {"text": "The backend server operates on port 8000."},
        {"text": "Python is a programming language."}
    ]
    result = compute_retrieval_metrics(retrieved_chunks, golden_target, embedder)
    
    assert result["context_recall"] == 1.0, f"Recall failed: {result['context_recall']}"
    assert result["context_precision"] == 1.0, f"Precision failed: {result['context_precision']}"

def test_retrieval_complete_miss(embedder):
    golden_target = "The server is running on port 8000."
    retrieved_chunks = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "I love drinking coffee in the morning."}
    ]
    result = compute_retrieval_metrics(retrieved_chunks, golden_target, embedder)
    
    assert result["context_recall"] == 0.0, f"Recall failed: {result['context_recall']}"
    assert result["context_precision"] == 0.0, f"Precision failed: {result['context_precision']}"

def test_retrieval_rank_penalty(embedder):
    golden_target = "The server is running on port 8000."
    retrieved_chunks = [
        {"text": "Python is a programming language."}, # Rank 1 (Miss)
        {"text": "I love drinking coffee."},           # Rank 2 (Miss)
        {"text": "The server runs on port 8000."}      # Rank 3 (Hit)
    ]
    result = compute_retrieval_metrics(retrieved_chunks, golden_target, embedder)
    
    assert result["context_recall"] == 1.0
    assert result["context_precision"] < 1.0, "MAP@K penalty failed!"
    assert result["context_precision"] > 0.0, "MAP@K is 0 but shouldn't be!"

def test_hit_rate_at_k(embedder):
    golden_target = "The company revenue in 2023 was 5 million dollars."
    miss_chunk = embedder.encode("The company was founded in 2015.")
    hit_chunk = embedder.encode("In 2023, revenue hit 5 million dollars.")
    
    mock_faiss = MockFaissIndex({101: miss_chunk, 102: hit_chunk})
    retrieved_chunks = [{"chunk_id": 101}, {"chunk_id": 102}]
    
    result = compute_hit_rate_at_k(
        retrieved_chunks=retrieved_chunks,
        golden_target=golden_target,
        faiss_index=mock_faiss,
        embedder=embedder,
        k=2,
        threshold=0.65
    )
    assert result == 1


# ===================================================================
# 3. STRUCTURAL METRICS (Boundary & Coherence)
# ===================================================================

def test_boundary_health_good_vs_bad():
    good_chunk = "This is a perfectly formed sentence."
    assert compute_boundary_health(good_chunk) == 1.0
    
    bad_chunk = "this is a terrible chunk that cuts off mid-sent-"
    assert compute_boundary_health(bad_chunk) < 0.5

def test_intra_chunk_coherence(embedder):
    coherent_chunk = "Machine learning uses neural networks. These networks process data."
    coherent_score = compute_intra_chunk_coherence([coherent_chunk], embedder)
    
    incoherent_chunk = "Machine learning uses neural networks. I like eating bananas."
    incoherent_score = compute_intra_chunk_coherence([incoherent_chunk], embedder)
    
    assert coherent_score > incoherent_score


# ===================================================================
# 4. REDUNDANCY METRIC (Inter-Chunk Overlap)
# ===================================================================

def test_inter_chunk_redundancy():
    redundant_chunks = [
        "The quick brown fox jumps over the dog.",
        "The quick brown fox jumps over the dog."
    ]
    assert compute_inter_chunk_redundancy(redundant_chunks, n_gram_size=3) == 1.0
    
    unique_chunks = [
        "The quick brown fox.",
        "Artificial intelligence is growing."
    ]
    assert compute_inter_chunk_redundancy(unique_chunks, n_gram_size=3) == 0.0


# ===================================================================
# 5. OPERATIONAL METRICS (Latency & Tokens)
# ===================================================================

def test_vector_search_latency():
    query_logs = [
        {"vector_db_search_time": 0.010}, 
        {"vector_db_search_time": 0.010}, 
        {"vector_db_search_time": 0.010}, 
        {"vector_db_search_time": 1.000}, 
    ]
    metrics = compute_vector_search_latency_metrics(query_logs)
    
    assert metrics["p50_ms"] == 10.0
    assert metrics["p95_ms"] > 800.0

def test_avg_tokens_per_query():
    # 1. Update this fake data to match what YOUR function actually expects
    mock_input = [
        "What is the revenue?", # Let's pretend this equals 5 tokens
        "Who is the CEO?"       # Let's pretend this equals 5 tokens
    ]
    
    # 2. Run the function
    result = compute_avg_tokens_per_query(mock_input)
    
    # 3. Assert whatever the exact math should be for your new mock input
    # Average of 5 and 5 is 5.0
    assert result == 5.0