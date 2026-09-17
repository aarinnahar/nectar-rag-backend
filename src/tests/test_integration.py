import pytest
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys
# 1. Get the absolute path to the root of your project (two levels up)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# 2. Add the root directory to Python's system path
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import your actual metric function
from src.evaluation.metrics.compute_retrieval_metrics import compute_retrieval_metrics

@pytest.fixture(scope="module")
def embedding_function():
    """Loads a real LangChain-compatible embedder for the vector DB."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def test_full_retrieval_pipeline_integration(embedding_function):
    """
    INTEGRATION TEST: Raw Text -> Chunker -> FAISS -> Evaluator
    """
    # 1. The Raw Input (Simulating a small uploaded PDF)
    raw_document = """
    Aarin is actively preparing to transition into a Generative AI engineering role. 
    He is currently enrolled in a Data Science training program. 
    He has built an automated AI newsletter engine utilizing LangGraph.
    """
    
    golden_target = "Aarin built an AI newsletter engine using LangGraph."
    golden_query = "What kind of project did Aarin build with LangGraph?"

    # 2. THE CHUNKER: Let the actual strategy do the work
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    docs = splitter.create_documents([raw_document])
    
    # 3. THE VECTOR DB: Build a real, temporary FAISS index in memory
    vector_db = FAISS.from_documents(docs, embedding_function)
    
    # 4. THE RETRIEVER: Execute a real search
    raw_retrieved_docs = vector_db.similarity_search(golden_query, k=2)
    
    # Format the output so your evaluator can read it
    retrieved_chunks = [{"text": doc.page_content} for doc in raw_retrieved_docs]

    # 5. THE EVALUATOR: Grade the actual pipeline output
    # (We pass the underlying SentenceTransformer model for the evaluator's internal math)
    eval_model = embedding_function.client 
    metrics = compute_retrieval_metrics(retrieved_chunks, golden_target, eval_model)

    # 6. THE ASSERTION: Does the pipeline successfully flow and find data?
    # We assert > 0.0 because the system successfully chunked, stored, and retrieved relevant context.
    assert metrics["context_recall"] > 0.0, "Integration pipeline failed to retrieve the target context!"
    assert len(retrieved_chunks) == 2, "Pipeline failed to return exactly K chunks."