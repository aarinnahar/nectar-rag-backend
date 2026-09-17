import time
import numpy as np
import logging
import faiss
import asyncio # Parallel execution kept intact!
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore import InMemoryDocstore

from src.embed_and_store.shared_embedder import get_shared_embedder

# Removed the old PyTorch-heavy embedding_model factory
from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state

logger = logging.getLogger("app") 

def build_single_vectorstore(strategy, chunks, model):
    """
    Helper function to build a single FAISS index. 
    Runs in a background thread for parallel execution natively in C++.
    """
    start_time = time.time()
    logger.info(f"Embed & Store started for {strategy} using FastEmbed (ONNX)")

    # 1. Generate embeddings (FastEmbed handles CPU batching natively)
    vectors = model.embed_documents(chunks)
    vectors = np.array(vectors).astype('float32')
    dimension = vectors.shape[1]  
    
    # 2. Populate FAISS C++ Index
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors) #type:ignore

    # 3. OPTIMIZATION: Use list/dict comprehensions instead of slow .append() loops
    documents = [
        Document(page_content=chunk, metadata={"chunk_id": i, "strategy": strategy})
        for i, chunk in enumerate(chunks)
    ]
    
    docstore_dict = {str(i): doc for i, doc in enumerate(documents)}
    index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

    # 4. Construct Docstore and Vectorstore
    docstore = InMemoryDocstore(docstore_dict)
    vectorstore = FAISS(
        embedding_function=model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    end_time = time.time()
    total = end_time - start_time
    
    return strategy, vectorstore, total


async def embed_and_store(state: AgentState) -> dict:
    """Sequential execution to keep memory strictly under 512MB."""
    logger.info("Starting sequential Embed & Store with FastEmbed...")
    
    # Instantiate once with single thread allocation
    model = get_shared_embedder()
    
    ingestion_time = {**state.get('ingestion_time', {})}
    chunk_factory = {**state.get('chunk_factory', {})}
    vectorstores = {}

    # Run sequentially instead of asyncio.gather to avoid RAM multiplication
    for strategy, chunks in chunk_factory.items():
        logger.info(f"Building vector store for: {strategy}")
        strat, vstore, time_taken = build_single_vectorstore(strategy, chunks, model)
        vectorstores[strat] = vstore
        ingestion_time[strat] = ingestion_time.get(strat, 0) + time_taken
        gc.collect()  # Immediately free intermediate allocations

    save_state(filename="total_ingestion_time", data=ingestion_time)
    return {"vectorstore": vectorstores, "ingestion_time": ingestion_time}
