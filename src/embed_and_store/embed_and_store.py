import time
import numpy as np
import logging
import faiss
import asyncio # Parallel execution kept intact!
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore import InMemoryDocstore

# 1. IMPORT FASTEMBED (The ONNX Magic - Zero PyTorch!)
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

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
    """
    Asynchronous LangGraph node that builds multiple FAISS databases concurrently.
    Now optimized for 512MB RAM cloud tiers.
    """
    # 2. INSTANTIATE FASTEMBED DIRECTLY
    # This model is quantized to INT8, runs natively on CPU, and costs ~150MB RAM.
    logger.info("Loading lightweight FastEmbed ONNX runtime...")
    model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Safely get state variables
    ingestion_time = {**state.get('ingestion_time', {})}
    chunk_factory = {**state.get('chunk_factory', {})}
    vectorstores = {}

    # Dispatch all 4 vector databases to build concurrently in background threads
    tasks = [
        asyncio.to_thread(build_single_vectorstore, strategy, chunks, model)
        for strategy, chunks in chunk_factory.items()
    ]
    
    # Wait for all of them to finish at the exact same time
    results = await asyncio.gather(*tasks)

    # Reassemble the results into the state dictionaries
    for strategy, vectorstore, total in results:
        vectorstores[strategy] = vectorstore
        ingestion_time[strategy] = ingestion_time.get(strategy, 0) + total

    logger.info("Embed & Store Completed (Parallel Execution with FastEmbed)")
    save_state(filename="total_ingestion_time", data=ingestion_time)
    
    return {"vectorstore": vectorstores, "ingestion_time": ingestion_time}