import time
import logging
from typing import Dict, Any, List
from src.utils.save_state import save_state


logger = logging.getLogger(__name__)

def retrieval(state: Dict[str, Any]):
    vectorstore = state["vectorstore"]
    golden_dataset = state["golden_dataset"]
    k = 4

    retrieved_chunks = {}

    for strategy, vstore in vectorstore.items():
        retrieved_chunks[strategy] = []
        logger.info(f"RETRIEVER NODE STARTS EVALUATION strategy: {strategy}")

        # Create retriever for current strategy
        retriever = vstore.as_retriever(search_kwargs={"k": k})

        for item in golden_dataset:
            query = item["query"]
            answer = item["answer"]

            start = time.perf_counter()
            # Fetch LangChain Document objects
            docs = retriever.invoke(query)
            end = time.perf_counter()

            total_time = end - start
            logger.info(f"Query: '{query}' | Strategy: {strategy} | Chunks retrieved: {len(docs)}")

            # Convert LangChain Documents into clean, JSON-serializable dictionaries
            serialized_chunks = []
            for idx, doc in enumerate(docs):
                serialized_chunks.append({
                    # Extract ID from metadata if set during indexing, or fallback to index position
                    "chunk_id": doc.metadata.get("chunk_id", idx),
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

            data = {
                "question": query,
                "answer": answer,
                "vector_db_search_time": total_time,
                "retrieved_chunks": serialized_chunks
            }
            retrieved_chunks[strategy].append(data)

    logger.info("RETRIEVAL FINISHED!")
    
    # Safely save state to JSON (now 100% serializable dicts)
    save_state(filename="retrieved_chunks", data=retrieved_chunks)
    logger.debug("Retrieval Completed")

    return {"retrieved_chunks": retrieved_chunks}