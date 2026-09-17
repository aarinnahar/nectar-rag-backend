from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state
import numpy as np
import logging

logger = logging.getLogger("app") 


def calculate_chunk_distribution(state: AgentState):
    retrieved_chunks = state.get("retrieved_chunks", {})
    chunk_distributions = {}
    for strategy, data in retrieved_chunks.items():
        total_chunks = []
        max_size = [] 
        min_size = []
        avg_size = []
        chunk_distributions[strategy] = {}
        for item in data:
            chunk_lengths = [
        len(chunk.page_content) if hasattr(chunk, 'page_content') else len(chunk) 
        for chunk in item['retrieved_chunks']
    ]
            
                # Calculate your distribution parameters
            total_chunks.append(len(item['retrieved_chunks']))
            max_size.append(max(chunk_lengths)) 
            min_size.append(min(chunk_lengths))
            avg_size.append(int(np.mean(chunk_lengths)))

        chunk_distributions[strategy] = {"total_chunks" : np.mean(total_chunks),
                                         "max_size" : np.mean(max_size),
                                         "min_size" : np.mean(min_size),
                                         "avg_size" : np.mean(avg_size)}
    
    logger.debug(f"Chunk Distribution Completed")   
    save_state(filename= "chunks_distribution", data = chunk_distributions)
    return {"chunks_distribution": chunk_distributions}