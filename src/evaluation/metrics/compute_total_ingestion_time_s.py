from typing import Dict, Any

def compute_total_ingestion_time_s(state: Dict[str, Any], strategy: str) -> float:
    """
    Extracts the total ingestion time (in seconds) for a specific chunking strategy 
    from the LangGraph state. 
    
    This metric represents the time it took to chunk the document, run the 
    SentenceTransformer model to create embeddings, and populate the FAISS index.

    :param state: The full LangGraph AgentState dictionary.
    :param strategy: The string name of the chunking strategy (e.g., 'character_text_splitter').
    :return: The total ingestion time in seconds (float), rounded to 2 decimal places.
    """
    if not state or not isinstance(state, dict):
        return 0.0

    # Extract the ingestion_time dictionary from state
    ingestion_time_dict = state.get("ingestion_time", {})
    
    # Fetch the specific time for this strategy, defaulting to 0.0 if missing
    total_time_sec = ingestion_time_dict.get(strategy, 0.0)

    return round(float(total_time_sec), 2)