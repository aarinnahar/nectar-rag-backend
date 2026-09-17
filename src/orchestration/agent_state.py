from typing import TypedDict, List, Dict, Any
from langchain_community.vectorstores import FAISS


class AgentState(TypedDict):
    provider : str
    model_choice : str
    api_key : str
    api_url : str
    embed_provider :str
    embed_api_url : str
    embed_model_choice :str
    file_path : str
    file_name : str
    text_markdown_path : str
    chunk_size: int
    chunk_overlap: int
    chunk_factory : Dict[str, Any]
    score : Dict[str, Any]
    text : str
    golden_dataset: List[Dict[str, str]]
    final_evaluation : Dict[str, Any]
    vectorstore: Dict[str, Any]
    
    failed_strategies : List[str]
    faithfullness_generator : dict[str, list]

    retrieved_chunks : dict[str, list]
    proxy_scores : dict[str, Any]

    recall_precision_faithfullness_scores : Dict[str, Any]
    tokens_latency : dict[str, Any]

    chunks_distribution : dict[str, Any]
    ingestion_time : dict[str, Any]

    efficiency_scores : dict[str, Any]

    combined_matrics : dict[str, Any]

    llm_json_payload : dict[str, Any]
    
    top_scorers : list
    final_report : Any

    llm_insights : dict[str, Any]
    evaluation_insights : dict[str, Any]

    performance_metrics : dict[str, Any]
    evaluation_scores : dict[str, Any]