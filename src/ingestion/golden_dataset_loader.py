from src.orchestration.agent_state import AgentState
import logging
import json
logger = logging.getLogger("app") 

def golden_dataset_loader(state:AgentState):
    logger.info("GOLDEN DATASET LOADER STARTED")
    path = "d:\\Practive Projects\\Chunking_Eval\\src\\data\\sample_docs\\golden_dataset.json"
    with open(path, "r") as file:
        golden_data = json.load(file)
        
    return {"golden_dataset" : golden_data}
