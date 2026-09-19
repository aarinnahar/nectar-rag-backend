import logging
import asyncio
from src.utils.logging import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END
from src.orchestration.agent_state import AgentState

from src.ingestion.text_loader import text_loader
from src.chunking.chunking_factory import chunking_factory
from src.embed_and_store.embed_and_store import embed_and_store
from src.retrieval.retireval import retrieval
from src.evaluation.evaluator import evaluate_retrieved_chunks
from src.evaluation.chunk_distribution import calculate_chunk_distribution
from src.evaluation.generate_evaluation_insights import generate_evaluation_insights
from src.results.llm_insights_generator import llm_insights_generator
from src.report_design.graphs import create_graphs
from src.results.result_report_generator import result_report_designer

graph = StateGraph(AgentState)

graph.add_node("text_loader", text_loader)
graph.add_node("chunking_factory", chunking_factory)
graph.add_node("embed_and_store", embed_and_store)
graph.add_node("retrieval", retrieval) #type:ignore
graph.add_node("evaluate_retrieved_chunks", evaluate_retrieved_chunks)
graph.add_node("calculate_chunk_distribution", calculate_chunk_distribution)
graph.add_node("generate_evaluation_insights", generate_evaluation_insights)
graph.add_node("llm_insights_generator", llm_insights_generator)
graph.add_node("create_graphs", create_graphs)
graph.add_node("result_report_designer", result_report_designer)

graph.add_edge(START,"text_loader")
graph.add_edge("text_loader","chunking_factory")
graph.add_edge("chunking_factory","embed_and_store")
graph.add_edge("embed_and_store","retrieval")
graph.add_edge("retrieval","evaluate_retrieved_chunks")
graph.add_edge("evaluate_retrieved_chunks","calculate_chunk_distribution")
graph.add_edge("calculate_chunk_distribution","generate_evaluation_insights")
graph.add_edge("generate_evaluation_insights","llm_insights_generator")
graph.add_edge("llm_insights_generator","create_graphs")
graph.add_edge("create_graphs","result_report_designer")

graph.add_edge("result_report_designer", END)

workflow = graph.compile()


# Change this function in graph_builder.py
async def run_evaluator(agent_state_input: dict):
    # Yields a dictionary mapping the node name to its state output
    async for chunk in graph.astream(agent_state_input, stream_mode="updates"):
        yield chunk

