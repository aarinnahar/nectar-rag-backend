from src.orchestration.agent_state import AgentState
from src.models.models import embd_model_name
from src.llm.llm_client import get_llm
from src.config.settings import get_settings
from jinja2 import Template, FileSystemLoader, Environment
import logging
from pathlib import Path
import os
import re

logger = logging.getLogger(__name__)
settings = get_settings()

def get_stars(score):
    """Helper function to map index scores to 1-5 star ratings for the UI"""
    if score >= 90: return 5
    elif score >= 80: return 4
    elif score >= 70: return 3
    elif score >= 60: return 2
    return 1

def result_report_designer(state: AgentState):
    
    file_path = state['file_path']
    evaluation_scores = state.get("evaluation_scores", {})
    
    if not evaluation_scores:
        logger.warning("No evaluation scores found in state to analyze.")
        return {"evaluation_insights": {}}

    # 1. Sort strategies deterministically by index_score descending
    sorted_strategies = sorted(
        evaluation_scores.items(),
        key=lambda item: item[1].get("overall_index_score", 0.0),
        reverse=True
    )

    performance_metrics = state.get("performance_metrics")

    strategies_ranked = [s[0] for s in sorted_strategies]

    evaluation_insights = state['evaluation_insights']    
    llm_insights = state.get('llm_insights', {})    

    match = re.search(r"^(.*?)(?=/)", file_path)
    if match:
        file_name = match[0][::-1]
    else:
        file_name = file_path
        
    golden_dataset = state["golden_dataset"]
    gd_len = len(golden_dataset)
    embed_model_name = state['embed_model_choice']
    extracted_name = state['model_choice']
    
    
    winner = strategies_ranked[0]
    runner_up = strategies_ranked[1]
    middle = strategies_ranked[2]
    loser = strategies_ranked[3]

    from pathlib import Path

    # Dynamically finds the folder where this Python script lives
    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    path = ROOT_DIR / "report_design" / "report_template.html"
    path1 = ROOT_DIR / "report_design" / "scatterplot.html"
    path2 = ROOT_DIR / "report_design" / "bargraph.html"

    # 1. Setup the environment to look in the current folder
    file_loader = FileSystemLoader(path.parent)
    env = Environment(loader=file_loader)

    # 2. Load your "template.html" file
    template = env.get_template(path.name)

    # Simplified path handling for cleaner Python execution
    with open(path1, 'r', encoding='utf-8') as file:
        scatter = file.read()

    with open(path2, 'r', encoding='utf-8') as file:
        bar = file.read()

    splitters = {
        "recursive_text_splitter_chunks": "Recursive Character Text Splitter", 
        "character_text_splitter_chunks": "Character Text Splitter",
        "semantic_chunks": "Semantic Text Splitter",
        "token_text_splitter_chunks": "Token Text Splitter" 
    }
    
    # Helper for safely extracting dynamic failure annotations from the LLM payload
    failure_tags = llm_insights.get("failure_tags", {})

    context_data = {
        "source_document": state['file_name'],
        "retrieval_generation_graph": scatter,
        "quality_cost_graph": bar,
        "chunk_size": state['chunk_size'],
        "chunk_overlap": state['chunk_overlap'],
        "evaluation_baseline": f"Golden Dataset ({gd_len} Test Queries)",
        "target_llm": extracted_name,
        "embedding_model": embed_model_name,

        # ================================
        # 1. SUMMARY REPORT TAB
        # ================================

        "winner_name": splitters[winner],
        # Updated to use index_score
        "winner_accuracy": evaluation_scores[winner]['overall_index_score'],
        "recommendation_summary_text": llm_insights.get("recommendation_summary_text", ""),
        
        # Ranking Overview Sidebar (Podium Breakdown with explicit tags)
        "strategies_ranking": [
            {
                "name": splitters[winner], 
                "podium_status": "Winner", 
                "failure_tag": failure_tags.get(winner, ""), 
                "accuracy": evaluation_scores[winner]["overall_index_score"], 
                "stars": get_stars(evaluation_scores[winner]["overall_index_score"])
            },
            {
                "name": splitters[runner_up], 
                "podium_status": "Runner-Up", 
                "failure_tag": failure_tags.get(runner_up, ""), 
                "accuracy": evaluation_scores[runner_up]["overall_index_score"], 
                "stars": get_stars(evaluation_scores[runner_up]["overall_index_score"])
            },
            {
                "name": splitters[middle], 
                "podium_status": "Loser", 
                "failure_tag": failure_tags.get(middle, "Context Truncated"), 
                "accuracy": evaluation_scores[middle]["overall_index_score"], 
                "stars": get_stars(evaluation_scores[middle]["overall_index_score"])
            },
            {
                "name": splitters[loser], 
                "podium_status": "Loser", 
                "failure_tag": failure_tags.get(loser, "High Redundancy"), 
                "accuracy": evaluation_scores[loser]["overall_index_score"], 
                "stars": get_stars(evaluation_scores[loser]["overall_index_score"])
            }
        ],

        # ================================
        #    2. PERFORMANCE REPORT TAB
        # ================================
        
        # Comparison performance Table with structural metrics and system grades
        "performance_rows": [
            {
                "strategy": splitters[winner], 
                "retrieval_accuracy" : performance_metrics[winner].get("retrieval_accuracy", "A+"),
                "chunk_quality" : performance_metrics[winner].get("chunk_quality", 0),
                "p95_latency" : performance_metrics[winner].get("p95_latency", 0),
                "cost_per1K" : performance_metrics[winner].get("cost_per1K", 0),
                "system_grade": performance_metrics[winner].get("system_grade", "A+"),
                "index_score": performance_metrics[winner].get("index_score", 0)
            },
            {
                "strategy": splitters[runner_up], 
                "retrieval_accuracy" : performance_metrics[runner_up].get("retrieval_accuracy", "A+"),
                "chunk_quality" : performance_metrics[runner_up].get("chunk_quality", 0),
                "p95_latency" : performance_metrics[runner_up].get("p95_latency", 0),
                "cost_per1K" : performance_metrics[runner_up].get("cost_per1K", 0),
                "system_grade": performance_metrics[runner_up].get("system_grade", "A+"),
                "index_score": performance_metrics[runner_up].get("index_score", 0)
            },
            {
                "strategy": splitters[middle], 
                "retrieval_accuracy" : performance_metrics[middle].get("retrieval_accuracy", "A+"),
                "chunk_quality" : performance_metrics[middle].get("chunk_quality", 0),
                "p95_latency" : performance_metrics[middle].get("p95_latency", 0),
                "cost_per1K" : performance_metrics[middle].get("cost_per1K", 0),
                "system_grade": performance_metrics[middle].get("system_grade", "A+"),
                "index_score": performance_metrics[middle].get("index_score", 0)
            },
            {
                "strategy": splitters[loser], 
                "retrieval_accuracy" : performance_metrics[loser].get("retrieval_accuracy", "A+"),
                "chunk_quality" : performance_metrics[loser].get("chunk_quality", 0),
                "p95_latency" : performance_metrics[loser].get("p95_latency", 0),
                "cost_per1K" : performance_metrics[loser].get("cost_per1K", 0),
                "system_grade": performance_metrics[loser].get("system_grade", "A+"),
                "index_score": performance_metrics[loser].get("index_score", 0)
    
            }
        ],
                

        # ====================================
        #    3. GRANULAR DIAGNOSE REPORT TAB
        # ====================================
        

        # Comparison Matrix Table with structural metrics and system grades
        "diagnose_rows": [
            {
                "strategy": splitters[winner], 
                "status_color": "bg-emerald-500", 
                "context_recall": evaluation_scores[winner]["context_recall"], 
                "context_precision": evaluation_scores[winner]["context_precision"], 
                "hit_rate_at_k": evaluation_scores[winner]["hit_rate_at_k"], 
                "boundary_health": evaluation_scores[winner]["boundary_health"],
                "intra_chunk_coherence": evaluation_scores[winner]["intra_chunk_coherence"],
                "inter_chunk_redundancy": evaluation_scores[winner]["inter_chunk_redundancy"],
                "avg_vector_search_latency_ms": evaluation_scores[winner]["avg_vector_search_latency_ms"]['mean_ms'], 
                "avg_tokens_per_query": evaluation_scores[winner]["avg_tokens_per_query"], 
                "total_ingestion_time_s": evaluation_scores[winner]["total_ingestion_time_s"], 
                "badge_color_class": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
            },
            {
                "strategy": splitters[runner_up], 
                "status_color": "bg-emerald-500", 
                "context_recall": evaluation_scores[runner_up]["context_recall"], 
                "context_precision": evaluation_scores[runner_up]["context_precision"], 
                "hit_rate_at_k": evaluation_scores[runner_up]["hit_rate_at_k"], 
                "boundary_health": evaluation_scores[runner_up]["boundary_health"],
                "intra_chunk_coherence": evaluation_scores[runner_up]["intra_chunk_coherence"],
                "inter_chunk_redundancy": evaluation_scores[runner_up]["inter_chunk_redundancy"],
                "avg_vector_search_latency_ms": evaluation_scores[runner_up]["avg_vector_search_latency_ms"]['mean_ms'], 
                "avg_tokens_per_query": evaluation_scores[runner_up]["avg_tokens_per_query"], 
                "total_ingestion_time_s": evaluation_scores[runner_up]["total_ingestion_time_s"], 
                "badge_color_class": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
            },

            {
                "strategy": splitters[middle], 
                "status_color": "bg-emerald-500", 
                "context_recall": evaluation_scores[middle]["context_recall"], 
                "context_precision": evaluation_scores[middle]["context_precision"], 
                "hit_rate_at_k": evaluation_scores[middle]["hit_rate_at_k"], 
                "boundary_health": evaluation_scores[middle]["boundary_health"],
                "intra_chunk_coherence": evaluation_scores[middle]["intra_chunk_coherence"],
                "inter_chunk_redundancy": evaluation_scores[middle]["inter_chunk_redundancy"],
                "avg_vector_search_latency_ms": evaluation_scores[middle]["avg_vector_search_latency_ms"]['mean_ms'], 
                "avg_tokens_per_query": evaluation_scores[middle]["avg_tokens_per_query"], 
                "total_ingestion_time_s": evaluation_scores[middle]["total_ingestion_time_s"], 
                "badge_color_class": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
            
            },
            {
                "strategy": splitters[loser], 
                "status_color": "bg-emerald-500", 
                "context_recall": evaluation_scores[loser]["context_recall"], 
                "context_precision": evaluation_scores[loser]["context_precision"], 
                "hit_rate_at_k": evaluation_scores[loser]["hit_rate_at_k"], 
                "boundary_health": evaluation_scores[loser]["boundary_health"],
                "intra_chunk_coherence": evaluation_scores[loser]["intra_chunk_coherence"],
                "inter_chunk_redundancy": evaluation_scores[loser]["inter_chunk_redundancy"],
                "avg_vector_search_latency_ms": evaluation_scores[loser]["avg_vector_search_latency_ms"]['mean_ms'], 
                "avg_tokens_per_query": evaluation_scores[loser]["avg_tokens_per_query"], 
                "total_ingestion_time_s": evaluation_scores[loser]["total_ingestion_time_s"], 
                "badge_color_class": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                        
                }
        ],

        # ====================================
        #    3. DEEP-DIVE PERFORMANCE INSIGHTS REPORT TAB
        # ====================================

        # Deterministic Rationale Cards Integration
        "llm_payload": {
            "winner_rationale": {
                "headline": llm_insights.get("winner_rationale", {}).get("headline", ""),
                "body": llm_insights.get("winner_rationale", {}).get("body", ""),
                "hidden_story": llm_insights.get("winner_rationale", {}).get("hidden_story", "")
            },
            "runner_up_rationale": {
                "headline": llm_insights.get("runner_up_rationale", {}).get("headline", ""),
                "body": llm_insights.get("runner_up_rationale", {}).get("body", ""),
                "hidden_story": llm_insights.get("runner_up_rationale", {}).get("hidden_story", "")
            },
            "loser_rationale": {
                "headline": llm_insights.get("loser_rationale", {}).get("headline", ""),
                "body": llm_insights.get("loser_rationale", {}).get("body", ""),
                "hidden_story": llm_insights.get("loser_rationale", {}).get("hidden_story", "")
            }
        }
    }

    rendered_html = template.render(context_data) 

    # 1. Set your directory
    output_dir = r"F:\chunking_test"
    file_name = "chunking_report.html"

    # 2. CREATE the folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 3. Combine them into a full path
    final_destination = os.path.join(output_dir, file_name)

    # 4. Save the file
    with open(final_destination, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print("Success! Chunking Report saved ")

    logger.debug(f"Chunking Report generated Successfully to: {final_destination}")

    return {}
































# from src.orchestration.agent_state import AgentState
# from src.models.models import embd_model_name
# from src.llm.llm_client import get_llm
# from src.config.settings import get_settings
# from jinja2 import Template, FileSystemLoader, Environment
# import logging
# from pathlib import Path
# import os
# import re
# logger = logging.getLogger(__name__)

# settings = get_settings()

# # This converts the string into a proper Path object
# def result_report_designer(state: AgentState):
   
#     file_path = state['file_path']
#     top_scorers = state['top_scorers']
#     llm_insights = state['llm_insights']
#     match = re.search(r"^(.*?)(?=/)", file_path)
#     if match:
#         file_name = match[0][::-1]
#     else:
#         file_name = file_path
#     golden_dataset = state["golden_dataset"]
#     gd_len = len(golden_dataset)
#     embed_model_name = embd_model_name(state['embed_provider'])
#     llm = get_llm()
#     extracted_name = llm.model
#     efficiency_scores = state['efficiency_scores']
#     combined_matrics = state['combined_matrics']
    
#     winner = top_scorers[0][0]
#     runner_up = top_scorers[1][0]
#     middle = top_scorers[2][0]
#     loser = top_scorers[3][0]

#     path = Path(r"C:\Users\RYZEN\Downloads\Chunking_Eval\Chunking_Eval\src\report_design\template.html")
#     path1 = Path(r"C:\Users\RYZEN\Downloads\Chunking_Eval\Chunking_Eval\src\report_design\scatterplot.html")
#     path2 = Path(r"C:\Users\RYZEN\Downloads\Chunking_Eval\Chunking_Eval\src\report_design\bargraph.html")


#     # 1. Setup the environment to look in the current folder
#     file_loader = FileSystemLoader(path.parent)
#     env = Environment(loader=file_loader)

#     # 2. Load your "template.html" file
#     template = env.get_template(path.name)

#     with open(path1._raw_paths[0], 'r',encoding='utf-8') as file:
#         scatter = file.read()

#     with open(path2._raw_paths[0], 'r',encoding='utf-8') as file:
#         bar = file.read()

#     glance_accuracy = {}
#     for strategy, data in combined_matrics.items():
        
#         if data['overall_accuracy'] >= 90:
#             glance_accuracy[strategy] = {"accuracy" :data['overall_accuracy'],
#                                          "badge_text" : "BEST ROI",
#                                           "stars" :  5}


#         elif data['overall_accuracy'] >= 85 and data['overall_accuracy'] <= 90:
#             glance_accuracy[strategy] = {"accuracy" :data['overall_accuracy'],
#                                          "badge_text" : "Balanced",
#                                           "stars" :  4}
        
#         elif data['overall_accuracy'] >= 60 and data['overall_accuracy'] <= 85:
#             glance_accuracy[strategy] = {"accuracy" :data['overall_accuracy'],
#                                          "badge_text" : "Too Expensive & Slow",
#                                           "stars" :  2}
        
#         else:
#             glance_accuracy[strategy] = {"accuracy" :data['overall_accuracy'],
#                                          "badge_text" : "Context is Cut Off",
#                                           "stars" :  1}

#     splitters = {"recursive_text_splitter_chunks" : "Recursive Character Text Splitter", 
#                  "character_text_splitter_chunks" : "Character Text Splitter",
#                  "semantic_chunks" : "Semantic Text Splitter",
#                   "token_text_splitter_chunks" : "Token Text Splitter" }         
#     context_data = {
#     "source_document": state['file_name'],
#     "retrieval_generation_graph" : scatter,
#     "quality_cost_graph" : bar,
#     "chunk_size": state['chunk_size'],
#     "chunk_overlap": state['chunk_overlap'],
#     "evaluation_baseline": f"Golden Dataset ({gd_len} Test Queries)",
#     "target_llm": extracted_name,
#     "embedding_model": embed_model_name,
#     "winner_name": splitters[winner],
#     "runner_up_name": splitters[runner_up],
#     "loser_name": splitters[loser],
#     "winner_accuracy": combined_matrics[winner]['overall_accuracy'],
#     "recommendation_summary_text": llm_insights["recommendation_summary_text"],
    
#     # Ranking Overview Sidebar data list
#     "strategies_ranking": [
#         {"name": splitters[winner], "badge_text": glance_accuracy[winner]["badge_text"], "accuracy": glance_accuracy[winner]["accuracy"], "stars": glance_accuracy[winner]["stars"]},
#         {"name": splitters[runner_up], "badge_text": glance_accuracy[runner_up]["badge_text"], "accuracy": glance_accuracy[runner_up]["accuracy"], "stars": glance_accuracy[runner_up]["stars"]},
#         {"name": splitters[middle], "badge_text": glance_accuracy[middle]["badge_text"], "accuracy": glance_accuracy[middle]["accuracy"], "stars": glance_accuracy[middle]["stars"]},
#         {"name": splitters[loser], "badge_text": glance_accuracy[loser]["badge_text"], "accuracy": glance_accuracy[loser]["accuracy"], "stars": glance_accuracy[loser]["stars"]}
#     ],
    
#     # Comparison Matrix Table loop data
#     "matrix_rows": [
#         {
#             "name": splitters[winner], "status_color": "bg-emerald-500", 
#             "context_recall": combined_matrics[winner]["context_recall"], "faithfulness": combined_matrics[winner]["faithfullness"], "latency": combined_matrics[winner]["latency_seconds"], 
#             "tokens_per_query": combined_matrics[winner]["total_tokens"], "premium": str(combined_matrics[winner]["premium_cost"]), "fast": str(combined_matrics[winner]["fast_cost"]), "open": str(combined_matrics[winner]["open_cost"]), "grade": "A+ (Winner)",
#             "badge_color_class": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20", "is_latency_bad": False
#         },
#         {
#             "name": splitters[runner_up], "status_color": "bg-amber-400", 
#             "context_recall": combined_matrics[runner_up]["context_recall"], "faithfulness": combined_matrics[runner_up]["faithfullness"], "latency": combined_matrics[runner_up]["latency_seconds"], 
#             "tokens_per_query": combined_matrics[runner_up]["total_tokens"], "premium": str(combined_matrics[runner_up]["premium_cost"]), "fast": str(combined_matrics[runner_up]["fast_cost"]), "open": str(combined_matrics[runner_up]["open_cost"]), "grade": "B",
#             "badge_color_class": "bg-slate-700 text-slate-300", "is_latency_bad": False
#         },
#         {
#             "name": splitters[middle], "status_color": "bg-rose-500", 
#             "context_recall": combined_matrics[middle]["context_recall"], "faithfulness": combined_matrics[middle]["faithfullness"], "latency": combined_matrics[middle]["latency_seconds"], 
#             "tokens_per_query": combined_matrics[middle]["total_tokens"], "premium": str(combined_matrics[middle]["premium_cost"]), "fast": str(combined_matrics[middle]["fast_cost"]), "open": str(combined_matrics[middle]["open_cost"]), "grade": "C-",
#             "badge_color_class": "bg-rose-500/10 text-rose-400 border border-rose-500/20", "is_latency_bad": True
#         },
#         {
#             "name": splitters[loser], "status_color": "bg-rose-700", 
#             "context_recall": combined_matrics[loser]["context_recall"], "faithfulness": combined_matrics[loser]["faithfullness"], "latency": combined_matrics[loser]["latency_seconds"], 
#             "tokens_per_query": combined_matrics[loser]["total_tokens"], "premium": str(combined_matrics[loser]["premium_cost"]), "fast": str(combined_matrics[loser]["fast_cost"]), "open": str(combined_matrics[loser]["open_cost"]), "grade": "F (Fails)",
#             "badge_color_class": "bg-rose-900/40 text-rose-500 border border-rose-900/50", "is_latency_bad": False
#         }
#     ],
    
#     # ─── YOUR PYDANTIC SCHEMA MATCHED OBJECTS ───
#     "llm_payload": {
#         "retrieval_card": {
#             "headline": llm_insights["retrieval_card"]["headline"],
#             "body": llm_insights["retrieval_card"]["body"],
#             "hidden_story": llm_insights["retrieval_card"]["hidden_story"]
#         },
#         "generation_card": {
#             "headline": llm_insights["generation_card"]["headline"],
#             "body": llm_insights["generation_card"]["body"],
#             "hidden_story": llm_insights["generation_card"]["hidden_story"]
#         },
#         "efficiency_card": {
#             "headline": llm_insights["efficiency_card"]["headline"],
#             "body": llm_insights["efficiency_card"]["body"],
#             "hidden_story": llm_insights["efficiency_card"]["hidden_story"]
#         }
#     }
# }



#     rendered_html = template.render(context_data) 

#     # 1. Set your directory
#     output_dir = r"D:\Practive Projects\Chunking_Eval\src\output"
#     file_name = f"chunking_report.html"

#     # 2. CREATE the folder if it doesn't exist (the magic line)
#     os.makedirs(output_dir, exist_ok=True)

#     # 3. Combine them into a full path
#     final_destination = os.path.join(output_dir, file_name)

#     # 4. Save the file
#     with open(final_destination, "w", encoding="utf-8") as f:
#         f.write(rendered_html)

#     print(f"Success! Chunking Report saved ")

#     logger.debug("Chunking Report generated Successfully to: {final_destination}")


#     return {}
