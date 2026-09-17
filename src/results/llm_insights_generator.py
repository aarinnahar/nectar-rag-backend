from src.orchestration.agent_state import AgentState
from src.prompt_templates.llm_insights_generation import LLM_INSIGHTS_GENERATION_PROMPT
from src.models.schemas.llm_insight_generation_schema import EvaluationReport
from src.utils.save_state import save_state, load_state
from langchain_core.prompts import PromptTemplate
from src.llm.llm_client import get_llm
import logging
logger = logging.getLogger("app") 
import os
import json

async def llm_insights_generator(state : AgentState):
    # if os.path.exists("src/utils/data/llm_insights_generator.json"):
    #     logger.info("Skipping big_story_section_writer")
    #     with open("src/utils/data/llm_insights_generator.json", encoding="utf-8") as f:
    #         return {"llm_insights" :json.load(f)}
    evaluation_insights = {**state['evaluation_insights']}
    evaluation_scores = state.get("evaluation_scores", {})

    if not evaluation_scores:
        logger.warning("No evaluation scores found in state to analyze.")
        return {"evaluation_insights": {}}

    # 1. Sort strategies deterministically by overall_index_score descending
    sorted_strategies = sorted(
        evaluation_scores.items(),
        key=lambda item: item[1].get("overall_index_score", 0.0),
        reverse=True
    )

    strategies_ranked = [s[0] for s in sorted_strategies]
    
    winner = strategies_ranked[0] if len(strategies_ranked) > 0 else "None"
    runner_up = strategies_ranked[1] if len(strategies_ranked) > 1 else None
    loser = strategies_ranked[-1] if len(strategies_ranked) > 2 else None

    splitters = {"recursive_text_splitter_chunks" : "Recursive Character Text Splitter", 
                 "character_text_splitter_chunks" : "Character Text Splitter",
                 "semantic_chunks" : "Semantic Text Splitter",
                  "token_text_splitter_chunks" : "Token Text Splitter" }  
    llm_insights = {}

    prompt = PromptTemplate(template= LLM_INSIGHTS_GENERATION_PROMPT, input_variables= ['winner_name', 
                                                                                        "calculated_insights",
                                                                                        'runner_up_name',
                                                                                        "loser_name"])
    llm = get_llm(provider=state['provider'], model_choice=state["model_choice"], api_key=state['api_key'], api_url=state['api_url'])
    structured_llm = llm.with_structured_output(schema=EvaluationReport, include_raw= True)

    main_prompt = prompt.format(winner_name= splitters[winner], 
                                runner_up_name = splitters[runner_up],
                                loser_name = splitters[loser],
                                calculated_insights=evaluation_insights)
    
    message = main_prompt 
    for trial in range(3):
        logger.info(f'Trial run {trial + 1} in llm_insights_generator')
        response = await structured_llm.ainvoke(message)
        if response['parsing_error']:
            logger.warning(f"Parsing error on llm_insights_generator trial {trial + 1}: {response['parsing_error']}")
            # Append the error TO THE MAIN PROMPT so it remembers the rules
            message = main_prompt + f"\n\nSYSTEM WARNING - FIX PREVIOUS ERROR: {response['parsing_error']}"
        else:
            llm_insights = {"winner_rationale" : {"headline" : response['parsed'].winner_rationale.headline,
                                        "body" : response['parsed'].winner_rationale.body,
                                        "hidden_story" : response['parsed'].winner_rationale.hidden_story},
                    "runner_up_rationale" :{"headline" : response['parsed'].runner_up_rationale.headline,
                                        "body" : response['parsed'].runner_up_rationale.body,
                                        "hidden_story" : response['parsed'].runner_up_rationale.hidden_story},
                    "loser_rationale" :{"headline" : response['parsed'].loser_rationale.headline,
                                        "body" : response['parsed'].loser_rationale.body,
                                        "hidden_story" : response['parsed'].loser_rationale.hidden_story},
                    "failure_tags" :response['parsed'].failure_tags,
                    "recommendation_summary_text" : response['parsed'].recommendation_summary_text}
            
            break
    else:
        # This 'else' triggers ONLY if the loop finishes without a 'break' (meaning it failed 3 times)
        logger.error(f"Failed to get a valid score for llm_insights_generator after 3 attempts. Assigning 0.")

    logger.debug(f"llm_insights_generator Completed")
    save_state(filename= "llm_insights_generator", data = llm_insights)
    return {"llm_insights" : llm_insights}










