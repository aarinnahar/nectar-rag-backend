import pytest
import json
import os
import asyncio
import sys
# 1. Get the absolute path to the root of your project (two levels up)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# 2. Add the root directory to Python's system path
if root_dir not in sys.path:
    sys.path.append(root_dir)
# -------------------------------------------------------------------
# IMPORT YOUR COMPILED LANGGRAPH WORKFLOW
# Replace 'your_graph_file' with the file where 'workflow' is defined
# -------------------------------------------------------------------
from src.orchestration.graph_builder import workflow 

def test_end_to_end_pipeline_html_generation():
    """
    E2E Test: Passes initial state to the async LangGraph workflow,
    verifies it generates an HTML file in the output folder, and 
    checks the HTML source code to ensure all strategies are present.
    """
    # 1. Define paths to your inputs and expected output
    json_file_path = r"C:\Users\RIYA\Downloads\chunk_updated\src\data\sample_docs\golden_dataset.json"
    pdf_file_path = r"C:\Users\RIYA\Downloads\chunk_updated\src\data\sample_docs\txt_document.pdf"
    
    # UPDATE THIS to match the exact name and folder your graph saves to
    output_html_path = r"F:\chunking_test\chunking_report.html" 
    
    # 2. File Check & Pre-Test Cleanup
    assert os.path.exists(json_file_path), f"Could not find {json_file_path}!"
    assert os.path.exists(pdf_file_path), f"Could not find {pdf_file_path}!"
    
    # CRITICAL: Remove the old HTML file so we don't get a "false positive" pass
    if os.path.exists(output_html_path):
        os.remove(output_html_path)
    
    # 3. Load the Golden Dataset
    with open(json_file_path, "r", encoding="utf-8") as file:
        golden_dataset = json.load(file)
        
    # 4. SET UP AND EXECUTE THE LANGGRAPH STATE
    initial_state = {
        "provider" : "Groq",
        "model_choice": "openai/gpt-oss-20b",
        "api_key" : "GROQ_API_KEY",
        "api_url" : "",
        "embed_provider" :"HuggingFace (Local)",
        "embed_api_url" : "",
        "embed_model_choice" :"sentence-transformers/all-MiniLM-L6-v2",
        "file_path": pdf_file_path,
        "golden_dataset": golden_dataset,
        "chunk_size": 1130,
        "chunk_overlap": 130,
        "file_name" : "text_document_testing_pdf" 
    }
    
    async def run_graph(state):
        return await workflow.ainvoke(state)
        
    try:
        # Run the graph and wait for it to finish saving the file
        asyncio.run(run_graph(initial_state))
    except Exception as e:
        pytest.fail(f"The LangGraph pipeline crashed! Error: {e}")
        
    # 5. VERIFY THE HTML FILE WAS ACTUALLY CREATED
    assert os.path.exists(output_html_path), "The workflow finished, but the HTML file was not saved to the output folder!"
    
    # 6. READ THE HTML AND VERIFY THE CONTENT
    with open(output_html_path, "r", encoding="utf-8") as html_file:
        html_content = html_file.read()
        
    # Verify the file isn't just an empty skeleton
    assert len(html_content) > 100, "The generated HTML file is empty!"
    
    # Verify all 4 chunking strategies made it into the final visual report
    expected_strategies = [
        "Semantic Text Splitter",
        "Recursive Character Text Splitter",
        "Token Text Splitter",
        "Character Text Splitter"
    ]
    
    for strategy in expected_strategies:
        assert strategy in html_content, f"The HTML report generated, but {strategy} is missing from the output!"