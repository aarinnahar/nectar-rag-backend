import sys
import os
import traceback
import pandas as pd
from typing import cast
import asyncio
# 1. Get the absolute path to the root of your project (two levels up)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# 2. Add the root directory to Python's system path
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.orchestration.graph_builder import workflow
from src.config.settings import Settings
from src.orchestration.agent_state import AgentState

import tempfile
import json
import streamlit as st
import logging
logger = logging.getLogger("app") 

settings = Settings()

st.title('🎯 RAG Evaluation Dashboard')

st.write("Welcome to the pipeline")

st.header("1. Upload Evaluation Docs")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Source Document", type = ['pdf'])

with col2:
    uploaded_json = st.file_uploader("Upload Golden QA Dataset", type = ['json'])

if uploaded_json and uploaded_file:
    st.success("✅ Files loaded successfully! Ready for configuration.")
else:
    st.info("Please upload both a PDF and your QA JSON file to proceed.")



st.sidebar.header("⚙️ Pipeline Settings")
chunk_size = st.sidebar.slider("Chunk Size (Tokens/Characters)", min_value=100, max_value= 2000, value = settings.chunk_size, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value= 500, value = settings.chunk_overlap, step=10)


st.sidebar.markdown("---")
st.header("2. Execute Pipeline")
# st.sidebar.write("Evaluation Strategies:")

# st.sidebar.checkbox("Character Splitter", value= True, disabled= True)
# st.sidebar.checkbox("Recursive Splitter", value= True, disabled= True)
# st.sidebar.checkbox("Token Splitter", value= True, disabled= True)
# st.sidebar.checkbox("Semantic Splitter", value= True, disabled= True)
st.sidebar.subheader("LLM Configuration")

# 1. Provider Selection now includes Ollama
provider = st.sidebar.selectbox("Select Provider", ["OpenAI", "Groq", "Ollama (Local)"])

if provider == "Ollama (Local)":
    st.sidebar.info("""
        **Prerequisites for Ollama:**
        1. Install Ollama on your computer.
        2. Open your terminal and run: `ollama pull llama3.1`
        3. Make sure Ollama is actively running.
        """)
    # For Ollama, we don't need an API key, we need their local URL
    api_url = st.sidebar.text_input("Ollama Endpoint", value="http://localhost:11434")
    model_choice = st.sidebar.text_input("Local Model Name", value="llama3.1", help="Enter the exact name of the model you pulled in Ollama.")
    
    st.info("💡 Ensure Ollama is running on your machine in the background.")
    api_key = "ollama" # Just a dummy key since Ollama doesn't require one

else:
    # Standard BYOK flow for cloud providers
    api_key = st.sidebar.text_input(f"Enter your {provider} API Key", type="password")
    api_url = ''
    if provider == "OpenAI":
        model_choice = st.sidebar.selectbox("Select Model", ["gpt-4o-mini", "gpt-4o"])
    elif provider == "Groq":
        model_choice = st.sidebar.selectbox("Select Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"])

st.sidebar.divider() # Adds a nice visual line to separate the sections
st.sidebar.subheader("Embedding Configuration")

# 1. Select Embedding Provider
embed_provider = st.sidebar.selectbox(
    "Select Embedding Provider", 
    ["OpenAI", "HuggingFace (Local)", "Ollama (Local)"]
)


if embed_provider == "OpenAI":
    # Checkbox to reuse the key from the LLM section
    embed_api_url = ''
    use_same_key = st.sidebar.checkbox("Use same OpenAI API Key", value=True)
    if use_same_key:
        # Reuses the key typed in the LLM section
        embed_api_key = api_key 
    else:
        # Option to enter a different key if needed
        embed_api_key = st.sidebar.text_input("OpenAI API Key for Embeddings", type="password")
        
    embed_model_choice = st.sidebar.selectbox(
        "Select Embedding Model", 
        ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
    )
elif embed_provider == "HuggingFace (Local)":
    embed_api_url = ''
    st.sidebar.info("💡 HuggingFace embeddings run directly in memory on your machine. No API key needed!")
    embed_model_choice = st.sidebar.text_input(
        "HuggingFace Model Name", 
        value="sentence-transformers/all-MiniLM-L6-v2"  
    )

elif embed_provider == "Ollama (Local)":
    st.sidebar.caption("Terminal command: `ollama pull nomic-embed-text`")
    embed_api_url = st.sidebar.text_input("Ollama Endpoint (Embeddings)", value="http://localhost:11434")
    embed_model_choice = st.sidebar.text_input(
        "Ollama Embedding Model", 
        value="nomic-embed-text", 
        help="Ensure you have pulled this model using 'ollama pull nomic-embed-text'."
    )
st.markdown("---")

if st.button("🚀 Run Evaluation Pipeline", use_container_width= True):

    if uploaded_file is not None:
        # Calculate file size in bytes (15 MB = 15 * 1024 * 1024 bytes)
        max_size_bytes = 15 * 1024 * 1024 
        
        # Check if the file size exceeds the limit
        if uploaded_file.size > max_size_bytes:
            st.error("File size exceeds 15MB Limit")
            st.stop() # Stops execution so the rest of the code doesn't run

    if not uploaded_file or not uploaded_json:
        st.error("Hold on! You need to upload both files first.")

    else:
        actual_file_name = uploaded_file.name
        with st.spinner("Running chunking strategies and LLM evaluation... This may take a minute."):
            df_results = None
            try:
                # 1. Parse the JSON file directly from RAM
                golden_data = json.load(uploaded_json)
                
                # 2. Save the PDF to a temporary file so pdfplumber can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_pdf_path = tmp_file.name

                # 3. Create the Initial State to pass to LangGraph
                initial_state = cast(AgentState,{
                    "provider" : provider,
                    "model_choice": model_choice,
                    "api_key" : api_key,
                    "api_url" : api_url,
                    "embed_provider" :embed_provider,
                    "embed_api_url" : embed_api_url,
                    "embed_model_choice" :embed_model_choice,
                    "file_path": temp_pdf_path,
                    "golden_dataset": golden_data,
                    # We can pass the Streamlit slider values directly to your Settings or State here!
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "file_name" : actual_file_name})

                # 4. 🔥 EXECUTE THE BACKEND PIPELINE 🔥
                async def run_pipeline_async(workflow, initial_state, status):
                    async for chunk in workflow.astream(
                        initial_state,
                        stream_mode="updates"
                    ):
                        for node_name, state_update in chunk.items():
                            status.write(f"✅ Finished node: **{node_name}**")
                            status.update(
                                label=f"Currently running: {node_name}..."
                            )

                
                with st.status("Starting Pipeline...", expanded=True) as status:

                    asyncio.run(
                        run_pipeline_async(
                            workflow,
                            initial_state,
                            status
                        )
                    )

                    status.update(
                        label="Pipeline Complete!",
                        state="complete",
                        expanded=False
                    )

                # 5. Clean up the temporary file
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

                                                
               
            except Exception as e:
                # If your backend crashes, it will show the error elegantly on the UI instead of breaking the app
                st.error(f"Pipeline failed: {traceback.format_exc()}")
            # 🛑 BREAK OUT OF THE SPINNER BLOCK HERE 🛑
            # Notice the indentation! We are back in the main column.
            
            st.success("Pipeline Execution Complete!")
            st.header("📊 Evaluation Results")
            
            with open(r"F:\chunking_test\chunking_report.html", "r", encoding="utf-8") as f:
                html_data = f.read()

            st.download_button(
                label="📥 Download Dashboard to View",
                data=html_data,
                file_name="chunking_report.html",
                mime="text/html"
)