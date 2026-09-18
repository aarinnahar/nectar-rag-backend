import os
import json

# MUST BE AT THE VERY TOP: Locks memory allocation to prevent OOM crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_MAX_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import fitz  # PyMuPDF
import zipfile
import io
import re
from pathlib import Path

# Import your LangGraph entry point here
from src.orchestration.graph_builder import run_evaluator 

app = FastAPI(title="RAG Evaluator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

def get_office_page_count(content: bytes, ext: str) -> int:
    """Extracts page/slide counts from DOCX and PPTX metadata without rendering."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(content), "r")
        app_xml = archive.read("docProps/app.xml").decode("utf-8")
        
        if ext == ".docx":
            match = re.search(r"<Pages>(\d+)</Pages>", app_xml)
        elif ext == ".pptx":
            match = re.search(r"<Slides>(\d+)</Slides>", app_xml)
        else:
            return 0
            
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 0


@app.post("/evaluate", response_class=HTMLResponse)
async def evaluate_document(
    # 1. File Inputs
    file: UploadFile = File(..., description="Upload the PDF, DOCX, or PPTX document"),
    dataset: UploadFile = File(..., description="Upload the Golden Dataset (Strictly JSON format)"),
    
    # 2. Required Configuration Form Fields
    provider: str = Form(..., description="The main LLM provider (e.g., openai, groq)"),
    model_choice: str = Form(..., description="The specific LLM model choice"),
    api_key: str = Form(..., description="API Key for the LLM"),
    api_url: str = Form(..., description="API URL for the LLM"),
    embed_provider: str = Form(..., description="The embedding model provider"),
    embed_api_url: str = Form(..., description="API URL for the embedding model"),
    embed_model_choice: str = Form(..., description="The specific embedding model choice"),
    chunk_size: int = Form(..., description="The chunk size for text splitting"),
    chunk_overlap: int = Form(..., description="The chunk overlap for text splitting")
):
    # ---------------------------------------------------------
    # 3. Handle & Validate the Primary Document
    # ---------------------------------------------------------
    filename = file.filename or "uploaded_document"
    ext = Path(filename).suffix.lower()
    allowed_exts = {".pdf", ".docx", ".pptx"}
    
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported document format '{ext}'. Only PDF, DOCX, and PPTX are permitted."
        )
        
    doc_content = await file.read()
    page_count = 0
    
    try:
        if ext == ".pdf":
            doc = fitz.open(stream=doc_content, filetype="pdf")
            page_count = doc.page_count
            doc.close()
        else:
            page_count = get_office_page_count(doc_content, ext)
            
        if page_count > 100:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large ({page_count} pages). Free tier is limited to 100 pages."
            )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail="Corrupted or unreadable document metadata.")
        
    # ---------------------------------------------------------
    # 4. Handle & STRICTLY Validate the Golden Dataset
    # ---------------------------------------------------------
    dataset_filename = dataset.filename or "golden_dataset.json"
    dataset_ext = Path(dataset_filename).suffix.lower()
    
    if dataset_ext != ".json":
        raise HTTPException(
            status_code=400, 
            detail="Unsupported dataset format. Please upload a strictly formatted .json file."
        )
    
    dataset_content = await dataset.read()

    # Apply strict JSON schema validation
    try:
        parsed_dataset = json.loads(dataset_content)
        
        if not isinstance(parsed_dataset, list):
            raise ValueError("The root element must be a JSON array (list of objects).")
            
        if len(parsed_dataset) == 0:
            raise ValueError("The JSON array cannot be empty.")
            
        for idx, row in enumerate(parsed_dataset):
            if not isinstance(row, dict):
                raise ValueError(f"Item at index {idx} must be a JSON object.")
                
            if "query" not in row or "answer" not in row:
                raise ValueError(f"Item at index {idx} is missing required keys: 'query' and 'answer'.")
                
            if not isinstance(row["query"], str) or not isinstance(row["answer"], str):
                raise ValueError(f"The 'query' and 'answer' values at index {idx} must be text strings.")
                
    except Exception as e:
        sample_format = """[
  {
    "query": "What first made Ravi realize his water pot might be special after the old man left?",
    "answer": "Ravi realized the pot was special when he tilted it to get a final drop...",
    "doc_id": "docA.pdf",
    "meta": { "page": 1 }
  }
]"""
        raise HTTPException(
            status_code=400, 
            detail=f"The JSON format you uploaded is not supported. Error: {str(e)}\n\nPlease see the sample JSON for the correct format:\n{sample_format}"
        )

    # ---------------------------------------------------------
    # 5. Save Both Files to Cloud-Safe Directory
    # ---------------------------------------------------------
    save_dir = Path("output/uploads")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    doc_path = save_dir / filename
    dataset_path = save_dir / dataset_filename
    
    with open(doc_path, "wb") as f:
        f.write(doc_content)
        
    with open(dataset_path, "wb") as f:
        f.write(dataset_content)
        
    # ---------------------------------------------------------
    # 6. Package the LangGraph State Dictionary
    # ---------------------------------------------------------
    # Maps the validated API inputs exactly to your workflow requirements
    agent_state_input = {
        "provider": provider,
        "model_choice": model_choice,
        "api_key": api_key,
        "api_url": api_url,
        "embed_provider": embed_provider,
        "embed_api_url": embed_api_url,
        "embed_model_choice": embed_model_choice,
        "file_path": str(doc_path),
        "golden_dataset": parsed_dataset,  # Sending the parsed JSON data directly
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "file_name": filename
    }
        
    # ---------------------------------------------------------
    # 7. Trigger LangGraph Orchestrator
    # ---------------------------------------------------------
   try:
        # AWAIT the async LangGraph execution
        final_html_report = await run_evaluator(agent_state_input) 
        return final_html_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
