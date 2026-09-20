import os
import json
import uuid
import shutil
import asyncio

# MUST BE AT THE VERY TOP: Locks memory allocation to prevent OOM crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_MAX_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
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

# --- THE GLOBAL CPU LOCK ---
# This ensures only one heavy evaluation runs at a time to protect the 1GB AWS server
evaluation_lock = asyncio.Lock()

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


@app.post("/evaluate")
async def evaluate_document(
    # 1. File Inputs
    file: UploadFile = File(..., description="Upload the PDF, DOCX, or PPTX document"),
    dataset: UploadFile = File(..., description="Upload the Golden Dataset (Strictly JSON format)"),
    
    # 2. Required Configuration Form Fields
    provider: str = Form(..., description="The main LLM provider (e.g., openai, groq)"),
    model_choice: str = Form(..., description="The specific LLM model choice"),
    api_key: str = Form(..., description="API Key for the LLM"),
    api_url: str = Form(..., description="API URL for the LLM"),
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
        sample_format = """"""
        raise HTTPException(
            status_code=400, 
            detail=f"The JSON format you uploaded is not supported. Error: {str(e)}\n\nPlease see the sample JSON for the correct format\n{sample_format}"
        )

    # ---------------------------------------------------------
    # 5. UUID Sandboxing: Save Files to an Isolated Directory
    # ---------------------------------------------------------
    run_id = str(uuid.uuid4())
    run_dir = Path(f"output/runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    doc_path = run_dir / filename
    dataset_path = run_dir / dataset_filename
    
    with open(doc_path, "wb") as f:
        f.write(doc_content)
        
    with open(dataset_path, "wb") as f:
        f.write(dataset_content)
        
    # ---------------------------------------------------------
    # 6. Package the LangGraph State Dictionary
    # ---------------------------------------------------------
    agent_state_input = {
        "run_id": run_id,                  # Pass unique ID to graph
        "output_dir": str(run_dir),        # Pass unique isolated directory
        "provider": provider,
        "model_choice": model_choice,
        "api_key": api_key,
        "api_url": api_url,
        "file_path": str(doc_path),        # Isolated file path
        "golden_dataset": parsed_dataset, 
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "file_name": filename
    }
        
    # ---------------------------------------------------------
    # 7 & 8. Stream LangGraph execution & Read Report
    # ---------------------------------------------------------
    async def event_generator():
        try:
            # Let the frontend know if they are waiting in the queue
            if evaluation_lock.locked():
                yield f"data: {json.dumps({'node': 'Waiting in Queue...'})}\n\n"

            # 1. ACQUIRE LOCK: Only one user enters this block at a time
            async with evaluation_lock:
                
                # Stream the nodes as they execute in LangGraph
                async for chunk in run_evaluator(agent_state_input):
                    for node_name in chunk.keys():
                        yield f"data: {json.dumps({'node': node_name})}\n\n"
            
                # 2. Pipeline finished! Read report safely.
                # Check the isolated directory first. If LangGraph is still saving globally, fallback.
                report_path = run_dir / "chunking_report.html"
                if not report_path.exists():
                    report_path = Path("output/reports/chunking_report.html")
                if not report_path.exists():
                    report_path = Path("chunking_report.html")

                if report_path.exists():
                    with open(report_path, "r", encoding="utf-8") as f:
                        html_string = f.read()
                    # Send the final report and close modal
                    yield f"data: {json.dumps({'status': 'completed', 'report_html': html_string})}\n\n"
                else:
                    yield f"data: {json.dumps({'error': 'Report not found on disk'})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
        finally:
            # 3. GARBAGE COLLECTION: Delete the user's files and directory immediately
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    # Return the generator as an active HTTP stream
    return StreamingResponse(event_generator(), media_type="text/event-stream")
