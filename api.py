import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_MAX_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import fitz  # PyMuPDF
import zipfile
import io
import re
from pathlib import Path

# Import your LangGraph entry point here
from src.orchestration.graph_builder import workflow as run_evaluator 

app = FastAPI(title="RAG Evaluator API")

# Bypasses CORS blocks so your Vercel React app can talk to this Render backend
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
async def evaluate_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_document"
    ext = Path(filename).suffix.lower()
    allowed_exts = {".pdf", ".docx", ".pptx"}
    
    # 1. Format Guardrail
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format '{ext}'. Only PDF, DOCX, and PPTX files are permitted."
        )
        
    content = await file.read()
    page_count = 0
    
    # 2. Length Guardrail: Prevent the 100-Second Gateway Timeout
    try:
        if ext == ".pdf":
            doc = fitz.open(stream=content, filetype="pdf")
            page_count = doc.page_count
            doc.close()
        else:
            page_count = get_office_page_count(content, ext)
            
        if page_count > 100:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large ({page_count} pages/slides). To prevent server timeouts, the free tier is strictly limited to 100 pages."
            )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail="Corrupted or unreadable file metadata.")
        
    # 3. Save the file to the dynamic cloud-safe directory
    save_dir = Path("output/uploads")
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / filename
    
    with open(file_path, "wb") as f:
        f.write(content)
        
    # 4. Trigger your LangGraph Orchestrator
    try:
        final_html_report = run_evaluator(str(file_path)) 
        return final_html_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
