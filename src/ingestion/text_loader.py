from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state, load_state
from src.utils.release_memory import release_system_memory
import os
import io
import time
import base64
import logging
import zipfile
import requests
from pathlib import Path
from PIL import Image

# Import C++ / XML parsing libraries (Zero PyTorch needed)
import pymupdf4llm
from docx import Document as DocxDocument
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

# Set up Production Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def call_gemini_vision_fallback(pil_image: Image.Image, retries: int = 3, delay: int = 5) -> str:
    """Calls Gemini VLM for embedded images in Word/PPTX. Zero local RAM cost."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        logger.warning("No Gemini API Key found. Skipping VLM extraction.")
        return ""

    try:
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gemini-2.5-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an expert data extraction AI. Examine the visual elements in this image "
                                "and extract ALL text, categories, numbers, and data. "
                                "Output the data strictly as a well-formatted Markdown table or clean text. "
                                "CRITICAL: Do NOT include conversational filler, introductions, or explanations."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096
        }

        current_delay = delay
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    logger.warning(f"Gemini Rate Limit (429). Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= 2
                else:
                    logger.error(f"Gemini API error ({response.status_code}): {response.text}")
                    break
            except requests.exceptions.RequestException as req_err:
                logger.warning(f"Network error calling Gemini: {req_err}")
                time.sleep(current_delay)
                current_delay *= 2

        return ""
    except Exception as e:
        logger.error(f"Fatal error preparing image for VLM fallback: {e}")
        return ""


def process_pdf_lightweight(file_path: str) -> str:
    """Lightning-fast digital PDF extraction using PyMuPDF (C++ Engine)."""
    logger.info("Routing to PyMuPDF for instant C++ extraction...")
    start_time = time.time()
    
    # Extracts text and markdown tables instantly without PyTorch
    final_markdown = pymupdf4llm.to_markdown(file_path)
    release_system_memory()
    
    conversion_time = time.time() - start_time
    print(f"\n{'='*50}\n🚀 DIGITAL PDF PARSED IN {conversion_time:.2f} SECONDS\n{'='*50}\n")

    # Keep your state saving mechanism intact
    save_state(filename="total_time_pymu", data={"total_time_pymu": conversion_time})
    return final_markdown


def process_docx_smart_autopilot(docx_path: str) -> str:
    """Processes Word (.docx) files securely using ZIP extraction."""
    logger.info(f"Processing Word document: {docx_path}")
    doc = DocxDocument(docx_path)
    markdown_sections = []

    for para in doc.paragraphs:
        if para.text.strip():
            markdown_sections.append(para.text.strip())

    for table in doc.tables:
        table_md = []
        for row in table.rows:
            row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            table_md.append("| " + " | ".join(row_cells) + " |")
        if table_md:
            header_sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
            table_md.insert(1, header_sep)
            markdown_sections.append("\n" + "\n".join(table_md) + "\n")

    markdown_sections.append("\n## Embedded Figures & Image Tables\n")
    valid_extensions = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp', 'emf', 'wmf']
    
    try:
        with zipfile.ZipFile(docx_path, 'r') as docx_zip:
            for item in docx_zip.namelist():
                if item.startswith('word/media/') and item.split('.')[-1].lower() in valid_extensions:
                    try:
                        image_data = docx_zip.read(item)
                        pil_img = Image.open(io.BytesIO(image_data))
                        if pil_img.mode not in ('RGB', 'L'):
                            pil_img = pil_img.convert('RGB')

                        logger.info(f"Found embedded image ({item}). Sending to Gemini Cloud API...")
                        vlm_description = call_gemini_vision_fallback(pil_img)
                        
                        if vlm_description:
                            markdown_sections.append(f"\n\n{vlm_description}\n")
                    except Exception as img_err:
                        logger.warning(f"Could not process image {item}: {img_err}")
    except Exception as e:
        logger.error(f"Error during Word ZIP extraction: {e}")

    return "\n\n".join(markdown_sections)


def process_pptx_smart_autopilot(pptx_path: str) -> str:
    """Processes PowerPoint presentations natively."""
    logger.info(f"Processing PowerPoint document: {pptx_path}")
    prs = Presentation(pptx_path)
    markdown_sections = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        markdown_sections.append(f"\n## Slide {slide_num}\n")
        native_text = []
        image_shapes = []

        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.has_text_frame and shape.text.strip():
                native_text.append(shape.text.strip())
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                image_shapes.append(shape)
            elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                for sub_shape in shape.shapes:
                    if sub_shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        image_shapes.append(sub_shape)

        if native_text:
            markdown_sections.append("\n".join(native_text))

        for shape in image_shapes:
            try:
                pil_img = Image.open(io.BytesIO(shape.image.blob))
                
                # Directly route to Gemini API instead of local RapidOCR
                vlm_description = call_gemini_vision_fallback(pil_img)
                if vlm_description:
                    markdown_sections.append(f"\n\n{vlm_description}\n")
            except Exception as img_err:
                logger.error(f"Slide {slide_num}: Shape error: {img_err}")

    return "\n\n".join(markdown_sections)


def text_loader(state: AgentState) -> dict:
    """Universal Production Router."""
    file_path = state['file_path']
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path_obj.suffix.lower()
    try:
        if ext == ".docx":
            final_markdown = process_docx_smart_autopilot(file_path)
        elif ext == ".pptx":
            final_markdown = process_pptx_smart_autopilot(file_path)
        elif ext in [".pdf", ".png", ".jpg", ".jpeg"]:
            final_markdown = process_pdf_lightweight(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # CLOUD-SAFE PATHING: Dynamic local directory relative to script execution
        output_dir = Path("output/temp_md_files")
        output_dir.mkdir(parents=True, exist_ok=True) # Creates folders if they don't exist
        
        output_path = output_dir / f"{file_path_obj.stem}_extracted.md"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)
            logger.info(f"Success! Output saved to: {output_path}")
            
    except Exception as main_err:
        logger.error(f"Pipeline execution failed: {main_err}")
        raise main_err
        
    return {"text_markdown_path": str(output_path)}
