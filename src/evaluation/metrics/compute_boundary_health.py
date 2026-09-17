import re
from typing import List, Union, Any

# 1. Compile regexes globally so they are only compiled once at startup
START_BOUNDARY_REGEX = re.compile(r'^(?:[A-Z0-9"\'\(\[\{`#]|[-*•]\s+|\d+\.\s+)')
LOWERCASE_START_REGEX = re.compile(r'^[a-z]')
HYPHEN_SLICE_REGEX = re.compile(r'\b[a-zA-Z]+-\s*$')
TERMINAL_PUNC_REGEX = re.compile(r'(?:[.!?]|```|["\'\)\}\]])\s*$')
CLAUSE_SEP_REGEX = re.compile(r'[,:\-\;\&]\s*$')
BACKTICK_REGEX = re.compile(r'```')

COMMON_ABBRS = {"e.g.", "i.e.", "etc.", "vs.", "dr.", "mr.", "mrs.", "prof.", "inc.", "ltd."}
CODE_KEYWORDS = {"def", "class", "import", "from", "return", "if", "else", "elif", "for", "while", "try", "except", "const", "let", "var", "function"}

def _extract_text(chunk: Any) -> str:
    """Safely extracts text from strings, dicts, LangChain Documents, or LlamaIndex Nodes."""
    if isinstance(chunk, str):
        return chunk
    # Handle LangChain Documents
    if hasattr(chunk, "page_content"):
        return str(chunk.page_content)
    # Handle LlamaIndex Nodes
    if hasattr(chunk, "text"):
        return str(chunk.text)
    # Handle dictionaries
    if isinstance(chunk, dict):
        return str(chunk.get("page_content", chunk.get("text", "")))
    return str(chunk)

def compute_boundary_health(chunk_input: Union[str, List[Any]]) -> float:
    """
    Computes a deterministic Boundary Health Score (0.0 to 1.0).
    Evaluates clean sentence boundaries, word truncation, and structural integrity.
    """
    # Normalize input to a list
    if not isinstance(chunk_input, list):
        chunk_input = [chunk_input]

    # Extract and clean text safely
    chunks = [_extract_text(c).strip() for c in chunk_input]
    chunks = [c for c in chunks if c]

    if not chunks:
        return 0.0

    scores = []

    for text in chunks:
        words = text.split()
        if not words:
            continue
            
        first_word = words[0].lower()
        last_word = words[-1].lower()
        
        start_score = 0.35
        end_score = 0.20

        # ---------------------------------------------------------------------
        # 1. START BOUNDARY EVALUATION (0.0 to 0.5)
        # ---------------------------------------------------------------------
        if START_BOUNDARY_REGEX.match(text) or first_word in CODE_KEYWORDS:
            start_score = 0.5
        elif LOWERCASE_START_REGEX.match(text):
            start_score = 0.15

        # ---------------------------------------------------------------------
        # 2. END BOUNDARY EVALUATION (0.0 to 0.5)
        # ---------------------------------------------------------------------
        if HYPHEN_SLICE_REGEX.search(text):
            end_score = 0.0
        elif TERMINAL_PUNC_REGEX.search(text) and last_word not in COMMON_ABBRS:
            end_score = 0.5
        elif CLAUSE_SEP_REGEX.search(text):
            end_score = 0.25

        # ---------------------------------------------------------------------
        # 3. CODE BLOCK INTEGRITY
        # ---------------------------------------------------------------------
        total_score = start_score + end_score
        
        # Penalize unclosed backtick pairs
        backtick_matches = len(BACKTICK_REGEX.findall(text))
        if backtick_matches % 2 != 0:
            total_score = max(0.0, total_score - 0.15)

        scores.append(total_score)

    return round(float(sum(scores) / len(scores)), 4) if scores else 0.0