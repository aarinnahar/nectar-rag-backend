from src.chunking.character_text_splitter import character_text_splitter_chunker
from src.chunking.recursive_character_text_splitter import recursive_character_text_splitter_chunker
from src.chunking.token_text_splitter import token_text_splitter_chunker
from src.chunking.semantic_chunking import semantic_chunking_pro
from src.orchestration.agent_state import AgentState
from src.utils.save_state import save_state, load_state
import time
import os
import logging
import asyncio # New import for parallel execution

logger = logging.getLogger("app") 

def extract_and_protect_tables(markdown_content: str):
    """
    Scans markdown text, extracts tables, and replaces them with placeholders.
    Returns the clean text and a list of extracted table strings.
    """
    lines = markdown_content.split('\n')
    new_lines = []
    tables = []
    in_table = False
    current_table = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            if not in_table:
                in_table = True
                current_table = []
            current_table.append(line)
        else:
            if in_table:
                tables.append("\n".join(current_table))
                new_lines.append(f"\n\n__TABLE_PLACEHOLDER_{len(tables) - 1}__\n\n")
                in_table = False
            new_lines.append(line)
    
    if in_table:
        tables.append("\n".join(current_table))
        new_lines.append(f"\n\n__TABLE_PLACEHOLDER_{len(tables) - 1}__\n\n")

    return "\n".join(new_lines), tables


async def run_chunker_with_timing(chunker_func, *args):
    """
    Helper function to run CPU-bound chunkers in a separate thread 
    and calculate execution time without blocking the async loop.
    """
    start_time = time.time()
    # Execute the synchronous chunker in a background thread
    chunks = await asyncio.to_thread(chunker_func, *args)
    end_time = time.time()
    return chunks, end_time - start_time


async def chunking_factory(state: AgentState):
    """
    Asynchronous LangGraph node that runs four chunking strategies concurrently.
    """
    file_path = state['text_markdown_path']
    with open(file_path, 'r', encoding="utf-8") as f:
        text = f.read()
        
    chunk_size = state['chunk_size']
    chunk_overlap = state['chunk_overlap']
    
    clean_text, tables = extract_and_protect_tables(text)

    # 1. Dispatch all 4 chunkers to run concurrently
    logger.debug("Dispatching parallel chunking tasks...")
    char_task = run_chunker_with_timing(character_text_splitter_chunker, clean_text, chunk_size, chunk_overlap)
    recur_task = run_chunker_with_timing(recursive_character_text_splitter_chunker, clean_text, chunk_size, chunk_overlap)
    token_task = run_chunker_with_timing(token_text_splitter_chunker, clean_text, chunk_size, chunk_overlap)
    semantic_task = run_chunker_with_timing(semantic_chunking_pro, clean_text)

    # 2. Wait for all of them to finish at the exact same time
    (
        (char_chunks, diff_chara),
        (recur_chunks, diff_recur),
        (token_chunks, diff_token),
        (semantic_chunks, diff_semantic)
    ) = await asyncio.gather(char_task, recur_task, token_task, semantic_task)

    # 3. Assemble the factory dictionaries
    chunk_factory = {
        "character_text_splitter_chunks": char_chunks,
        "recursive_text_splitter_chunks": recur_chunks,
        "token_text_splitter_chunks": token_chunks,
        "semantic_chunks": semantic_chunks
    }
    
    ingestion = {
        "character_text_splitter_chunks": diff_chara,
        "recursive_text_splitter_chunks": diff_recur,
        "token_text_splitter_chunks": diff_token,
        "semantic_chunks": diff_semantic
    }

    # 4. Restore Table Placeholders
    for strategy, chunks in chunk_factory.items():
        updated_chunks = []
        for chunk in chunks:
            if "__TABLE_PLACEHOLDER_" in chunk:
                for i, table_content in enumerate(tables):
                    placeholder = f"__TABLE_PLACEHOLDER_{i}__"
                    if placeholder in chunk:
                        chunk = chunk.replace(placeholder, table_content)
            updated_chunks.append(chunk)
        chunk_factory[strategy] = updated_chunks

    # 5. Save state and return
    save_state(filename="chunk_factory", data=chunk_factory)
    save_state(filename="ingestion_time", data=ingestion)
    
    logger.debug("Chunking Factory Completed (Parallel Execution)")
    return {"chunk_factory": chunk_factory, "ingestion_time": ingestion}