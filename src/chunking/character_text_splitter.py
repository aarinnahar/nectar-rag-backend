from langchain_text_splitters import CharacterTextSplitter
from src.config.settings import get_settings
import logging
logger = logging.getLogger("app") 

def character_text_splitter_chunker(text, chunk_size, chunk_overlap):
    logger.info("CharacterTextSplitter CHUNKER STARTED")
    print(f"text is {type(text)}")
    settings = get_settings()
    splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, separator= "\n")
    chunk = splitter.split_text(text)
    logger.debug(f"CharacterTextSplitter Chunker {chunk}")
    logger.info(f"EXITING CharacterTextSplitter CHUNKER length of chunks are {len(chunk)}")
    return chunk