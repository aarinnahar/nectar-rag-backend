from langchain_text_splitters import TokenTextSplitter
from src.config.settings import get_settings
import logging
logger = logging.getLogger("app") 

def token_text_splitter_chunker(text, chunk_size, chunk_overlap):
    logger.info("TokenTextSplitter CHUNKER STARTED")
    print(f"text is {type(text)}")
    settings = get_settings()
    splitter = TokenTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunk = splitter.split_text(text)
    logger.debug(f"TokenTextSplitter Chunker {chunk}")
    logger.info(f"EXITING TokenTextSplitter CHUNKER length of chunks are {len(chunk)}")
    return chunk