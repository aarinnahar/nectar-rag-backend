from src.config.settings import Settings
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import html
from bs4 import BeautifulSoup
import logging



logger = logging.getLogger("app") 

# ===================================================================
# GLOBAL INITIALIZATIONS (Runs only once when the server starts)
# ===================================================================

# 2. OPTIMIZATION: Strip out heavy NLP tasks. 
# We disable everything except 'senter' (the lightning-fast sentence segmenter)
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer", "textcat", "attribute_ruler"])
nlp.enable_pipe("senter")

# ===================================================================

def robust_clean(raw_text):
    unescaped_text = html.unescape(raw_text)
    soup = BeautifulSoup(unescaped_text, "html.parser")
    clean_text = soup.get_text()
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def semantic_chunking_pro(texts, window_size=3, percentile=10):
    text = robust_clean(texts)
    
    # 1. Split sentences (Now lightning fast due to the stripped SpaCy pipeline)
    import spacy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
    
    if len(sentences) < window_size:
        return [" ".join(sentences)]

    # 2. LAZY LOAD: Initialize model here so it gets destroyed after chunking
    
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=1)
    
    # 3. Batch Embedding (Now utilizes the FastEmbed ONNX model)
    # Cast to a list first to ensure compatibility with LangChain's generator returns, then to numpy
    vectors = np.array(list(embedding_model.embed_documents(sentences)))

    # 4. Sliding Window Means 
    windows = sliding_window_view(vectors, window_shape=(window_size,), axis=0)
    mean_vectors = windows.mean(axis=1)

    # 5. OPTIMIZATION: Vectorized Cosine Similarity (The for-loop is gone!)
    # This computes the similarity matrix in C instantly, then grabs the adjacent pairs
    sim_matrix = cosine_similarity(mean_vectors)
    similarities = np.diagonal(sim_matrix, offset=1)

    # 6. Thresholding
    threshold = np.percentile(similarities, percentile)
    
    # 7. Find Breakpoints
    offset = window_size // 2
    breakpoints = [i + offset for i, s in enumerate(similarities) if s <= threshold]

    # 8. Slicing
    chunks = []
    start_idx = 0
    for bp in breakpoints:
        chunks.append(" ".join(sentences[start_idx:bp]))
        start_idx = bp
    
    chunks.append(" ".join(sentences[start_idx:]))
    
    return chunks
