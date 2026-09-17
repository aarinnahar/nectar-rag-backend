import re
from typing import List

# Compile regex globally so it isn't rebuilt for every chunk
WORD_REGEX = re.compile(r'\b\w+\b')

def get_ngrams(text: str, n: int = 3) -> set:
    """Generates a clean set of n-grams, ignoring punctuation."""
    words = WORD_REGEX.findall(text.lower())
    
    if len(words) < n:
        return set([" ".join(words)]) if words else set()
        
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

def compute_inter_chunk_redundancy(
    all_document_chunks: List[str], 
    n_gram_size: int = 3
) -> float:
    """
    Calculates phrase-level overlap (Overlap Coefficient) between consecutive chunks.
    
    Uses Overlap Coefficient (Intersection / Min(Set1, Set2)) instead of Jaccard 
    to accurately measure redundancy even when chunks vary significantly in size.
    """
    if not all_document_chunks or len(all_document_chunks) <= 1:
        return 0.0
        
    # Extract chunks safely in case framework objects (like LangChain Documents) are passed
    clean_texts = [
        c.page_content if hasattr(c, 'page_content') 
        else c.get("text", "") if isinstance(c, dict) 
        else str(c) 
        for c in all_document_chunks
    ]
        
    chunk_ngram_sets = [get_ngrams(chunk, n=n_gram_size) for chunk in clean_texts]
    overlaps = []
    
    for i in range(len(chunk_ngram_sets) - 1):
        set1 = chunk_ngram_sets[i]
        set2 = chunk_ngram_sets[i + 1]
        
        min_len = min(len(set1), len(set2))
        
        if min_len == 0:
            overlap_coeff = 0.0
        else:
            # Overlap Coefficient: How much of the smaller chunk is just a repeat?
            overlap_coeff = len(set1.intersection(set2)) / min_len
            
        overlaps.append(overlap_coeff)
        
    average_redundancy = sum(overlaps) / len(overlaps)
    
    return round(average_redundancy, 4)