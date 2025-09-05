# clientmanager/rag_engine.py
import re
import numpy as np
from functools import lru_cache
from typing import List, Dict
import logging

from django.contrib.staticfiles import finders
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Configuration
PDF_STATIC_PATH = "docs/EMPLOYEE USER MANUAL.pdf"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CHUNK_CHARS = 100
MAX_CHUNK_CHARS = 800

def _clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Replace non-breaking spaces and normalize whitespace
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _split_into_chunks(text: str, page_num: int) -> List[Dict]:
    """Split text into manageable chunks."""
    # Split on double newlines (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    
    for para in paragraphs:
        para = _clean_text(para)
        if len(para) < MIN_CHUNK_CHARS:
            continue
            
        # If paragraph is too long, split on sentences
        if len(para) > MAX_CHUNK_CHARS:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= MAX_CHUNK_CHARS:
                    current_chunk = (current_chunk + " " + sentence).strip()
                else:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk,
                            "page": page_num
                        })
                    current_chunk = sentence
            
            if current_chunk and len(current_chunk) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "text": current_chunk,
                    "page": page_num
                })
        else:
            chunks.append({
                "text": para,
                "page": page_num
            })
    
    return chunks

def _load_pdf() -> List[Dict]:
    """Load PDF and extract chunks with page numbers."""
    # Find the PDF file
    abs_path = finders.find(PDF_STATIC_PATH)
    if not abs_path:
        raise FileNotFoundError(f"PDF not found: {PDF_STATIC_PATH}")
    
    logger.info(f"Loading PDF from: {abs_path}")
    
    # Read PDF
    reader = PdfReader(abs_path)
    all_chunks = []
    
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
            if text:
                chunks = _split_into_chunks(text, page_idx)
                all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"Error processing page {page_idx}: {e}")
            continue
    
    # Remove duplicates
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        # Create a signature for deduplication
        signature = chunk["text"][:100].lower().strip()
        if signature not in seen and len(chunk["text"]) >= MIN_CHUNK_CHARS:
            seen.add(signature)
            unique_chunks.append(chunk)
    
    logger.info(f"Loaded {len(unique_chunks)} chunks from {len(reader.pages)} pages")
    return unique_chunks

@lru_cache(maxsize=1)
def _get_model():
    """Get the sentence transformer model (cached)."""
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)

@lru_cache(maxsize=1)
def _build_index():
    """Build the search index (cached)."""
    try:
        # Load chunks
        chunks = _load_pdf()
        
        # Get model
        model = _get_model()
        
        # Create embeddings
        texts = [chunk["text"] for chunk in chunks]
        logger.info(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        logger.info("Index built successfully")
        return chunks, np.array(embeddings, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise

def retrieve_top_k(query: str, k: int = 5) -> List[Dict]:
    """Retrieve top-k most relevant chunks for a query."""
    if not query or not query.strip():
        return []
    
    try:
        # Get index
        chunks, embeddings = _build_index()
        
        # Get model and encode query
        model = _get_model()
        query_embedding = model.encode([query.strip()], normalize_embeddings=True)[0]
        
        # Calculate similarities
        similarities = np.dot(embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append({
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(similarities[idx])
            })
        
        logger.debug(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
        return results
        
    except Exception as e:
        logger.error(f"Error in retrieve_top_k: {e}")
        return []

def refresh_cache() -> int:
    """Clear cache and rebuild index. Returns number of chunks loaded."""
    try:
        # Clear caches
        _build_index.cache_clear()
        _get_model.cache_clear()
        
        # Rebuild
        chunks, _ = _build_index()
        logger.info("Cache refreshed successfully")
        return len(chunks)
        
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        raise

def get_index_info() -> Dict:
    """Get information about the current index."""
    try:
        chunks, embeddings = _build_index()
        
        # Count pages
        pages = set(chunk["page"] for chunk in chunks)
        
        # Calculate average chunk length
        avg_length = np.mean([len(chunk["text"]) for chunk in chunks])
        
        return {
            "total_chunks": len(chunks),
            "total_pages": len(pages),
            "embedding_dimension": embeddings.shape[1],
            "avg_chunk_length": int(avg_length),
            "model_name": EMBED_MODEL
        }
        
    except Exception as e:
        logger.error(f"Error getting index info: {e}")
        return {"error": str(e)}

def health_check() -> Dict:
    """Check if the RAG engine is working properly."""
    try:
        # Check if PDF exists
        abs_path = finders.find(PDF_STATIC_PATH)
        if not abs_path:
            return {"status": "error", "message": f"PDF not found: {PDF_STATIC_PATH}"}
        
        # Try to load model
        model = _get_model()
        
        # Try a simple query
        results = retrieve_top_k("test", k=1)
        
        return {
            "status": "healthy",
            "pdf_found": True,
            "model_loaded": True,
            "can_retrieve": len(results) > 0,
            "index_info": get_index_info()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }