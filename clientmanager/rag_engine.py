# clientmanager/rag_engine.py
import re
import os
import numpy as np
from functools import lru_cache
from typing import List, Dict
import logging

from django.conf import settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 🔧 CONFIGURATION
PDF_PATH = os.path.join(
    settings.BASE_DIR,
    'clientmanager', 'static', 'clientmanager', 'assets', 'docs',
    'EMPLOYEE USER MANUAL.pdf'
)
# ─────────────────────────────────────────────

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CHUNK_CHARS = 40   # lowered from 100 so URLs and short lines are kept
MAX_CHUNK_CHARS = 800


def _clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _split_into_chunks(text: str, page_num: int) -> List[Dict]:
    """Split text into manageable chunks, preserving short but important lines like URLs."""
    
    # ── FIX: Merge orphaned short lines before chunking ──
    # Instead of splitting on double newlines (which loses short lines like URLs),
    # we walk line by line and group them into paragraphs intelligently.
    lines = text.split('\n')
    merged = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            # Blank line = paragraph break
            if buffer:
                merged.append(buffer)
                buffer = ""
        else:
            buffer = (buffer + " " + line).strip()

    if buffer:
        merged.append(buffer)

    paragraphs = merged
    chunks = []

    for para in paragraphs:
        para = _clean_text(para)
        if len(para) < MIN_CHUNK_CHARS:
            continue

        if len(para) > MAX_CHUNK_CHARS:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= MAX_CHUNK_CHARS:
                    current_chunk = (current_chunk + " " + sentence).strip()
                else:
                    if current_chunk:
                        chunks.append({"text": current_chunk, "page": page_num})
                    current_chunk = sentence

            if current_chunk and len(current_chunk) >= MIN_CHUNK_CHARS:
                chunks.append({"text": current_chunk, "page": page_num})
        else:
            chunks.append({"text": para, "page": page_num})

    return chunks


def _load_pdf() -> List[Dict]:
    """Load PDF from disk and extract chunks with page numbers."""
    logger.info(f"Loading PDF from: {PDF_PATH}")

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found at: {PDF_PATH}\n"
            f"Make sure 'EMPLOYEE USER MANUAL.pdf' is in: "
            f"clientmanager/static/clientmanager/assets/docs/"
        )

    reader = PdfReader(PDF_PATH)
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
        chunks = _load_pdf()
        model = _get_model()

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
        chunks, embeddings = _build_index()
        model = _get_model()
        query_embedding = model.encode([query.strip()], normalize_embeddings=True)[0]

        similarities = np.dot(embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]

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
        _build_index.cache_clear()
        _get_model.cache_clear()

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
        pages = set(chunk["page"] for chunk in chunks)
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
        if not os.path.exists(PDF_PATH):
            return {
                "status": "error",
                "message": f"PDF not found at: {PDF_PATH}"
            }

        model = _get_model()
        results = retrieve_top_k("test", k=1)

        return {
            "status": "healthy",
            "pdf_path": PDF_PATH,
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