# clientmanager/rag_engine.py
import re
import os
import io
import numpy as np
from functools import lru_cache
from typing import List, Dict
import logging
import requests

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 🔧 CONFIGURATION
PDF_URL = "https://crmsystemaistmarytx.onrender.com/static/clientmanager/assets/docs/EMPLOYEE%20USER%20MANUAL.pdf"

try:
    from django.conf import settings
    PDF_FALLBACK_PATH = os.path.join(
        settings.BASE_DIR,
        'clientmanager', 'static', 'clientmanager', 'assets', 'docs',
        'EMPLOYEE USER MANUAL.pdf'
    )
except Exception:
    PDF_FALLBACK_PATH = None
# ─────────────────────────────────────────────

#EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2" # Lighter model
MIN_CHUNK_CHARS = 40
MAX_CHUNK_CHARS = 800

# Regex to detect URLs anywhere in a line
URL_RE = re.compile(r'https?://\S+')


def _clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_url_chunks(text: str, page_num: int) -> List[Dict]:
    """
    Extract dedicated chunks for every line that contains a URL.
    This ensures URLs are never buried inside larger paragraphs and
    always get their own embedding so they score highly on URL queries.

    For example:
        "Employee Analytics & Data Selector Hub (Advanced Reporting)
         URL: https://crmsystemaistmarytx.onrender.com/reporting-tool/"

    becomes a standalone chunk:
        "Employee Analytics & Data Selector Hub (Advanced Reporting)
         URL: https://crmsystemaistmarytx.onrender.com/reporting-tool/"
    """
    url_chunks = []
    lines = text.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if URL_RE.search(line):
            # Grab the line before (likely the section title) + the URL line
            context_lines = []
            if i > 0 and lines[i - 1].strip():
                context_lines.append(lines[i - 1].strip())
            context_lines.append(line)
            # Also grab the line after if it exists and is short (e.g. "Purpose Statement")
            if i + 1 < len(lines) and lines[i + 1].strip() and len(lines[i + 1].strip()) < 80:
                context_lines.append(lines[i + 1].strip())

            chunk_text = _clean_text(' '.join(context_lines))
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                url_chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "is_url_chunk": True   # tag for debugging
                })
        i += 1

    return url_chunks


def _split_into_chunks(text: str, page_num: int) -> List[Dict]:
    """Split text into manageable chunks, preserving short lines like URLs."""
    lines = text.split('\n')
    merged = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            if buffer:
                merged.append(buffer)
                buffer = ""
        else:
            buffer = (buffer + " " + line).strip()

    if buffer:
        merged.append(buffer)

    chunks = []
    for para in merged:
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


def _download_pdf_bytes() -> io.BytesIO:
    """Download PDF from URL with retry and local fallback."""
    logger.info(f"Downloading PDF from: {PDF_URL}")

    for attempt in range(2):
        try:
            response = requests.get(PDF_URL, timeout=60)
            response.raise_for_status()
            content = response.content

            if not content.startswith(b'%PDF'):
                raise ValueError(f"Not a valid PDF (starts with: {content[:20]})")

            logger.info(f"PDF downloaded successfully ({len(content):,} bytes)")
            return io.BytesIO(content)

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 1:
                if PDF_FALLBACK_PATH and os.path.exists(PDF_FALLBACK_PATH):
                    logger.warning("Falling back to local PDF file")
                    with open(PDF_FALLBACK_PATH, 'rb') as f:
                        return io.BytesIO(f.read())
                raise RuntimeError(f"Failed to download PDF: {e}")


def _load_pdf() -> List[Dict]:
    """Download PDF and extract chunks with page numbers."""
    pdf_bytes = _download_pdf_bytes()
    reader = PdfReader(pdf_bytes)
    all_chunks = []

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
            if not text:
                continue

            # 1️⃣ First: extract dedicated URL chunks (high priority)
            url_chunks = _extract_url_chunks(text, page_idx)
            all_chunks.extend(url_chunks)

            # 2️⃣ Then: normal paragraph chunks
            para_chunks = _split_into_chunks(text, page_idx)
            all_chunks.extend(para_chunks)

        except Exception as e:
            logger.warning(f"Error processing page {page_idx}: {e}")
            continue

    # Remove duplicates (keep first occurrence — URL chunks come first so they're preserved)
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        signature = chunk["text"][:100].lower().strip()
        if signature not in seen and len(chunk["text"]) >= MIN_CHUNK_CHARS:
            seen.add(signature)
            unique_chunks.append(chunk)

    url_count = sum(1 for c in unique_chunks if c.get("is_url_chunk"))
    logger.info(f"Loaded {len(unique_chunks)} chunks ({url_count} URL chunks) from {len(reader.pages)} pages")
    return unique_chunks


@lru_cache(maxsize=1)
def _get_model():
    """Get the sentence transformer model (cached)."""
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _build_index():
    """Build the search index (cached in memory). Called lazily on first request."""
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
    """Clear cache and re-download + rebuild index."""
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
        url_count = sum(1 for c in chunks if c.get("is_url_chunk"))

        return {
            "total_chunks": len(chunks),
            "url_chunks": url_count,
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
        model = _get_model()
        results = retrieve_top_k("reporting tool url", k=1)

        return {
            "status": "healthy",
            "pdf_source": PDF_URL,
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