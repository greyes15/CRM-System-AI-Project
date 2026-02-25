# clientmanager/rag_engine.py
import re
import os
import io
import numpy as np
from functools import lru_cache
from typing import List, Dict
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 🔧 CONFIGURATION
try:
    from django.conf import settings

    PDF_FALLBACK_PATH = os.path.join(
        settings.BASE_DIR,
        "clientmanager",
        "static",
        "clientmanager",
        "assets",
        "docs",
        "EMPLOYEE USER MANUAL.pdf",
    )
except Exception:
    PDF_FALLBACK_PATH = None

# Use OpenAI embeddings instead of sentence-transformers to avoid Render OOM
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

MIN_CHUNK_CHARS = 40
MAX_CHUNK_CHARS = 800

# Regex to detect URLs anywhere in a line
URL_RE = re.compile(r"https?://\S+")
# ─────────────────────────────────────────────


def _clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_url_chunks(text: str, page_num: int) -> List[Dict]:
    """
    Extract dedicated chunks for every line that contains a URL.
    This ensures URLs are never buried inside larger paragraphs and
    always get their own embedding so they score highly on URL queries.
    """
    url_chunks = []
    lines = text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if URL_RE.search(line):
            context_lines = []
            if i > 0 and lines[i - 1].strip():
                context_lines.append(lines[i - 1].strip())
            context_lines.append(line)
            if i + 1 < len(lines) and lines[i + 1].strip() and len(lines[i + 1].strip()) < 80:
                context_lines.append(lines[i + 1].strip())

            chunk_text = _clean_text(" ".join(context_lines))
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                url_chunks.append(
                    {"text": chunk_text, "page": page_num, "is_url_chunk": True}
                )
        i += 1

    return url_chunks


def _split_into_chunks(text: str, page_num: int) -> List[Dict]:
    """Split text into manageable chunks, preserving short lines like URLs."""
    lines = text.split("\n")
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
            sentences = re.split(r"(?<=[.!?])\s+", para)
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


def _load_pdf_from_disk() -> io.BytesIO:
    """
    Load the PDF from disk only (Render-safe; avoids 0-byte HTTP fetch issues).
    """
    if not PDF_FALLBACK_PATH:
        raise RuntimeError("PDF_FALLBACK_PATH could not be constructed (settings unavailable).")

    if not os.path.exists(PDF_FALLBACK_PATH):
        raise FileNotFoundError(f"PDF not found at: {PDF_FALLBACK_PATH}")

    with open(PDF_FALLBACK_PATH, "rb") as f:
        content = f.read()

    if not content.startswith(b"%PDF"):
        raise ValueError(f"Local file is not a valid PDF (starts with: {content[:20]})")

    logger.info(f"Loaded PDF from disk: {PDF_FALLBACK_PATH} ({len(content):,} bytes)")
    return io.BytesIO(content)


def _load_pdf() -> List[Dict]:
    """Load PDF from disk and extract chunks with page numbers."""
    pdf_bytes = _load_pdf_from_disk()
    reader = PdfReader(pdf_bytes)
    all_chunks = []

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
            if not text:
                continue

            # 1️⃣ Dedicated URL chunks first
            url_chunks = _extract_url_chunks(text, page_idx)
            all_chunks.extend(url_chunks)

            # 2️⃣ Normal paragraph chunks
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


def _cosine_sim_matrix(embeddings: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """
    embeddings: (n, d) normalized
    query_vec: (d,) normalized
    returns: (n,) cosine similarities via dot product
    """
    return np.dot(embeddings, query_vec)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return x / n


@lru_cache(maxsize=1)
def _get_openai_client():
    """
    Cached OpenAI client.
    Supports both new and older SDKs.
    """
    # New SDK: from openai import OpenAI
    try:
        from openai import OpenAI  # type: ignore

        return OpenAI()
    except Exception:
        pass

    # Older SDK: import openai
    try:
        import openai  # type: ignore

        return openai
    except Exception as e:
        raise RuntimeError(
            "OpenAI SDK not installed. Add `openai` to requirements.txt."
        ) from e


def _embed_texts_openai(texts: List[str]) -> np.ndarray:
    """
    Create embeddings using OpenAI (no local model).
    Returns float32 matrix (n, d), normalized.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    client = _get_openai_client()

    # New SDK path
    if hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
        vecs = [d.embedding for d in resp.data]
        mat = np.array(vecs, dtype=np.float32)
        return _normalize_rows(mat)

    # Older SDK path
    if hasattr(client, "Embedding") and hasattr(client.Embedding, "create"):
        resp = client.Embedding.create(model=OPENAI_EMBED_MODEL, input=texts)
        vecs = [d["embedding"] for d in resp["data"]]
        mat = np.array(vecs, dtype=np.float32)
        return _normalize_rows(mat)

    raise RuntimeError("Unsupported OpenAI client; could not call embeddings API.")


@lru_cache(maxsize=1)
def _build_index():
    """
    Build the search index (cached in memory). Called lazily on first request.
    Uses OpenAI embeddings to avoid Render OOM from sentence-transformers.
    """
    try:
        chunks = _load_pdf()

        texts = [chunk["text"] for chunk in chunks]
        logger.info(f"Creating OpenAI embeddings for {len(texts)} chunks using {OPENAI_EMBED_MODEL}...")
        embeddings = _embed_texts_openai(texts)

        logger.info("Index built successfully (OpenAI embeddings)")
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

        query_vec = _embed_texts_openai([query.strip()])[0]
        query_vec = _normalize_vec(query_vec.astype(np.float32, copy=False))

        similarities = _cosine_sim_matrix(embeddings, query_vec)
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "text": chunks[idx]["text"],
                    "page": chunks[idx]["page"],
                    "score": float(similarities[idx]),
                }
            )

        logger.debug(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
        return results

    except Exception as e:
        logger.error(f"Error in retrieve_top_k: {e}")
        return []


def refresh_cache() -> int:
    """Clear cache and reload + rebuild index."""
    try:
        _build_index.cache_clear()
        # _get_openai_client is safe to keep cached, but clearing it is fine too
        _get_openai_client.cache_clear()

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
        avg_length = np.mean([len(chunk["text"]) for chunk in chunks]) if chunks else 0
        url_count = sum(1 for c in chunks if c.get("is_url_chunk"))

        return {
            "total_chunks": len(chunks),
            "url_chunks": url_count,
            "total_pages": len(pages),
            "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
            "avg_chunk_length": int(avg_length),
            "model_name": OPENAI_EMBED_MODEL,
            "pdf_source": PDF_FALLBACK_PATH,
        }
    except Exception as e:
        logger.error(f"Error getting index info: {e}")
        return {"error": str(e)}


def health_check() -> Dict:
    """Check if the RAG engine is working properly."""
    try:
        # sanity: can we build index + retrieve?
        results = retrieve_top_k("reporting tool url", k=1)

        return {
            "status": "healthy",
            "pdf_source": PDF_FALLBACK_PATH,
            "embedding_model": OPENAI_EMBED_MODEL,
            "can_retrieve": len(results) > 0,
            "index_info": get_index_info(),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
            "pdf_source": PDF_FALLBACK_PATH,
            "embedding_model": OPENAI_EMBED_MODEL,
        }