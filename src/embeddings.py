"""
embeddings.py

Shared Ollama/nomic-embed-text embedding helper.

Extracted from rag_pipeline.py so multiple backends (FAISS, Weaviate, etc.)
can use the same embedding logic without circular imports.
"""

import json
import os
import urllib.request

import numpy as np
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def embed_text(text: str, task_prefix: str = "search_query:") -> np.ndarray:
    """
    Embed a single string via Ollama/nomic-embed-text.

    Args:
        text: the text to embed.
        task_prefix: nomic-embed-text expects "search_query:" for queries and
            "search_document:" for documents.  Defaults to search_query.

    Returns:
        L2-normalised 2-D array of shape (1, dim).
    """
    payload = json.dumps({
        "model": EMBED_MODEL,
        "prompt": f"{task_prefix} {text}",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read())
        vec = np.array([result["embedding"]], dtype=np.float32)
        # nomic-embed-text is already normalised, but re-normalise to be safe
        # for cosine-similarity indexes.
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        if norm > 0:
            vec = vec / norm
        return vec


def embed_query(text: str) -> np.ndarray:
    """Embed a query string (search_query: prefix)."""
    return embed_text(text, task_prefix="search_query:")


def embed_document(text: str) -> np.ndarray:
    """Embed a document/source phrase (search_document: prefix)."""
    return embed_text(text, task_prefix="search_document:")
