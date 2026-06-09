"""
Embeddings Interface — Phase 3.

Wraps calls to local Ollama Nomic Text Embedding model. Incorporates a
deterministic, reproducible fallback embedding generator for offline testing.
"""

from __future__ import annotations
import hashlib
import logging
import numpy as np
import requests
from config.settings import OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

EMBED_MODEL = "nomic-embed-text:latest"

def get_nomic_embedding(text: str) -> list[float]:
    """Retrieve text embedding from local Ollama nomic-embed-text model.

    Falls back to a deterministic pseudo-random normalized vector if Ollama is unreachable.

    Args:
        text: The string to embed.

    Returns:
        List of 768 float values representing the embedding.
    """
    if not text:
        text = "empty"

    try:
        # Try Ollama's newer /api/embed endpoint first
        url_embed = f"{OLLAMA_BASE_URL}/api/embed"
        payload_embed = {
            "model": EMBED_MODEL,
            "input": text
        }
        res = requests.post(url_embed, json=payload_embed, timeout=5)
        if res.status_code == 200:
            embeddings = res.json().get("embeddings")
            if embeddings and len(embeddings) > 0:
                return [float(x) for x in embeddings[0]]

        # Fallback to older /api/embeddings endpoint
        url_embeddings = f"{OLLAMA_BASE_URL}/api/embeddings"
        payload_embeddings = {
            "model": EMBED_MODEL,
            "prompt": text
        }
        res = requests.post(url_embeddings, json=payload_embeddings, timeout=5)
        if res.status_code == 200:
            embedding = res.json().get("embedding")
            if embedding:
                return [float(x) for x in embedding]

    except Exception as exc:
        logger.debug("Ollama embedding service unreachable, using fallback: %s", exc)

    # ── Deterministic Fallback Mock Embedding ──
    # Hashes the text to seed a numpy RNG, producing a normalized 768-dim vector.
    # This guarantees consistent vector search tests even when offline.
    sha256 = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(sha256[:4], byteorder="big")
    
    rng = np.random.default_rng(seed)
    vector = rng.normal(loc=0.0, scale=1.0, size=768)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
        
    return vector.tolist()
