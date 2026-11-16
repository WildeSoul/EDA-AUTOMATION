"""
Embeddings Interface — Phase 3.

Wraps calls to local Ollama Nomic Text Embedding model. Incorporates a
deterministic, reproducible fallback embedding generator for offline testing.

Optimised with a fast socket-level reachability check and a short TTL cache
to avoid 4-second TCP connection timeouts when Ollama is offline.
"""

from __future__ import annotations
import hashlib
import logging
import socket
import time
import numpy as np
import requests
from config.settings import OLLAMA_BASE_URL
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

EMBED_MODEL = "nomic-embed-text:latest"

# ── Fast Ollama Reachability Cache ──────────────────────────────────────
# On Windows, a TCP connect to a closed port takes ~4 seconds to timeout.
# We cache the reachability result for _REACHABLE_TTL seconds so that
# subsequent calls within the TTL window skip the network round-trip
# entirely and jump straight to the deterministic fallback.
_ollama_reachable: bool | None = None
_ollama_checked_at: float = 0.0
_REACHABLE_TTL = 15.0  # seconds


def _is_ollama_reachable() -> bool:
    """Fast socket-level reachability check with TTL cache."""
    global _ollama_reachable, _ollama_checked_at

    now = time.monotonic()
    if _ollama_reachable is not None and (now - _ollama_checked_at) < _REACHABLE_TTL:
        return _ollama_reachable

    parsed = urlparse(OLLAMA_BASE_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434

    try:
        with socket.create_connection((host, port), timeout=0.3):
            _ollama_reachable = True
    except (OSError, socket.timeout):
        _ollama_reachable = False

    _ollama_checked_at = now
    return _ollama_reachable


def get_nomic_embedding(text: str) -> list[float]:
    """Retrieve text embedding from local Ollama nomic-embed-text model.

    Falls back to a deterministic pseudo-random normalized vector if Ollama is unreachable.

    Args:
        text: The string to embed.

    Returns:
        List of 768 float values representing the embedding.
    """
    global _ollama_reachable, _ollama_checked_at
    if not text:
        text = "empty"

    # ── Fast-path: skip HTTP calls if Ollama is known to be offline ──
    if _is_ollama_reachable():
        try:
            # Try Ollama's newer /api/embed endpoint first
            url_embed = f"{OLLAMA_BASE_URL}/api/embed"
            payload_embed = {
                "model": EMBED_MODEL,
                "input": text
            }
            res = requests.post(url_embed, json=payload_embed, timeout=1.0)
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
            res = requests.post(url_embeddings, json=payload_embeddings, timeout=1.0)
            if res.status_code == 200:
                embedding = res.json().get("embedding")
                if embedding:
                    return [float(x) for x in embedding]

        except Exception as exc:
            logger.debug("Ollama embedding service unreachable or timed out, caching offline status: %s", exc)
            _ollama_reachable = False
            _ollama_checked_at = time.monotonic()
    else:
        logger.debug("Ollama offline (cached), using deterministic fallback embedding.")

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
