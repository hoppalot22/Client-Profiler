from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._backend = None
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            # Load from local cache only so offline/corporate-network environments
            # fail fast and use deterministic fallback embeddings.
            self._backend = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self._backend = None

    def embed_text(self, text: str) -> list[float]:
        if self._backend is not None:
            vector = self._backend.encode([text], convert_to_numpy=True)[0]
            return vector.astype(float).tolist()
        return self._hash_embedding(text)

    def embed_chunks(self, chunks: Iterable[str]) -> list[list[float]]:
        chunks_list = list(chunks)
        if not chunks_list:
            return []

        if self._backend is not None:
            vectors = self._backend.encode(chunks_list, convert_to_numpy=True)
            return [v.astype(float).tolist() for v in vectors]

        return [self._hash_embedding(chunk) for chunk in chunks_list]

    def _hash_embedding(self, text: str, dim: int = 384) -> list[float]:
        raw = text.encode("utf-8", errors="ignore")
        digest = hashlib.sha256(raw).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, size=(dim,))
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist()
        return (vec / norm).tolist()
