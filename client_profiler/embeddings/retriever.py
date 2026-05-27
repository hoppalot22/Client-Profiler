from __future__ import annotations

import numpy as np

from client_profiler.storage import SqliteStorage


class VectorRetriever:
    def __init__(self, storage: SqliteStorage) -> None:
        self.storage = storage

    def search(self, query_embedding: list[float], top_k: int = 5, client_name: str | None = None) -> list[dict]:
        rows = self.storage.fetch_vectors(client_name=client_name)
        if not rows:
            return []

        query = np.array(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query) + 1e-12

        scored: list[tuple[float, dict]] = []
        for row in rows:
            emb = np.array(row["embedding"], dtype=float)
            denom = (np.linalg.norm(emb) + 1e-12) * query_norm
            score = float(np.dot(query, emb) / denom)
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "score": score,
                "chunk_text": row["chunk_text"],
                "source_document": row["source_document"],
                "metadata": row["metadata"],
                "client_name": row["client_name"],
            }
            for score, row in scored[:top_k]
        ]
