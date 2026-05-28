from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from client_profiler.storage import SqliteStorage


class VectorRetriever:
    def __init__(self, storage: SqliteStorage) -> None:
        self.storage = storage

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        client_name: str | None = None,
        source_documents: list[str] | None = None,
        metadata_filters: dict | None = None,
        query_text: str | None = None,
        hybrid_alpha: float = 0.78,
        use_mmr: bool = True,
        mmr_lambda: float = 0.75,
        candidate_pool: int = 60,
    ) -> list[dict]:
        rows = self.storage.fetch_vectors(
            client_name=client_name,
            source_documents=source_documents,
            metadata_filters=metadata_filters,
        )
        if not rows:
            return []

        query = np.array(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query) + 1e-12

        candidate_limit = max(top_k, candidate_pool)
        scored: list[tuple[float, dict]] = []
        for row in rows:
            emb = np.array(row["embedding"], dtype=float)
            denom = (np.linalg.norm(emb) + 1e-12) * query_norm
            score = float(np.dot(query, emb) / denom)
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:candidate_limit]

        query_tokens = self._tokenize(query_text or "")
        lexical_scores = self._lexical_scores([row for _, row in scored], query_tokens) if query_tokens else [0.0] * len(scored)

        ranked: list[dict] = []
        for idx, (semantic, row) in enumerate(scored):
            semantic_norm = (semantic + 1.0) / 2.0
            lexical_norm = lexical_scores[idx]
            final_score = (hybrid_alpha * semantic_norm) + ((1.0 - hybrid_alpha) * lexical_norm)
            ranked.append(
                {
                    "score": float(final_score),
                    "semantic_score": float(semantic),
                    "lexical_score": float(lexical_norm),
                    "chunk_text": row["chunk_text"],
                    "source_document": row["source_document"],
                    "metadata": row["metadata"],
                    "client_name": row["client_name"],
                    "embedding": row["embedding"],
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        if use_mmr:
            ranked = self._mmr_select(ranked, top_k=top_k, mmr_lambda=mmr_lambda)
        else:
            ranked = ranked[:top_k]

        out: list[dict] = []
        for item in ranked:
            out.append(
                {
                    "score": item["score"],
                    "semantic_score": item["semantic_score"],
                    "lexical_score": item["lexical_score"],
                    "chunk_text": item["chunk_text"],
                    "source_document": item["source_document"],
                    "metadata": item["metadata"],
                    "client_name": item["client_name"],
                }
            )
        return out

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]{2,}", str(text or "").lower())
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "what",
            "when",
            "where",
            "which",
            "about",
            "project",
            "client",
            "field",
        }
        return [token for token in tokens if token not in stopwords]

    def _lexical_scores(self, rows: list[dict], query_tokens: list[str]) -> list[float]:
        if not rows or not query_tokens:
            return [0.0 for _ in rows]

        docs: list[list[str]] = []
        for row in rows:
            metadata = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
            meta_text = " ".join(
                [
                    str(metadata.get("title") or ""),
                    str(metadata.get("project_name") or ""),
                    str(metadata.get("project_code") or ""),
                    str(metadata.get("report_type") or ""),
                ]
            )
            docs.append(self._tokenize(f"{row.get('chunk_text') or ''} {meta_text}"))

        query_counter = Counter(query_tokens)
        df_counter: Counter[str] = Counter()
        for tokens in docs:
            for token in set(tokens):
                if token in query_counter:
                    df_counter[token] += 1

        n_docs = max(1, len(docs))
        avg_len = sum(len(tokens) for tokens in docs) / n_docs
        k1 = 1.4
        b = 0.72

        scores: list[float] = []
        for tokens in docs:
            tf = Counter(tokens)
            doc_len = max(1, len(tokens))
            total = 0.0
            for term, qtf in query_counter.items():
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                df = df_counter.get(term, 0)
                idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
                denom = f + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_len)))
                total += idf * ((f * (k1 + 1.0)) / max(1e-9, denom)) * (1.0 + 0.05 * max(0, qtf - 1))
            scores.append(total)

        max_score = max(scores) if scores else 0.0
        if max_score <= 1e-9:
            return [0.0 for _ in scores]
        return [float(score / max_score) for score in scores]

    def _mmr_select(self, ranked: list[dict], top_k: int, mmr_lambda: float) -> list[dict]:
        if len(ranked) <= top_k:
            return ranked

        selected: list[dict] = []
        candidates = list(ranked)

        # Seed with highest relevance.
        selected.append(candidates.pop(0))

        while candidates and len(selected) < top_k:
            best_index = 0
            best_mmr = -1e9
            for idx, item in enumerate(candidates):
                rel = float(item.get("score") or 0.0)
                emb = np.array(item.get("embedding") or [], dtype=float)
                if emb.size == 0:
                    similarity_penalty = 0.0
                else:
                    similarity_penalty = 0.0
                    for chosen in selected:
                        chosen_emb = np.array(chosen.get("embedding") or [], dtype=float)
                        if chosen_emb.size == 0:
                            continue
                        denom = (np.linalg.norm(emb) + 1e-12) * (np.linalg.norm(chosen_emb) + 1e-12)
                        sim = float(np.dot(emb, chosen_emb) / denom)
                        if sim > similarity_penalty:
                            similarity_penalty = sim
                mmr = (mmr_lambda * rel) - ((1.0 - mmr_lambda) * similarity_penalty)
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_index = idx

            selected.append(candidates.pop(best_index))

        return selected
