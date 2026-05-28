from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from client_profiler import ClientProfiler, ProfilerConfig
from client_profiler.embeddings import VectorRetriever


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against a labeled set")
    parser.add_argument("--dataset", type=Path, default=Path("./data/retrieval_eval_set.json"))
    parser.add_argument("--db", type=Path, default=Path("./data/profiler.db"))
    parser.add_argument("--model", type=str, default="qwen2.5:0.5b")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--show-failures", action="store_true")
    return parser


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Dataset must be a JSON array")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        expected = item.get("expected_sources") or []
        if not query or not isinstance(expected, list) or not expected:
            continue
        rows.append(item)
    if not rows:
        raise ValueError("Dataset has no valid rows")
    return rows


def _basename(value: str) -> str:
    return Path(str(value or "")).name.lower()


def main() -> None:
    args = _build_parser().parse_args()

    config = ProfilerConfig(
        db_path=args.db,
        data_dir=args.db.parent,
        llm_model=args.model,
        ollama_base_url=args.ollama_url,
    )
    profiler = ClientProfiler(config)
    retriever = VectorRetriever(profiler.storage)

    dataset = _load_dataset(args.dataset)

    total = len(dataset)
    hit_at_k = 0
    precision_sum = 0.0
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    failures: list[dict[str, Any]] = []

    for row in dataset:
        query = str(row.get("query") or "").strip()
        client_name = str(row.get("client_name") or "").strip() or None
        source_documents = row.get("source_documents") if isinstance(row.get("source_documents"), list) else None
        metadata_filters = row.get("metadata_filters") if isinstance(row.get("metadata_filters"), dict) else None

        expected_set = {_basename(s) for s in row.get("expected_sources", []) if str(s).strip()}
        query_embedding = profiler.embedder.embed_text(query)
        hits = retriever.search(
            query_embedding,
            top_k=args.top_k,
            client_name=client_name,
            source_documents=source_documents,
            metadata_filters=metadata_filters,
            query_text=query,
            hybrid_alpha=0.78,
            use_mmr=True,
            mmr_lambda=0.75,
            candidate_pool=max(60, args.top_k * 8),
        )

        predicted = [_basename(hit.get("source_document") or "") for hit in hits]
        matches = [name for name in predicted if name in expected_set]
        unique_matches = set(matches)

        has_hit = len(unique_matches) > 0
        if has_hit:
            hit_at_k += 1

        precision = len(unique_matches) / max(1, len(predicted))
        recall = len(unique_matches) / max(1, len(expected_set))
        precision_sum += precision
        recall_sum += recall

        rr = 0.0
        for idx, name in enumerate(predicted, start=1):
            if name in expected_set:
                rr = 1.0 / idx
                break
        reciprocal_rank_sum += rr

        if not has_hit:
            failures.append(
                {
                    "id": row.get("id"),
                    "query": query,
                    "expected_sources": sorted(expected_set),
                    "top_results": predicted[: args.top_k],
                }
            )

    metrics = {
        "dataset_size": total,
        "top_k": args.top_k,
        "hit_rate_at_k": round(hit_at_k / max(1, total), 4),
        "mean_precision_at_k": round(precision_sum / max(1, total), 4),
        "mean_recall_at_k": round(recall_sum / max(1, total), 4),
        "mrr_at_k": round(reciprocal_rank_sum / max(1, total), 4),
        "miss_count": len(failures),
    }

    print(json.dumps(metrics, indent=2))
    if args.show_failures and failures:
        print("\nMisses:")
        print(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
