from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DurationStat:
    path: str
    seconds: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze JSONL ingest event logs and DB quality signals")
    parser.add_argument("--event-log", type=Path, required=True, help="Path to ingest event JSONL file")
    parser.add_argument("--db", type=Path, default=None, help="Optional profiler SQLite DB path for quality checks")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON report path")
    parser.add_argument("--max-samples", type=int, default=20, help="Max anomaly examples to include")
    return parser.parse_args()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    idx = min(max(idx, 0), len(ordered) - 1)
    return float(ordered[idx])


def _analyze_events(event_log: Path) -> dict[str, Any]:
    if not event_log.exists():
        raise FileNotFoundError(f"Event log not found: {event_log}")

    event_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    file_starts: dict[str, datetime] = {}
    durations: list[DurationStat] = []
    errors: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    with event_log.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = str(event.get("event") or "").strip() or "unknown"
            event_counts[kind] += 1

            path = str(event.get("path") or "").strip()
            ts = _parse_ts(str(event.get("ts") or ""))

            if kind == "file_started" and path and ts is not None:
                file_starts[path] = ts

            if kind in {"file_completed", "file_error"} and path:
                started = file_starts.get(path)
                if started is not None and ts is not None:
                    durations.append(DurationStat(path=path, seconds=max(0.0, (ts - started).total_seconds())))

            if kind == "file_completed":
                status = str(event.get("status") or "unknown")
                status_counts[status] += 1

            if kind == "file_error":
                errors.append(
                    {
                        "path": path,
                        "error": str(event.get("error") or ""),
                        "ts": str(event.get("ts") or ""),
                    }
                )
            if kind == "file_unsupported":
                unsupported.append(
                    {
                        "path": path,
                        "reason": str(event.get("error") or ""),
                        "ts": str(event.get("ts") or ""),
                    }
                )

    latency_values = [row.seconds for row in durations]
    return {
        "event_counts": dict(event_counts),
        "file_status_counts": dict(status_counts),
        "file_duration_seconds": {
            "count": len(latency_values),
            "avg": round(sum(latency_values) / max(1, len(latency_values)), 3),
            "p50": round(_percentile(latency_values, 0.50), 3),
            "p95": round(_percentile(latency_values, 0.95), 3),
            "slowest": [
                {"path": row.path, "seconds": round(row.seconds, 3)}
                for row in sorted(durations, key=lambda x: x.seconds, reverse=True)[:10]
            ],
        },
        "errors": errors,
        "unsupported_files": unsupported,
    }


def _query_one(cur: sqlite3.Cursor, sql: str) -> int:
    return int(cur.execute(sql).fetchone()[0])


def _db_quality(db_path: Path, max_samples: int) -> dict[str, Any]:
    if not db_path.exists():
        return {"db_found": False, "db_path": str(db_path)}

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        counts = {
            "documents": _query_one(cur, "SELECT COUNT(*) FROM documents"),
            "profiles": _query_one(cur, "SELECT COUNT(*) FROM profiles"),
            "timeline": _query_one(cur, "SELECT COUNT(*) FROM timeline"),
            "vectors": _query_one(cur, "SELECT COUNT(*) FROM vectors"),
            "null_client_docs": _query_one(
                cur,
                "SELECT COUNT(*) FROM documents WHERE json_extract(metadata_json, '$.client_name') IS NULL",
            ),
            "report_docs_missing_date": _query_one(
                cur,
                """
                SELECT COUNT(*)
                FROM documents
                WHERE COALESCE(json_extract(metadata_json, '$.document_kind'), '') = 'report'
                  AND COALESCE(json_extract(metadata_json, '$.report_date'), '') = ''
                """,
            ),
        }

        kind_counts = dict(
            cur.execute(
                """
                SELECT COALESCE(json_extract(metadata_json, '$.document_kind'), 'unknown') AS kind,
                       COUNT(*)
                FROM documents
                GROUP BY kind
                ORDER BY COUNT(*) DESC
                """
            ).fetchall()
        )

        mismatch_rows = cur.execute(
            """
            SELECT source_path,
                   COALESCE(json_extract(metadata_json, '$.document_kind'), 'unknown') AS kind
            FROM documents
            WHERE (
                LOWER(source_path) LIKE '%quote%' AND kind <> 'quote'
            ) OR (
                LOWER(source_path) LIKE '%purchase_order%' AND kind <> 'purchase_order'
            ) OR (
                LOWER(source_path) LIKE '%email_chain%' AND kind <> 'email_chain'
            ) OR (
                LOWER(source_path) LIKE '%access_request%' AND kind <> 'access_request'
            )
            ORDER BY source_path
            LIMIT ?
            """,
            (max_samples,),
        ).fetchall()

        client_rows = cur.execute(
            """
            SELECT DISTINCT TRIM(COALESCE(json_extract(metadata_json, '$.client_name'), ''))
            FROM documents
            WHERE TRIM(COALESCE(json_extract(metadata_json, '$.client_name'), '')) <> ''
            """
        ).fetchall()

    by_canonical: dict[str, set[str]] = defaultdict(set)
    for row in client_rows:
        value = str(row[0]).strip()
        if value:
            by_canonical[value.casefold()].add(value)

    canonical_collisions = [
        sorted(list(names))
        for names in by_canonical.values()
        if len(names) > 1
    ]

    return {
        "db_found": True,
        "db_path": str(db_path),
        "counts": counts,
        "document_kind_counts": kind_counts,
        "filename_kind_mismatches": [
            {"path": str(row[0]), "document_kind": str(row[1])}
            for row in mismatch_rows
        ],
        "client_name_canonical_collisions": canonical_collisions,
    }


def main() -> None:
    args = _parse_args()

    event_summary = _analyze_events(args.event_log)
    db_summary = _db_quality(args.db, args.max_samples) if args.db is not None else None

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event_log": str(args.event_log),
        "event_summary": event_summary,
    }
    if db_summary is not None:
        report["db_quality"] = db_summary

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
