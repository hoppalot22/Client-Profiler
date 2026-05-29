from __future__ import annotations

import json
import math
import shutil
import sqlite3
import statistics
import gc
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from client_profiler.config import ProfilerConfig
from client_profiler.embeddings import LocalEmbedder, VectorRetriever
from client_profiler.extraction import OllamaClient
from client_profiler.ingestion import DocumentReader
from client_profiler.projects.project_associator import ProjectAssociator
from client_profiler.projects.project_field_service import ProjectFieldService
from client_profiler.storage import SqliteStorage
from client_profiler.web.app import _build_client_projects, _enrich_client_documents


MODELS = ["qwen2.5:1.5b", "qwen2.5:0.5b", "qwen3:0.6b"]
MAX_CASES = 9
MAX_PROJECTS = 8


@dataclass
class BenchmarkCase:
    client_name: str
    project_key: str
    project_name: str
    field_key: str
    expected_status: str
    expected_value: str


def _token_set(text: str) -> set[str]:
    return {token for token in str(text or "").lower().replace("/", " ").replace("-", " ").split() if token}


def _value_similarity(expected: str, actual: str) -> float:
    left = _token_set(expected)
    right = _token_set(actual)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(right)
    recall = overlap / len(left)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _case_score(case: BenchmarkCase, result: dict[str, Any]) -> float:
    if not result.get("ok"):
        return 0.0

    actual_status = str(result.get("status") or "").strip()
    status_match = actual_status == case.expected_status
    score = 0.6 if status_match else 0.0

    if status_match and case.expected_status == "absent":
        score += 0.3
    elif status_match and case.expected_status == "filled":
        score += 0.3 * _value_similarity(case.expected_value, str(result.get("value") or ""))

    if str(result.get("method") or "").strip().lower() == "ai":
        score += 0.1

    return min(score, 1.0)


def _load_cases(db_path: Path) -> list[BenchmarkCase]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT client_name, project_key, project_name, fields_json
            FROM project_key_fields
            ORDER BY updated_at DESC
            """
        ).fetchall()

    cases: list[BenchmarkCase] = []
    project_seen: set[tuple[str, str]] = set()
    for client_name, project_key, project_name, fields_json in rows:
        project_id = (str(client_name), str(project_key))
        if project_id not in project_seen and len(project_seen) >= MAX_PROJECTS:
            continue

        try:
            fields = json.loads(fields_json or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(fields, dict):
            continue

        project_added = False
        for field_key, payload in fields.items():
            if len(cases) >= MAX_CASES:
                return cases
            if not isinstance(payload, dict):
                continue
            if str(payload.get("method") or "").strip().lower() != "ai":
                continue
            expected_status = str(payload.get("status") or "").strip()
            expected_value = str(payload.get("value") or "").strip()
            if expected_status not in {"filled", "absent"}:
                continue
            cases.append(
                BenchmarkCase(
                    client_name=str(client_name),
                    project_key=str(project_key),
                    project_name=str(project_name or ""),
                    field_key=str(field_key),
                    expected_status=expected_status,
                    expected_value=expected_value,
                )
            )
            project_added = True
        if project_added:
            project_seen.add(project_id)
    return cases


def _prepare_project_lookup(
    storage: SqliteStorage,
    reader: DocumentReader,
    project_associator: ProjectAssociator,
    field_service: ProjectFieldService,
    client_names: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for client_name in sorted(client_names):
        documents = _enrich_client_documents(storage.list_client_documents(client_name))
        projects = _build_client_projects(
            client_name,
            documents,
            reader,
            project_associator,
            storage,
            field_service,
            llm_available=True,
        )
        for project in projects:
            lookup[(client_name, str(project.get("project_key") or ""))] = project
    return lookup


def _evaluate_model(config: ProfilerConfig, model_name: str, cases: list[BenchmarkCase]) -> dict[str, Any]:
    print(f"[benchmark] starting {model_name} on {len(cases)} cases", flush=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="cp-model-bench-"))
    try:
        temp_db = tmp_dir / "profiler.db"
        shutil.copy2(config.db_path, temp_db)

        storage = SqliteStorage(temp_db)
        llm = OllamaClient(config.ollama_base_url, model_name, timeout=max(config.ollama_timeout_seconds, 90))
        embedder = LocalEmbedder(config.embedding_model)
        retriever = VectorRetriever(storage)
        field_service = ProjectFieldService(
            storage=storage,
            embedder=embedder,
            retriever=retriever,
            llm=llm,
            key_fields_path=config.project_key_fields_path,
            debug_enabled=False,
        )
        reader = DocumentReader()
        project_associator = ProjectAssociator(storage, llm)
        project_lookup = _prepare_project_lookup(
            storage,
            reader,
            project_associator,
            field_service,
            {case.client_name for case in cases},
        )
        definitions = {row["key"]: row for row in field_service.field_definitions()}

        case_results: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            project = project_lookup.get((case.client_name, case.project_key))
            definition = definitions.get(case.field_key)
            print(f"[benchmark] {model_name} case {index}/{len(cases)}: {case.client_name} {case.project_key} {case.field_key}", flush=True)
            if project is None or definition is None:
                case_results.append(
                    {
                        "client_name": case.client_name,
                        "project_key": case.project_key,
                        "field_key": case.field_key,
                        "ok": False,
                        "error": "project_or_field_missing",
                        "score": 0.0,
                    }
                )
                continue

            existing = {
                row["key"]: {
                    "value": row.get("value") or "",
                    "status": row.get("status") or "",
                }
                for row in project.get("key_fields", [])
            }

            started = time.perf_counter()
            result = field_service.generate_field(
                client_name=case.client_name,
                project_key=case.project_key,
                project_name=case.project_name,
                field_key=case.field_key,
                field_prompt=definition["prompt"],
                documents=project["documents"],
                existing_fields=existing,
            )
            elapsed = time.perf_counter() - started
            score = _case_score(case, result)
            case_results.append(
                {
                    "client_name": case.client_name,
                    "project_key": case.project_key,
                    "field_key": case.field_key,
                    "expected_status": case.expected_status,
                    "expected_value": case.expected_value,
                    "ok": bool(result.get("ok")),
                    "status": result.get("status"),
                    "value": result.get("value"),
                    "method": result.get("method"),
                    "error_code": result.get("error_code"),
                    "error_message": result.get("error_message"),
                    "latency_seconds": round(elapsed, 3),
                    "score": round(score, 4),
                }
            )

        del project_lookup
        del project_associator
        del reader
        del field_service
        del retriever
        del embedder
        del llm
        del storage
        gc.collect()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    successful = [row for row in case_results if row.get("ok")]
    filled_cases = [row for row in successful if row.get("expected_status") == "filled"]
    ai_cases = [row for row in successful if str(row.get("method") or "").strip().lower() == "ai"]
    summary = {
        "model": model_name,
        "case_count": len(case_results),
        "ok_rate": round(len(successful) / len(case_results), 4) if case_results else 0.0,
        "ai_method_rate": round(len(ai_cases) / len(case_results), 4) if case_results else 0.0,
        "status_accuracy": round(
            sum(1 for row in successful if row.get("status") == row.get("expected_status")) / len(case_results), 4
        )
        if case_results
        else 0.0,
        "avg_score": round(statistics.mean(row["score"] for row in case_results), 4) if case_results else 0.0,
        "avg_latency_seconds": round(statistics.mean(row["latency_seconds"] for row in case_results), 3) if case_results else math.inf,
        "filled_value_similarity": round(
            statistics.mean(
                _value_similarity(str(row.get("expected_value") or ""), str(row.get("value") or ""))
                for row in filled_cases
                if str(row.get("status") or "") == "filled"
            ),
            4,
        )
        if any(str(row.get("status") or "") == "filled" for row in filled_cases)
        else 0.0,
        "cases": case_results,
    }
    print(
        f"[benchmark] completed {model_name}: avg_score={summary['avg_score']} status_accuracy={summary['status_accuracy']} ai_rate={summary['ai_method_rate']} avg_latency={summary['avg_latency_seconds']}",
        flush=True,
    )
    return summary


def main() -> None:
    config = ProfilerConfig()
    max_cases = int(os.environ.get("CP_BENCH_MAX_CASES", str(MAX_CASES)) or MAX_CASES)
    requested_models_raw = str(os.environ.get("CP_BENCH_MODELS", "") or "").strip()
    models = [item.strip() for item in requested_models_raw.split(",") if item.strip()] or MODELS
    cases = _load_cases(config.db_path)
    cases = cases[:max_cases]
    if not cases:
        raise SystemExit("No benchmark cases found in project_key_fields.")

    results = [_evaluate_model(config, model_name, cases) for model_name in models]
    ranked = sorted(
        results,
        key=lambda row: (
            row["avg_score"],
            row["status_accuracy"],
            row["ai_method_rate"],
            -row["avg_latency_seconds"],
        ),
        reverse=True,
    )
    print(
        json.dumps(
            {
                "cases_used": len(cases),
                "models": [
                    {
                        "model": row["model"],
                        "case_count": row["case_count"],
                        "ok_rate": row["ok_rate"],
                        "ai_method_rate": row["ai_method_rate"],
                        "status_accuracy": row["status_accuracy"],
                        "avg_score": row["avg_score"],
                        "avg_latency_seconds": row["avg_latency_seconds"],
                        "filled_value_similarity": row["filled_value_similarity"],
                    }
                    for row in ranked
                ],
                "best_model": ranked[0]["model"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()