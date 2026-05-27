from __future__ import annotations

import json
from pathlib import Path

from client_profiler.config import ProfilerConfig
from client_profiler.embeddings import LocalEmbedder, VectorRetriever
from client_profiler.extraction import OllamaClient
from client_profiler.ingestion import DocumentReader
from client_profiler.projects import ProjectAssociator, ProjectSummaryService
from client_profiler.projects.project_summary_service import ABSENT_ANSWER
from client_profiler.storage import SqliteStorage
from client_profiler.web.app import _build_client_projects, _enrich_client_documents


def quality_score(result: dict) -> float:
    if not result.get("ok"):
        return 0.0
    summary = str(result.get("summary") or "")
    runs = result.get("question_runs") or []
    if not isinstance(runs, list):
        runs = []

    total = max(1, len(runs))
    accepted = sum(1 for run in runs if str(run.get("status") or "") == "accepted")
    answers = result.get("answers") or {}
    absent = sum(1 for value in answers.values() if str(value) == ABSENT_ANSWER)

    length_component = min(len(summary) / 650.0, 1.0)
    accepted_component = accepted / total
    completeness_component = max(0.0, 1.0 - (absent / max(1, len(answers))))

    return round((0.45 * length_component) + (0.35 * accepted_component) + (0.20 * completeness_component), 4)


def run() -> None:
    config = ProfilerConfig()
    storage = SqliteStorage(config.db_path)
    reader = DocumentReader()
    llm = OllamaClient(config.ollama_base_url, config.llm_model, timeout=config.ollama_timeout_seconds)
    associator = ProjectAssociator(storage, llm)
    embedder = LocalEmbedder(config.embedding_model)
    retriever = VectorRetriever(storage)
    service = ProjectSummaryService(storage, embedder, retriever, llm, config.project_summary_questionnaire_path)

    projects_to_test: list[tuple[str, str, str, list[dict]]] = []
    for client in storage.list_clients():
        docs = _enrich_client_documents(storage.list_client_documents(client))
        projects = _build_client_projects(client, docs, reader, associator, storage, llm_available=True)
        for project in projects:
            projects_to_test.append((client, project["project_key"], project["project_name"], project["documents"]))
        if len(projects_to_test) >= 3:
            break

    projects_to_test = projects_to_test[:3]
    comparison_rows: list[dict] = []

    for client_name, project_key, project_name, documents in projects_to_test:
        non_batched = service.generate_and_store(
            client_name,
            project_key,
            project_name,
            documents,
            strategy="non_batched",
            store=False,
        )
        batched = service.generate_and_store(
            client_name,
            project_key,
            project_name,
            documents,
            strategy="batched",
            store=False,
        )

        row = {
            "client_name": client_name,
            "project_key": project_key,
            "project_name": project_name,
            "non_batched": {
                "ok": non_batched.get("ok"),
                "error_code": non_batched.get("error_code"),
                "summary_len": len(str(non_batched.get("summary") or "")),
                "quality_score": quality_score(non_batched),
            },
            "batched": {
                "ok": batched.get("ok"),
                "error_code": batched.get("error_code"),
                "summary_len": len(str(batched.get("summary") or "")),
                "quality_score": quality_score(batched),
            },
        }
        comparison_rows.append(row)

    report = {
        "model": config.llm_model,
        "timeout_seconds": config.ollama_timeout_seconds,
        "questionnaire_path": str(config.project_summary_questionnaire_path),
        "projects_tested": len(comparison_rows),
        "rows": comparison_rows,
    }

    out_path = Path("data") / "summary_strategy_comparison.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    run()
