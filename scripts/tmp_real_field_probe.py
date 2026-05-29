from __future__ import annotations

import json
import sys
import time

from client_profiler.config import ProfilerConfig
from client_profiler.embeddings import LocalEmbedder, VectorRetriever
from client_profiler.extraction import OllamaClient
from client_profiler.ingestion import DocumentReader
from client_profiler.projects.project_associator import ProjectAssociator
from client_profiler.projects.project_field_service import ProjectFieldService
from client_profiler.storage import SqliteStorage
from client_profiler.web.app import _build_client_projects, _enrich_client_documents


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b"
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    field_key = sys.argv[3] if len(sys.argv) > 3 else "title"

    cfg = ProfilerConfig()
    storage = SqliteStorage(cfg.db_path)
    llm = OllamaClient(cfg.ollama_base_url, model, timeout=timeout)
    embedder = LocalEmbedder(cfg.embedding_model)
    retriever = VectorRetriever(storage)
    field_service = ProjectFieldService(
        storage=storage,
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        key_fields_path=cfg.project_key_fields_path,
        debug_enabled=False,
    )

    client_name = "Acme Refining"
    project_key = "ar-inv-2026"

    docs = _enrich_client_documents(storage.list_client_documents(client_name))
    projects = _build_client_projects(
        client_name,
        docs,
        DocumentReader(),
        ProjectAssociator(storage, llm),
        storage,
        field_service,
        llm_available=True,
    )
    project = next((p for p in projects if p.get("project_key") == project_key), None)
    if project is None:
        raise SystemExit("Project not found")

    defs = {row["key"]: row for row in field_service.field_definitions()}
    field_def = defs[field_key]

    existing = {
        row["key"]: {
            "value": row.get("value") or "",
            "status": row.get("status") or "",
        }
        for row in project.get("key_fields", [])
    }

    start = time.perf_counter()
    result = field_service.generate_field(
        client_name=client_name,
        project_key=project_key,
        project_name=str(project.get("project_name") or ""),
        field_key=field_key,
        field_prompt=field_def["prompt"],
        documents=project["documents"],
        existing_fields=existing,
    )
    elapsed = time.perf_counter() - start

    print(
        json.dumps(
            {
                "model": model,
                "field": field_key,
                "timeout_seconds": timeout,
                "elapsed_seconds": round(elapsed, 3),
                "result": result,
                "llm_last_error": llm.last_error,
                "llm_last_error_detail": getattr(llm, "last_error_detail", None),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
