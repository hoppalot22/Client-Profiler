from __future__ import annotations

import json
import time
from typing import Any

import requests

from client_profiler.config import ProfilerConfig
from client_profiler.embeddings import LocalEmbedder, VectorRetriever
from client_profiler.extraction import OllamaClient
from client_profiler.ingestion import DocumentReader
from client_profiler.projects.project_associator import ProjectAssociator
from client_profiler.projects.project_field_service import ProjectFieldService
from client_profiler.storage import SqliteStorage
from client_profiler.web.app import _build_client_projects, _enrich_client_documents
from scripts.benchmark_llm_models import _load_cases

MODELS = ["qwen2.5:0.5b", "qwen3:0.6b"]


def _build_case_prompt() -> tuple[ProfilerConfig, Any, str, dict[str, Any], dict[str, Any]]:
    cfg = ProfilerConfig()
    cases = _load_cases(cfg.db_path)
    case = next(
        c for c in cases if c.client_name == "Acme Refining" and c.project_key == "ar-inv-2026" and c.field_key == "title"
    )

    storage = SqliteStorage(cfg.db_path)
    seed_llm = OllamaClient(cfg.ollama_base_url, cfg.llm_model, timeout=60)
    embedder = LocalEmbedder(cfg.embedding_model)
    retriever = VectorRetriever(storage)
    field_service = ProjectFieldService(storage, embedder, retriever, seed_llm, cfg.project_key_fields_path, debug_enabled=False)
    reader = DocumentReader()
    project_associator = ProjectAssociator(storage, seed_llm)

    documents = _enrich_client_documents(storage.list_client_documents(case.client_name))
    projects = _build_client_projects(case.client_name, documents, reader, project_associator, storage, field_service, llm_available=True)
    project = next(p for p in projects if str(p.get("project_key") or "") == case.project_key)
    definition = {row["key"]: row for row in field_service.field_definitions()}[case.field_key]
    existing = {
        row["key"]: {
            "value": row.get("value") or "",
            "status": row.get("status") or "",
        }
        for row in project.get("key_fields", [])
    }

    profile = field_service._field_profile(case.field_key)
    max_words_raw = profile.get("max_words")
    max_words = int(max_words_raw) if isinstance(max_words_raw, (int, float)) else None
    local_evidence = field_service._build_local_field_evidence(case.field_key, definition["prompt"], project["documents"])
    project_hints = field_service._project_hints(case.project_key, project["documents"])
    rag_hits: list[dict[str, Any]] = []
    if len(local_evidence) < 2:
        query_text = field_service._build_query_text(case.project_name, case.field_key, definition["prompt"], project["documents"])
        query_embedding = embedder.embed_text(query_text)
        rag_hits = field_service._search_field_rag(case.field_key, query_embedding, case.client_name, project_hints)
    rag_context = field_service._build_rag_context(case.field_key, rag_hits)
    local_evidence, rag_context = field_service._fit_context_budget(case.field_key, local_evidence, rag_context)

    context_fields = {
        key: str((value or {}).get("value") or "").strip()
        for key, value in existing.items()
        if str((value or {}).get("value") or "").strip()
    }
    compact_fields = field_service._compact_known_fields(context_fields, case.field_key)
    max_words_instruction = (
        f"If present, value must be <= {max_words} words. "
        if max_words is not None
        else "If present, return a concise value with complete meaning. "
    )
    prompt = (
        'Return STRICT JSON only: {"is_present":true|false,"value":"...","evidence":"..."}. '
        f"{max_words_instruction}"
        "Use only explicit project evidence below. If missing/uncertain set is_present=false and value to an empty string. "
        "Do not guess.\n"
        f"Field-specific rules: {field_service._field_specific_guardrails(case.field_key)}\n\n"
        f"Project: {case.project_name}\n"
        f"Field: {case.field_key}\n"
        f"Instruction: {definition['prompt']}\n"
        f"Known fields: {compact_fields}\n"
        f"Project evidence:\n{field_service._format_lines(local_evidence)}\n"
        f"Related evidence:\n{field_service._format_lines(rag_context)}"
    )
    meta = {
        "prompt_chars": len(prompt),
        "local_evidence_count": len(local_evidence),
        "rag_hit_count": len(rag_hits),
        "prompt_preview": prompt[:1200],
    }
    return cfg, case, prompt, meta, definition


def _probe_raw(cfg: ProfilerConfig, model: str, prompt: str, use_json_mode: bool) -> None:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "top_p": 1, "seed": 42},
    }
    if use_json_mode:
        payload["format"] = "json"

    started = time.perf_counter()
    try:
        response = requests.post(
            cfg.ollama_base_url.rstrip("/") + "/api/generate",
            json=payload,
            timeout=180,
        )
        elapsed = time.perf_counter() - started
        print(f"HTTP {response.status_code} in {elapsed:.2f}s")
        text = response.text
        print(f"RAW_LEN {len(text)}")
        print(text[:3000])
    except Exception as exc:  # pragma: no cover - diagnostic script
        elapsed = time.perf_counter() - started
        print(f"ERROR after {elapsed:.2f}s: {type(exc).__name__}: {exc}")


def _probe_client(cfg: ProfilerConfig, model: str, prompt: str) -> None:
    client = OllamaClient(cfg.ollama_base_url, model, timeout=180)
    started = time.perf_counter()
    result = client.extract_structured(prompt)
    elapsed = time.perf_counter() - started
    print(f"CLIENT_RESULT {json.dumps(result, ensure_ascii=True)}")
    print(f"CLIENT_ERROR {client.last_error}")
    print(f"CLIENT_ERROR_DETAIL {client.last_error_detail}")
    print(f"CLIENT_LATENCY {elapsed:.2f}s")


def main() -> None:
    cfg, case, prompt, meta, definition = _build_case_prompt()
    print("CASE", json.dumps(case.__dict__, indent=2))
    print("FIELD_DEFINITION", json.dumps(definition, indent=2))
    print("PROMPT_META", json.dumps(meta, indent=2))

    for model in MODELS:
        print("\n" + "=" * 80)
        print(f"MODEL {model}")
        print("-" * 80)
        _probe_client(cfg, model, prompt)
        for use_json_mode in (True, False):
            print(f"RAW_PROBE json_mode={use_json_mode}")
            _probe_raw(cfg, model, prompt, use_json_mode)


if __name__ == "__main__":
    main()
