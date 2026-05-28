from __future__ import annotations

import logging
import re
from datetime import date, datetime
from mimetypes import guess_type
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown import markdown

from client_profiler.embeddings import LocalEmbedder, VectorRetriever
from client_profiler.extraction import OllamaClient
from client_profiler.ingestion import DocumentReader
from client_profiler.projects.project_associator import ProjectAssociator
from client_profiler.projects.project_field_service import ProjectFieldService
from client_profiler.projects.project_summary_service import ProjectSummaryService

from client_profiler.config import ProfilerConfig
from client_profiler.storage import SqliteStorage


PROJECT_TOKEN_STOPWORDS = {
    "client",
    "doc",
    "document",
    "engineering",
    "estimate",
    "for",
    "inspection",
    "order",
    "project",
    "purchase",
    "quote",
    "quotation",
    "report",
    "request",
    "rev",
    "revision",
    "site",
    "summary",
    "the",
    "variation",
    "work",
}


logger = logging.getLogger(__name__)


def create_app(config: ProfilerConfig | None = None) -> FastAPI:
    config = config or ProfilerConfig()
    config.ensure_dirs()
    storage = SqliteStorage(config.db_path)

    app = FastAPI(title="Client Profiler UI", version="0.1.0")

    base_dir = Path(__file__).parent
    templates = Jinja2Templates(directory=str(base_dir / "templates"))

    def _llm_status() -> dict[str, Any]:
        active_model = str(getattr(llm, "model", "") or "") if llm is not None else ""
        configured_model = str(config.llm_model or "")
        return {
            "enabled": llm is not None,
            "provider": str(config.llm_provider or ""),
            "base_url": str(config.ollama_base_url or ""),
            "configured_model": configured_model,
            "active_model": active_model,
            "fallback_active": bool(active_model and configured_model and active_model != configured_model),
        }

    templates.env.globals["llm_status"] = _llm_status

    app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
    reader = DocumentReader()
    llm = (
        OllamaClient(config.ollama_base_url, config.llm_model, timeout=config.ollama_timeout_seconds)
        if config.llm_provider == "ollama"
        else None
    )
    project_associator = ProjectAssociator(storage, llm)
    embedder = LocalEmbedder(config.embedding_model)
    retriever = VectorRetriever(storage)
    summary_service = ProjectSummaryService(
        storage=storage,
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        questionnaire_path=config.project_summary_questionnaire_path,
    )
    field_service = ProjectFieldService(
        storage=storage,
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        key_fields_path=config.project_key_fields_path,
        debug_enabled=config.project_field_debug_enabled,
        debug_log_path=config.project_field_debug_log_path,
    )
    logger.info(
        "Client Profiler web app initialized (db=%s, llm_provider=%s, llm_model=%s, embedding_model=%s)",
        config.db_path,
        config.llm_provider,
        config.llm_model,
        config.embedding_model,
    )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        clients = storage.list_clients()
        cards = []
        for client in clients:
            summary = storage.get_client_summary(client)
            report_count = len(storage.list_client_reports(client, limit=5000))
            cards.append({"name": client, "summary": summary, "report_count": report_count})

        recent_timeline = storage.list_timeline(limit=50)
        recent_timeline = _enrich_timeline_items(recent_timeline, storage)
        context = {
            "request": request,
            "clients": cards,
            "recent_timeline": recent_timeline,
        }
        return templates.TemplateResponse(request=request, name="index.html", context=context)

    @app.get("/client/{client_name}", response_class=HTMLResponse)
    def client_detail(client_name: str, request: Request) -> HTMLResponse:
        clients = set(storage.list_clients())
        if client_name not in clients:
            raise HTTPException(status_code=404, detail="Client not found")

        nodes = storage.list_profile_nodes(client_name)
        timeline = storage.list_timeline(client_name=client_name, limit=300)
        timeline = _enrich_timeline_items(timeline, storage)
        summary = storage.get_client_summary(client_name)
        node_history = storage.list_node_history(client_name)
        tree = _build_tree(nodes, node_history=node_history)
        client_reports = _enrich_reports(storage.list_client_reports(client_name, limit=1000))
        client_documents = _enrich_client_documents(storage.list_client_documents(client_name))
        client_non_reports = [doc for doc in client_documents if doc.get("document_kind") != "report"]
        projects = _build_client_projects(
            client_name,
            client_documents,
            reader,
            project_associator,
            storage,
            field_service,
            llm_available=llm is not None,
        )

        context = {
            "request": request,
            "client_name": client_name,
            "summary": summary,
            "tree": tree,
            "timeline": timeline,
            "nodes": nodes,
            "client_reports": client_reports,
            "client_non_reports": client_non_reports,
            "projects": projects,
            "project_field_definitions": field_service.field_definitions(),
            "llm_available": llm is not None,
        }
        return templates.TemplateResponse(request=request, name="client.html", context=context)

    @app.get("/timeline", response_class=HTMLResponse)
    def timeline_view(
        request: Request,
        client: str | None = None,
        doc_kind: str | None = None,
        contact: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 300,
    ) -> HTMLResponse:
        items = storage.list_timeline(client_name=client, limit=max(1, min(limit, 2000)))
        items = _enrich_timeline_items(items, storage)

        selected_contact = (contact or "").strip()
        if selected_contact:
            items = [
                item
                for item in items
                if selected_contact.lower()
                in " ".join(item.get("client_contacts", [])).lower()
            ]

        selected_doc_kind = (doc_kind or "").strip()
        if selected_doc_kind:
            items = [item for item in items if item.get("document_kind") == selected_doc_kind]

        start_date = _parse_date(date_from)
        end_date = _parse_date(date_to)
        if start_date:
            items = [
                item for item in items if (_parse_date(item.get("event_date") or "") or datetime.min.date()) >= start_date
            ]
        if end_date:
            items = [
                item for item in items if (_parse_date(item.get("event_date") or "") or datetime.max.date()) <= end_date
            ]

        context = {
            "request": request,
            "timeline": items,
            "timeline_visual_items": _build_timeline_visual_items(items),
            "clients": storage.list_clients(),
            "document_kinds": storage.list_document_kinds(),
            "contacts": _all_contacts(storage),
            "filters": {
                "client": client or "",
                "doc_kind": selected_doc_kind,
                "contact": selected_contact,
                "date_from": date_from or "",
                "date_to": date_to or "",
                "limit": limit,
            },
        }
        return templates.TemplateResponse(request=request, name="timeline.html", context=context)

    @app.get("/reports", response_class=HTMLResponse)
    def report_lookup(
        request: Request,
        client: str | None = None,
        doc_kind: str | None = None,
        contact: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 500,
    ) -> HTMLResponse:
        reports = storage.list_reports(
            client_name=client or None,
            doc_kind=doc_kind or None,
            contact=contact or None,
            date_from=date_from or None,
            date_to=date_to or None,
            limit=limit,
        )
        reports = _enrich_reports(reports)
        context = {
            "request": request,
            "reports": reports,
            "clients": storage.list_clients(),
            "document_kinds": storage.list_document_kinds(),
            "contacts": storage.all_report_contacts(),
            "filters": {
                "client": client or "",
                "doc_kind": doc_kind or "",
                "contact": contact or "",
                "date_from": date_from or "",
                "date_to": date_to or "",
                "limit": limit,
            },
        }
        return templates.TemplateResponse(request=request, name="reports.html", context=context)

    @app.get("/report/versions", response_class=HTMLResponse)
    def report_versions(request: Request, path: str) -> HTMLResponse:
        versions = storage.list_report_versions(path)
        context = {
            "request": request,
            "source_path": path,
            "document_name": Path(path).name,
            "document_href": f"/document?path={quote(path, safe='')}",
            "versions": _decorate_report_versions(versions),
        }
        return templates.TemplateResponse(request=request, name="report_versions.html", context=context)

    @app.get("/admin/maintenance", response_class=HTMLResponse)
    def admin_maintenance(request: Request, status: str | None = None, message: str | None = None) -> HTMLResponse:
        clients = storage.list_clients()
        cards = []
        for client in clients:
            summary = storage.get_client_summary(client)
            cards.append(
                {
                    "name": client,
                    "summary": summary,
                    "document_count": len(storage.list_client_documents(client)),
                }
            )

        context = {
            "request": request,
            "clients": cards,
            "suspicious_clients": storage.find_suspicious_single_doc_clients(),
            "recent_reports": _enrich_reports(storage.list_reports(limit=200)),
            "status": status or "",
            "message": message or "",
        }
        return templates.TemplateResponse(request=request, name="maintenance.html", context=context)

    @app.post("/admin/merge-client", response_model=None)
    def admin_merge_client(
        source_client: str = Form(...),
        target_client: str = Form(...),
    ) -> RedirectResponse:
        try:
            result = storage.merge_clients(source_client, target_client)
            message = (
                f"Merged '{source_client}' into '{target_client}'. "
                f"Moved nodes: {result['moved_nodes']}, merged nodes: {result['merged_nodes']}, updated documents: {result['updated_documents']}."
            )
            return RedirectResponse(url=f"/admin/maintenance?status=ok&message={quote(message, safe='')}", status_code=303)
        except Exception as exc:
            return RedirectResponse(
                url=f"/admin/maintenance?status=error&message={quote(str(exc), safe='')}",
                status_code=303,
            )

    @app.post("/admin/delete-client", response_model=None)
    def admin_delete_client(
        client_name: str = Form(...),
        delete_documents: bool = Form(False),
    ) -> RedirectResponse:
        result = storage.delete_client(client_name, delete_documents=delete_documents)
        message = (
            f"Deleted client '{client_name}'. Profiles: {result['profiles']}, timeline: {result['timeline']}, documents_deleted: {result['documents_deleted']}."
        )
        return RedirectResponse(url=f"/admin/maintenance?status=ok&message={quote(message, safe='')}", status_code=303)

    @app.post("/admin/delete-node", response_model=None)
    def admin_delete_node(
        client_name: str = Form(...),
        node_path: str = Form(...),
    ) -> RedirectResponse:
        deleted = storage.delete_profile_node(client_name, node_path)
        message = f"Deleted node '{node_path}' for '{client_name}'. Rows removed: {deleted}."
        return RedirectResponse(url=f"/admin/maintenance?status=ok&message={quote(message, safe='')}", status_code=303)

    @app.post("/admin/delete-report", response_model=None)
    def admin_delete_report(path: str = Form(...)) -> RedirectResponse:
        result = storage.delete_document(path)
        message = (
            f"Deleted report '{Path(path).name}'. documents: {result['documents']}, timeline: {result['timeline']}, vectors: {result['vectors']}."
        )
        return RedirectResponse(url=f"/admin/maintenance?status=ok&message={quote(message, safe='')}", status_code=303)

    @app.post("/admin/cleanup-bad-clients", response_model=None)
    def admin_cleanup_bad_clients(delete_documents: bool = Form(False)) -> RedirectResponse:
        removed = storage.cleanup_suspicious_single_doc_clients(delete_documents=delete_documents)
        message = f"Cleanup complete. Removed suspicious clients: {len(removed)}."
        return RedirectResponse(url=f"/admin/maintenance?status=ok&message={quote(message, safe='')}", status_code=303)

    @app.post("/report/set-date", response_model=None)
    def report_set_date(
        path: str = Form(...),
        report_date: str = Form(...),
        next_url: str = Form("/reports"),
    ) -> RedirectResponse:
        storage.set_report_date(path, report_date)
        return RedirectResponse(url=next_url, status_code=303)

    @app.post("/api/client/{client_name}/project/{project_key}/summarise")
    def project_summarise(client_name: str, project_key: str, request: Request) -> dict:
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM provider is not configured.")

        logger.info("[runtime] project summary generation started (client=%s, project_key=%s)", client_name, project_key)

        all_docs = _enrich_client_documents(storage.list_client_documents(client_name))
        projects = _build_client_projects(
            client_name,
            all_docs,
            reader,
            project_associator,
            storage,
            field_service,
            llm_available=llm is not None,
        )
        project = next((p for p in projects if p["project_key"] == project_key), None)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")

        generated = summary_service.generate_and_store(
            client_name=client_name,
            project_key=project_key,
            project_name=project["project_name"],
            documents=project["documents"],
        )
        if not generated.get("ok"):
            reason = str(generated.get("error_message") or "Unknown error.")
            code = str(generated.get("error_code") or "unknown_error")
            recoverable_codes = {
                "llm_not_configured",
                "request_error",
                "request_timed_out",
            }
            if code.startswith("http_error_") or code in recoverable_codes:
                fallback_summary = project_associator.summarize_project(
                    client_name,
                    project["project_name"],
                    project["documents"],
                    skip_llm=True,
                )
                storage.upsert_project_summary(
                    client_name=client_name,
                    project_key=project_key,
                    project_name=project["project_name"],
                    summary_text=fallback_summary,
                    summary_method=f"rule_fallback_{code}",
                    questionnaire_answers={"fallback_reason": reason},
                )
                summary_service.refresh_summary_embedding(
                    client_name=client_name,
                    project_key=project_key,
                    project_name=project["project_name"],
                    summary_text=fallback_summary,
                    summary_method=f"rule_fallback_{code}",
                    questionnaire_answers={"fallback_reason": reason},
                )
                stored = storage.get_project_summary(client_name, project_key) or {}
                logger.warning(
                    "[runtime] project summary LLM unavailable; returned fallback summary "
                    "(client=%s, project_key=%s, code=%s)",
                    client_name,
                    project_key,
                    code,
                )
                return {
                    "summary": fallback_summary,
                    "updated_at": stored.get("updated_at"),
                    "method": "rule_fallback",
                    "fallback_reason": reason,
                }
            logger.warning(
                "[runtime] project summary generation failed (client=%s, project_key=%s, code=%s, reason=%s)",
                client_name,
                project_key,
                code,
                reason,
            )
            raise HTTPException(
                status_code=500,
                detail=f"AI summary generation failed ({code}): {reason}",
            )

        logger.info("[runtime] project summary generation completed (client=%s, project_key=%s)", client_name, project_key)

        return {
            "summary": generated["summary"],
            "updated_at": generated.get("updated_at"),
            "method": "ai",
        }

    @app.post("/api/client/{client_name}/project/{project_key}/field/{field_key}/generate")
    def project_generate_field(client_name: str, project_key: str, field_key: str) -> dict:
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM provider is not configured.")

        logger.info(
            "[runtime] project field generation started (client=%s, project_key=%s, field=%s)",
            client_name,
            project_key,
            field_key,
        )

        definitions = {row["key"]: row for row in field_service.field_definitions()}
        field_def = definitions.get(field_key)
        if field_def is None:
            raise HTTPException(status_code=404, detail=f"Unknown field: {field_key}")

        all_docs = _enrich_client_documents(storage.list_client_documents(client_name))
        projects = _build_client_projects(
            client_name,
            all_docs,
            reader,
            project_associator,
            storage,
            field_service,
            llm_available=llm is not None,
        )
        project = next((p for p in projects if p["project_key"] == project_key), None)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")

        existing = {
            row["key"]: {
                "value": row.get("value") or "",
                "status": row.get("status") or "",
            }
            for row in project.get("key_fields", [])
        }
        result = field_service.generate_field(
            client_name=client_name,
            project_key=project_key,
            project_name=project["project_name"],
            field_key=field_key,
            field_prompt=field_def["prompt"],
            documents=project["documents"],
            existing_fields=existing,
        )
        if not result.get("ok"):
            reason = str(result.get("error_message") or "Unknown error.")
            code = str(result.get("error_code") or "unknown_error")
            logger.warning(
                "[runtime] project field generation failed (client=%s, project_key=%s, field=%s, code=%s, reason=%s)",
                client_name,
                project_key,
                field_key,
                code,
                reason,
            )
            raise HTTPException(status_code=502, detail=f"AI generation failed ({code}): {reason}")
        logger.info(
            "[runtime] project field generation completed (client=%s, project_key=%s, field=%s, status=%s)",
            client_name,
            project_key,
            field_key,
            result.get("status") or "",
        )
        return {
            "field_key": field_key,
            "value": result.get("value") or "",
            "status": result.get("status") or "",
            "updated_at": result.get("updated_at") or "",
            "method": result.get("method") or "ai",
            "error_message": "",
        }

    @app.get("/document", response_model=None)
    def view_document(path: str, request: Request) -> Any:
        document_path = _resolve_document_path(path)
        if document_path is None or not document_path.exists() or not document_path.is_file():
            raise HTTPException(status_code=404, detail="Document not found")

        if _is_browser_friendly(document_path):
            media_type, _ = guess_type(str(document_path))
            return FileResponse(path=document_path, media_type=media_type)

        try:
            document = reader.read(document_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not render document: {exc}") from exc

        md = (
            f"# {document_path.name}\n\n"
            f"- Source path: `{document_path}`\n"
            f"- Source type: `{document.source_type}`\n\n"
            "## Extracted Content\n\n"
            f"```\n{document.text}\n```\n"
        )
        rendered = markdown(md, extensions=["fenced_code", "tables"])
        context = {
            "request": request,
            "document_title": document_path.name,
            "document_path": str(document_path),
            "rendered_markdown": rendered,
        }
        return templates.TemplateResponse(request=request, name="document_markdown.html", context=context)

    @app.get("/api/clients")
    def api_clients() -> list[dict[str, Any]]:
        return [
            {
                "name": client,
                "summary": storage.get_client_summary(client),
            }
            for client in storage.list_clients()
        ]

    @app.get("/api/client/{client_name}")
    def api_client(client_name: str) -> dict[str, Any]:
        clients = set(storage.list_clients())
        if client_name not in clients:
            raise HTTPException(status_code=404, detail="Client not found")

        nodes = storage.list_profile_nodes(client_name)
        return {
            "client_name": client_name,
            "summary": storage.get_client_summary(client_name),
            "tree": _build_tree(nodes),
            "timeline": storage.list_timeline(client_name=client_name, limit=300),
            "nodes": nodes,
        }

    return app


def _build_tree(nodes: list[dict[str, Any]], node_history: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    root: dict[str, Any] = {"name": "root", "children": {}, "facts": {}}
    node_history = node_history or {}

    for node in nodes:
        path = node.get("node_path", "")
        parts = [p.strip() for p in path.split("/") if p.strip()]
        cursor = root
        for part in parts:
            cursor = cursor["children"].setdefault(part, {"name": part, "children": {}, "facts": {}})
        cursor["facts"] = _decorate_node_facts(node.get("facts", {}))
        cursor["updated_at"] = node.get("updated_at")
        cursor["history"] = _decorate_history_rows(node_history.get(path, []))

    return root


def _resolve_document_path(raw_path: str) -> Path | None:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def _is_browser_friendly(path: Path) -> bool:
    return path.suffix.lower() in {".pdf", ".txt", ".md", ".markdown", ".html", ".htm", ".csv"}


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _all_contacts(storage: SqliteStorage) -> list[str]:
    contacts: set[str] = set()
    for client_name in storage.list_clients():
        for contact in storage.list_client_contacts(client_name):
            contacts.add(contact)
    return sorted(contacts)


def _enrich_timeline_items(items: list[dict[str, Any]], storage: SqliteStorage) -> list[dict[str, Any]]:
    enriched = []
    for item in items:
        source_document = item.get("source_document", "")
        record = storage.get_latest_document_record(source_document)
        metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}

        event = dict(item)
        event["document_kind"] = metadata.get("document_kind") or "unknown"
        event["source_type"] = record.get("source_type") if isinstance(record, dict) else None
        event["document_name"] = Path(source_document).name
        event["document_href"] = f"/document?path={quote(source_document, safe='')}"
        event["client_contacts"] = storage.list_client_contacts(item.get("client_name", ""))
        event["report_date"] = metadata.get("report_date")
        event["project_name"] = metadata.get("project_name")
        event["project_code"] = metadata.get("project_code")
        event["display_date"] = item.get("event_date") or metadata.get("report_date") or item.get("created_at")
        event["missing_date"] = not bool(item.get("event_date") or metadata.get("report_date"))
        enriched.append(event)
    return enriched


def _build_timeline_visual_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(items, key=_timeline_sort_key):
        stamp = _coerce_timeline_datetime(item.get("display_date") or item.get("created_at"))
        timestamp = int(stamp.timestamp() * 1000) if stamp else 0
        rows.append(
            {
                "timestamp": timestamp,
                "date_label": item.get("display_date") or item.get("created_at") or "Undated",
                "summary": item.get("summary") or "Event",
                "client_name": item.get("client_name") or "Unknown client",
                "document_kind": item.get("document_kind") or "unknown",
                "project_name": item.get("project_name") or "",
                "document_name": item.get("document_name") or "Document",
                "document_href": item.get("document_href") or "#",
            }
        )
    return rows


def _timeline_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    stamp = _coerce_timeline_datetime(item.get("display_date") or item.get("created_at"))
    return (int(stamp.timestamp()) if stamp else 0, str(item.get("summary") or ""))


def _coerce_timeline_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _decorate_node_facts(facts: dict[str, Any]) -> dict[str, Any]:
    decorated = dict(facts) if isinstance(facts, dict) else {}
    reports = decorated.get("source_reports", [])
    rows = []
    if isinstance(reports, list):
        for entry in reports:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path", "")).strip()
            if not path:
                continue
            rows.append(
                {
                    "path": path,
                    "date": entry.get("date"),
                    "name": Path(path).name,
                    "href": f"/document?path={quote(path, safe='')}",
                }
            )
    decorated["source_reports"] = rows
    return decorated


def _decorate_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        doc = row.get("superseded_by_document")
        label = f"Superseded by {Path(str(doc)).name}" if doc else "Superseded"
        result.append(
            {
                **row,
                "label": label,
                "superseded_href": f"/document?path={quote(str(doc), safe='')}" if doc else None,
            }
        )
    return result


def _enrich_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for report in reports:
        row = dict(report)
        source_path = str(report.get("source_path", ""))
        row["document_name"] = Path(source_path).name
        row["document_href"] = f"/document?path={quote(source_path, safe='')}"
        row["versions_href"] = f"/report/versions?path={quote(source_path, safe='')}"
        row["missing_date"] = not bool(report.get("report_date"))
        enriched.append(row)
    return enriched


def _enrich_client_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for document in documents:
        row = dict(document)
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        source_path = str(document.get("source_path", ""))
        row["document_href"] = f"/document?path={quote(source_path, safe='')}"
        row["document_kind"] = str(metadata.get("document_kind", "unknown"))
        row["report_date"] = metadata.get("report_date")
        row["ingested_at"] = document.get("ingested_at") or ""
        row["is_report"] = row["document_kind"] == "report"
        enriched.append(row)
    return enriched


def _build_client_projects(
    client_name: str,
    documents: list[dict[str, Any]],
    reader: DocumentReader,
    project_associator: ProjectAssociator,
    storage: SqliteStorage,
    field_service: ProjectFieldService,
    llm_available: bool,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}

    for document in documents:
        doc = _decorate_project_document(document, reader)
        project_key = str(doc.get("project_key") or "").strip()
        if project_key:
            # Respect explicit project keys from ingestion/reconciliation metadata.
            # Fuzzy group matching is only for documents that do not have a key.
            # Otherwise, similarly worded projects for the same client can collapse
            # into a single node (e.g., multiple NorthRiver projects).
            doc["project_key"] = project_key
        else:
            project_key = _match_existing_project_group(doc, groups) or _derive_project_key(doc)
            doc["project_key"] = project_key

        group = groups.get(project_key)
        if group is None:
            group = {
                "project_key": project_key,
                "project_name": doc.get("project_name") or _derive_project_name(doc),
                "project_code": doc.get("project_code") or "",
                "references": [],
                "authors": [],
                "contacts": [],
                "documents": [],
                "reports": [],
                "non_reports": [],
                "latest_date": None,
                "latest_sort": "",
                "summary": "",
                "stored_summary": "",
                "has_ai_summary": False,
                "needs_ai_summary": False,
                "token_hints": set(),
            }
            groups[project_key] = group

        project_name = str(doc.get("project_name") or "").strip()
        if project_name and (not group["project_name"] or group["project_name"].lower().startswith("untitled ")):
            group["project_name"] = project_name
        if not group["project_code"] and doc.get("project_code"):
            group["project_code"] = doc["project_code"]

        group["documents"].append(doc)
        target_bucket = group["reports"] if doc.get("is_report") else group["non_reports"]
        target_bucket.append(doc)

        group["references"] = _merge_unique_text(group["references"], doc.get("related_references", []))
        group["authors"] = _merge_unique_text(group["authors"], doc.get("authors", []))
        group["contacts"] = _merge_unique_text(group["contacts"], doc.get("contacts", []))
        group["token_hints"].update(_document_project_tokens(doc))

        if doc.get("project_summary") and not group["stored_summary"]:
            group["stored_summary"] = str(doc["project_summary"]).strip()

        sort_value = _sortable_project_date(doc)
        if sort_value and (not group["latest_sort"] or sort_value > group["latest_sort"]):
            group["latest_sort"] = sort_value
            group["latest_date"] = doc.get("report_date") or doc.get("ingested_at") or ""

    result: list[dict[str, Any]] = []
    for group in groups.values():
        group["project_name"] = group["project_name"] or "Untitled Project"
        group["reports"] = _sort_project_documents(group["reports"])
        group["non_reports"] = _sort_project_documents(group["non_reports"])
        group["documents"] = _sort_project_documents(group["documents"])
        ai_summary = storage.get_project_summary(client_name, group["project_key"])
        summary_text = str((ai_summary or {}).get("summary_text") or "").strip() if isinstance(ai_summary, dict) else ""
        summary_method = str((ai_summary or {}).get("summary_method") or "").strip() if isinstance(ai_summary, dict) else ""
        has_persisted_summary = bool(summary_text)
        has_ai_summary = bool(has_persisted_summary and summary_method.startswith("ai_"))
        group["has_ai_summary"] = has_ai_summary
        group["summary_method"] = summary_method
        group["summary_source"] = "AI" if has_ai_summary else ("Fallback" if has_persisted_summary else "Rule")
        group["needs_ai_summary"] = bool(llm_available and not has_ai_summary and not has_persisted_summary)
        group["ai_summary_updated_at"] = str(ai_summary.get("updated_at") or "") if ai_summary else ""
        if has_persisted_summary:
            group["summary"] = summary_text
        else:
            group["summary"] = group["stored_summary"] or project_associator.summarize_project(
                client_name,
                group["project_name"],
                group["documents"],
                skip_llm=True,
            )
        group["document_count"] = len(group["documents"])
        group["report_count"] = len(group["reports"])
        group["non_report_count"] = len(group["non_reports"])

        definitions = field_service.field_definitions()
        persisted = field_service.project_field_values(client_name, group["project_key"])
        key_fields: list[dict[str, Any]] = []
        for definition in definitions:
            row = persisted.get(definition["key"], {}) if isinstance(persisted, dict) else {}
            value = str((row if isinstance(row, dict) else {}).get("value") or "").strip()
            status = str((row if isinstance(row, dict) else {}).get("status") or "").strip()
            evidence = str((row if isinstance(row, dict) else {}).get("evidence") or "").strip()
            method = str((row if isinstance(row, dict) else {}).get("method") or "").strip().lower()
            error_message = str((row if isinstance(row, dict) else {}).get("error_message") or "").strip()
            if not method and evidence.lower().startswith("fallback due to llm"):
                method = "fallback"
            if not method and status:
                method = "ai"
            if not error_message and method == "fallback" and evidence:
                fallback_prefix = "Fallback due to LLM unavailability:"
                if evidence.startswith(fallback_prefix):
                    error_message = evidence[len(fallback_prefix) :].strip() or "AI generation unavailable."
                elif evidence.lower().startswith("fallback due to llm"):
                    error_message = evidence
            key_fields.append(
                {
                    "key": definition["key"],
                    "label": definition["label"],
                    "value": value,
                    "status": status,
                    "method": method,
                    "error_message": error_message,
                    "needs_generation": bool(llm_available and not value and status != "absent"),
                }
            )
        group["key_fields"] = key_fields

        group.pop("token_hints", None)
        result.append(group)

    return sorted(
        result,
        key=lambda group: (group.get("latest_sort") or "", group.get("project_name") or ""),
        reverse=True,
    )


def _decorate_project_document(document: dict[str, Any], reader: DocumentReader) -> dict[str, Any]:
    row = dict(document)
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    source_path = str(row.get("source_path") or "").strip()
    path = Path(source_path) if source_path else None
    document_kind = str(row.get("document_kind") or metadata.get("document_kind") or "unknown")
    title = str(metadata.get("title") or row.get("document_name") or row.get("document_stem") or "Untitled document").strip()
    excerpt = _read_document_excerpt(reader, path)

    return {
        **row,
        "document_name": row.get("document_name") or (path.name if path else "Unknown document"),
        "document_stem": row.get("document_stem") or (path.stem if path else ""),
        "document_href": row.get("document_href") or f"/document?path={quote(source_path, safe='')}",
        "versions_href": f"/report/versions?path={quote(source_path, safe='')}",
        "document_kind": document_kind,
        "title": title,
        "client_name": str(metadata.get("client_name") or "").strip(),
        "project_key": str(metadata.get("project_key") or "").strip(),
        "project_name": str(metadata.get("project_name") or "").strip(),
        "project_code": str(metadata.get("project_code") or "").strip(),
        "project_summary": str(metadata.get("project_summary") or "").strip(),
        "report_type": str(metadata.get("report_type") or "").strip(),
        "key_findings": _normalize_text_list(metadata.get("key_findings", [])),
        "recommendations": _normalize_text_list(metadata.get("recommendations", [])),
        "authors": _normalize_text_list(metadata.get("authors", [])),
        "contacts": _normalize_text_list(metadata.get("contacts", [])),
        "related_references": _collect_document_references(metadata),
        "excerpt": excerpt,
        "currency_amounts": _extract_currency_amounts(excerpt),
        "is_report": document_kind == "report",
    }


def _match_existing_project_group(document: dict[str, Any], groups: dict[str, dict[str, Any]]) -> str | None:
    project_code = str(document.get("project_code") or "").upper()
    project_name = _normalize_project_name(document.get("project_name") or _derive_project_name(document))
    references = {value.upper() for value in document.get("related_references", [])}
    contacts = {value.lower() for value in document.get("contacts", [])}
    authors = {value.lower() for value in document.get("authors", [])}
    tokens = _project_tokens(document.get("title") or "") | _project_tokens(document.get("document_stem") or "")
    doc_year = _extract_year(document.get("report_date") or document.get("ingested_at") or "")

    best_key: str | None = None
    best_score = 0
    for key, group in groups.items():
        score = 0
        if project_code and project_code == str(group.get("project_code") or "").upper():
            score += 6
        if project_name and project_name == _normalize_project_name(group.get("project_name") or ""):
            score += 5
        group_refs = {value.upper() for value in group.get("references", [])}
        score += len(references & group_refs) * 3
        group_contacts = {value.lower() for value in group.get("contacts", [])}
        if contacts & group_contacts:
            score += 2
        group_authors = {value.lower() for value in group.get("authors", [])}
        if authors & group_authors:
            score += 1
        group_tokens = set(group.get("token_hints", set()))
        shared_tokens = tokens & group_tokens
        if len(shared_tokens) >= 2:
            score += 2
        group_year = _extract_year(group.get("latest_sort") or group.get("latest_date") or "")
        if doc_year and group_year and doc_year == group_year:
            score += 1
        if score > best_score:
            best_score = score
            best_key = key

    return best_key if best_score >= 4 else None


def _derive_project_key(document: dict[str, Any]) -> str:
    candidate = (
        document.get("project_code")
        or document.get("project_name")
        or _derive_project_name(document)
        or document.get("document_stem")
        or document.get("document_name")
        or "untitled-project"
    )
    return _slugify(str(candidate))


def _derive_project_name(document: dict[str, Any]) -> str:
    project_name = str(document.get("project_name") or "").strip()
    if project_name:
        return project_name

    report_type = str(document.get("report_type") or "").strip()
    if report_type:
        return report_type

    title = str(document.get("title") or "").strip()
    if _normalize_project_name(title) in {"technical consulting report", "report", "final report"}:
        title = str(document.get("document_stem") or document.get("document_name") or "").strip()
    if not title:
        title = str(document.get("document_stem") or document.get("document_name") or "").strip()

    title = re.sub(r"[_\-]+", " ", title)
    title = re.sub(r"\bv\d+\b", " ", title, flags=re.IGNORECASE)

    client_name = str((document.get("metadata") or {}).get("client_name") or "").strip()
    if client_name:
        title = re.sub(re.escape(client_name), " ", title, flags=re.IGNORECASE)
    title = re.sub(r"\b(q|po|rev|report|invoice|quote|purchase order|access request|email chain)\b", " ", title, flags=re.IGNORECASE)
    title = re.sub(r"\b\d{2,4}\b", " ", title)
    title = re.sub(r"\s+", " ", title).strip(" -")
    return title.title() or "Untitled Project"


def _sort_project_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        documents,
        key=lambda document: (_sortable_project_date(document), document.get("document_name") or ""),
        reverse=True,
    )


def _sortable_project_date(document: dict[str, Any]) -> str:
    value = str(document.get("report_date") or document.get("ingested_at") or "").strip()
    parsed = _parse_date(value)
    if parsed:
        return parsed.isoformat()
    parsed_dt = _parse_datetime(value)
    if parsed_dt:
        return parsed_dt.date().isoformat()
    return value


def _parse_datetime(text: str) -> datetime | None:
    text = str(text or "").strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize_text_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _collect_document_references(metadata: dict[str, Any]) -> list[str]:
    values = []
    for key in ["project_code", "quote_number", "purchase_order_number", "access_reference"]:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    values.extend(_normalize_text_list(metadata.get("related_references", [])))
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if len(value) < 3:
            continue
        if value.lower() in {"for", "to", "reference", "uplift"}:
            continue
        key = value.upper()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


def _merge_unique_text(existing: list[str], new_values: list[str]) -> list[str]:
    result = list(existing)
    seen = {value.lower() for value in existing}
    for value in new_values:
        text = str(value).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        result.append(text)
    return result


def _normalize_project_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _project_tokens(value: Any) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]{3,}", str(value or "").lower()))
    return {token for token in tokens if token not in PROJECT_TOKEN_STOPWORDS}

def _document_project_tokens(document: dict[str, Any]) -> set[str]:
    client_tokens = _project_tokens(document.get("client_name") or "")
    tokens = set()
    for value in [document.get("project_name"), document.get("title"), document.get("document_stem")]:
        tokens.update(_project_tokens(value or ""))
    return {token for token in tokens if token not in client_tokens}

def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug or "untitled-project"


def _extract_year(value: Any) -> str:
    match = re.search(r"\b(20\d{2}|19\d{2})\b", str(value or ""))
    return match.group(1) if match else ""


def _read_document_excerpt(reader: DocumentReader, path: Path | None, limit: int = 1400) -> str:
    if path is None or not path.exists():
        return ""
    try:
        document = reader.read(path)
    except Exception:
        return ""
    text = re.sub(r"\s+", " ", document.text).strip()
    if not text:
        return ""

    # Keep an opening slice for identity/context, then pull targeted windows
    # around downstream sections where findings/recommendations often appear.
    opening = text[:limit]
    windows: list[str] = []
    seen_starts: set[int] = set()
    for pattern in [
        r"\bfindings?\b",
        r"\brecommendations?\b",
        r"\bactions?\b",
        r"\bconclusions?\b",
        r"\bissues?\b",
        r"\brisks?\b",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        start = max(0, match.start() - 120)
        if start in seen_starts:
            continue
        seen_starts.add(start)
        end = min(len(text), start + 1100)
        windows.append(text[start:end].strip())

    if not windows:
        return opening

    merged_parts = [opening, *windows]
    merged = "\n".join(part for part in merged_parts if part)
    return merged[: max(limit * 3, limit)]


def _extract_currency_amounts(text: str) -> list[str]:
    found = re.findall(r"(?:\$|AUD\s*)\d[\d,]*(?:\.\d{2})?", text, flags=re.IGNORECASE)
    result: list[str] = []
    seen: set[str] = set()
    for value in found:
        cleaned = value.strip()
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result[:5]


def _decorate_report_versions(versions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    total = len(versions)
    for idx, version in enumerate(versions, start=1):
        row = dict(version)
        row["label"] = f"Version {total - idx + 1}"
        row["document_href"] = f"/document?path={quote(str(version.get('source_path', '')), safe='')}"
        rows.append(row)
    return rows


app = create_app()
