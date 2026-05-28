from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from client_profiler import ClientProfiler, ProfilerConfig
from client_profiler.embeddings import VectorRetriever


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Client profiler for mixed document collections")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_file = sub.add_parser("ingest-file", help="Ingest one file")
    ingest_file.add_argument("path", type=Path)
    ingest_file.add_argument("--force", action="store_true", help="Force re-ingest even if content hash matches")
    ingest_file.add_argument(
        "--enable-ingest-llm",
        action="store_true",
        help="Enable LLM calls during ingest (extraction/project matching). Disabled by default for faster ingest.",
    )
    ingest_file.add_argument(
        "--generate-project-summaries",
        action="store_true",
        help="Generate project summaries during ingest (requires --enable-ingest-llm).",
    )
    ingest_file.add_argument(
        "--status",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live ingest status readouts and progress bars in the terminal.",
    )
    ingest_file.add_argument(
        "--reconcile-projects",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run post-ingest project reconciliation for touched clients.",
    )
    ingest_file.add_argument(
        "--auto-merge-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run high-confidence duplicate-client merge cleanup after ingest.",
    )
    ingest_file.add_argument(
        "--merge-confidence",
        type=float,
        default=0.95,
        help="Minimum confidence threshold for automatic client merge cleanup.",
    )

    ingest_dir = sub.add_parser("ingest-dir", help="Ingest a directory")
    ingest_dir.add_argument("path", type=Path)
    ingest_dir.add_argument("--non-recursive", action="store_true")
    ingest_dir.add_argument("--force", action="store_true", help="Force re-ingest all files")
    ingest_dir.add_argument(
        "--enable-ingest-llm",
        action="store_true",
        help="Enable LLM calls during ingest (extraction/project matching). Disabled by default for faster ingest.",
    )
    ingest_dir.add_argument(
        "--generate-project-summaries",
        action="store_true",
        help="Generate project summaries during ingest (requires --enable-ingest-llm).",
    )
    ingest_dir.add_argument(
        "--status",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live ingest status readouts and progress bars in the terminal.",
    )
    ingest_dir.add_argument(
        "--reconcile-projects",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run post-ingest project reconciliation for touched clients.",
    )
    ingest_dir.add_argument(
        "--auto-merge-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run high-confidence duplicate-client merge cleanup after ingest.",
    )
    ingest_dir.add_argument(
        "--merge-confidence",
        type=float,
        default=0.95,
        help="Minimum confidence threshold for automatic client merge cleanup.",
    )

    query = sub.add_parser("query", help="Semantic search over embedded chunks")
    query.add_argument("text", type=str)
    query.add_argument("--client", type=str, default=None)
    query.add_argument("--top-k", type=int, default=5)

    clients = sub.add_parser("list-clients", help="List known clients")

    set_report_date = sub.add_parser("set-report-date", help="Set report date for a source document")
    set_report_date.add_argument("path", type=str)
    set_report_date.add_argument("date", type=str, help="Date in YYYY-MM-DD")

    delete_client = sub.add_parser("delete-client", help="Delete a client and associated profile/timeline components")
    delete_client.add_argument("name", type=str)
    delete_client.add_argument("--delete-documents", action="store_true", help="Also delete underlying document rows and vectors")

    delete_node = sub.add_parser("delete-node", help="Delete a profile node for a client")
    delete_node.add_argument("client", type=str)
    delete_node.add_argument("node_path", type=str)

    delete_report = sub.add_parser("delete-report", help="Delete all records for a source document path")
    delete_report.add_argument("path", type=str)

    suspicious_clients = sub.add_parser("list-suspicious-clients", help="List suspicious clients that match their only document name")

    cleanup_bad_clients = sub.add_parser("cleanup-bad-clients", help="Remove suspicious one-document clients")
    cleanup_bad_clients.add_argument("--delete-documents", action="store_true", help="Also delete underlying document records")

    cleanup_merge_clients = sub.add_parser(
        "cleanup-merge-clients",
        help="Detect and merge clients only when confidence is very high",
    )
    cleanup_merge_clients.add_argument(
        "--min-confidence",
        type=float,
        default=0.95,
        help="Minimum confidence threshold in [0,1] for auto-merge candidates.",
    )
    cleanup_merge_clients.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview merge candidates without applying merges.",
    )

    reset_db = sub.add_parser("reset-db", help="Reset the profiler database")
    reset_db.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Backup existing DB before reset.",
    )

    merge_client = sub.add_parser("merge-client", help="Merge one client into another")
    merge_client.add_argument("source", type=str, help="Source client to merge from")
    merge_client.add_argument("target", type=str, help="Target client to merge into")

    reconcile_projects = sub.add_parser(
        "reconcile-projects",
        help="Re-cluster and reconcile project assignments for one client or all clients",
    )
    reconcile_projects.add_argument("--client", type=str, default=None, help="Reconcile only one client name")
    reconcile_projects.add_argument(
        "--apply",
        action="store_true",
        help="Apply project assignment changes to storage (default is dry-run preview)",
    )
    reconcile_projects.add_argument(
        "--allow-llm-match",
        action="store_true",
        help="Reserved flag for future LLM-assisted arbitration in reconciliation",
    )

    for p in [
        ingest_file,
        ingest_dir,
        query,
        clients,
        set_report_date,
        delete_client,
        delete_node,
        delete_report,
        suspicious_clients,
        cleanup_bad_clients,
        cleanup_merge_clients,
        merge_client,
        reconcile_projects,
        reset_db,
    ]:
        p.add_argument("--db", type=Path, default=Path("./data/profiler.db"))
        p.add_argument("--model", type=str, default="llama3.1")
        p.add_argument("--ollama-url", type=str, default="http://localhost:11434")

    return parser


def _build_profiler(args: argparse.Namespace) -> ClientProfiler:
    ingest_llm_enabled = bool(getattr(args, "enable_ingest_llm", False))
    ingest_generate_summaries = bool(getattr(args, "generate_project_summaries", False))
    config = ProfilerConfig(
        db_path=args.db,
        data_dir=args.db.parent,
        llm_model=args.model,
        ollama_base_url=args.ollama_url,
        ingest_use_llm_extraction=ingest_llm_enabled,
        ingest_use_llm_project_matching=ingest_llm_enabled,
        ingest_generate_project_summaries=ingest_llm_enabled and ingest_generate_summaries,
        ingest_reconcile_projects=bool(getattr(args, "reconcile_projects", True)),
        ingest_auto_merge_cleanup=bool(getattr(args, "auto_merge_cleanup", True)),
        merge_cleanup_min_confidence=float(getattr(args, "merge_confidence", 0.95)),
    )
    return ClientProfiler(config)


class _CliStatusReporter:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._files_bar: tqdm | None = None
        self._chunks_bar: tqdm | None = None

    def _write(self, message: str) -> None:
        if not self.enabled:
            return
        if self._files_bar is not None:
            self._files_bar.write(message)
            return
        print(message, file=sys.stderr)

    def _close_chunk_bar(self) -> None:
        if self._chunks_bar is not None:
            self._chunks_bar.close()
            self._chunks_bar = None

    def close(self) -> None:
        self._close_chunk_bar()
        if self._files_bar is not None:
            self._files_bar.close()
            self._files_bar = None

    def __call__(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return

        kind = str(event.get("event") or "")
        path = str(event.get("path") or "")
        file_name = Path(path).name if path else ""

        if kind == "directory_scanned":
            total = int(event.get("total_files") or 0)
            self._files_bar = tqdm(
                total=total,
                desc="Ingesting files",
                unit="file",
                leave=True,
                dynamic_ncols=True,
                file=sys.stderr,
            )
            self._write(f"[status] scanned {total} files in {event.get('directory')}")
            return

        if kind == "file_started":
            self._write(f"[status] processing {file_name}")
            return

        if kind == "file_skipped_duplicate":
            if self._files_bar is not None:
                self._files_bar.update(1)
            self._write(f"[status] skipped duplicate {file_name}")
            return

        if kind == "chunk_embedding_started":
            total_chunks = int(event.get("total_chunks") or 0)
            self._close_chunk_bar()
            if total_chunks > 1:
                self._chunks_bar = tqdm(
                    total=total_chunks,
                    desc=f"Embedding {file_name[:32]}",
                    unit="chunk",
                    leave=False,
                    dynamic_ncols=True,
                    file=sys.stderr,
                )
            return

        if kind == "chunk_embedded":
            if self._chunks_bar is not None:
                self._chunks_bar.update(1)
            return

        if kind == "chunk_embedding_completed":
            self._close_chunk_bar()
            return

        if kind == "summary_generation_started":
            self._write(f"[status] generating project summary for {event.get('project_name') or file_name}")
            return

        if kind == "project_reconciliation_started":
            subject = event.get("client_name") or file_name
            self._write(f"[status] reconciling project groups for {subject}")
            return

        if kind == "project_reconciliation_completed":
            subject = event.get("client_name") or file_name
            self._write(f"[status] completed project reconciliation for {subject}")
            return

        if kind == "file_completed":
            if self._files_bar is not None:
                self._files_bar.update(1)
            status = event.get("status") or "done"
            self._write(f"[status] completed {file_name} ({status})")
            return

        if kind == "file_error":
            if self._files_bar is not None:
                self._files_bar.update(1)
            self._write(f"[status] error {file_name}: {event.get('error')}")
            return

        if kind == "directory_completed":
            self._close_chunk_bar()
            if self._files_bar is not None:
                self._files_bar.close()
                self._files_bar = None
            self._write(
                "[status] directory ingest complete "
                f"(total={event.get('total_files')}, succeeded={event.get('succeeded')}, failed={event.get('failed')})"
            )
            return


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    profiler = _build_profiler(args)
    status_reporter = _CliStatusReporter(enabled=bool(getattr(args, "status", False)))

    if args.command == "ingest-file":
        try:
            result = profiler.ingest_file(
                args.path,
                force_reingest=args.force,
                status_callback=status_reporter,
            )
        finally:
            status_reporter.close()
        print(json.dumps(result, indent=2))
        return

    if args.command == "ingest-dir":
        try:
            results = profiler.ingest_directory(
                args.path,
                recursive=not args.non_recursive,
                force_reingest=args.force,
                status_callback=status_reporter,
            )
        finally:
            status_reporter.close()
        print(json.dumps(results, indent=2))
        return

    if args.command == "query":
        query_embedding = profiler.embedder.embed_text(args.text)
        retriever = VectorRetriever(profiler.storage)
        hits = retriever.search(
            query_embedding,
            top_k=args.top_k,
            client_name=args.client,
            query_text=args.text,
        )
        print(json.dumps(hits, indent=2))
        return

    if args.command == "list-clients":
        print(json.dumps(profiler.storage.list_clients(), indent=2))
        return

    if args.command == "set-report-date":
        ok = profiler.storage.set_report_date(args.path, args.date)
        print(json.dumps({"updated": ok, "path": args.path, "date": args.date}, indent=2))
        return

    if args.command == "delete-client":
        result = profiler.storage.delete_client(args.name, delete_documents=args.delete_documents)
        print(json.dumps({"client": args.name, **result}, indent=2))
        return

    if args.command == "delete-node":
        deleted = profiler.storage.delete_profile_node(args.client, args.node_path)
        print(json.dumps({"client": args.client, "node_path": args.node_path, "deleted": deleted}, indent=2))
        return

    if args.command == "delete-report":
        result = profiler.storage.delete_document(args.path)
        print(json.dumps({"path": args.path, **result}, indent=2))
        return

    if args.command == "list-suspicious-clients":
        results = profiler.storage.find_suspicious_single_doc_clients()
        print(json.dumps(results, indent=2))
        return

    if args.command == "cleanup-bad-clients":
        results = profiler.storage.cleanup_suspicious_single_doc_clients(delete_documents=args.delete_documents)
        print(json.dumps(results, indent=2))
        return

    if args.command == "cleanup-merge-clients":
        result = profiler.cleanup_high_confidence_client_merges(
            min_confidence=args.min_confidence,
            dry_run=bool(args.dry_run),
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "reset-db":
        result = profiler.reset_db(backup=bool(args.backup))
        print(json.dumps(result, indent=2))
        return

    if args.command == "merge-client":
        result = profiler.storage.merge_clients(args.source, args.target)
        print(json.dumps({"source": args.source, "target": args.target, **result}, indent=2))
        return

    if args.command == "reconcile-projects":
        if args.client:
            clients = [str(args.client).strip()]
        else:
            clients = sorted(
                {
                    str((record.get("metadata") or {}).get("client_name") or "").strip()
                    for record in profiler.storage.list_document_records()
                    if isinstance(record.get("metadata"), dict)
                }
                - {""}
            )

        results = []
        for client in clients:
            results.append(
                profiler.project_associator.reconcile_client_projects(
                    client,
                    apply_changes=bool(args.apply),
                    allow_llm_match=bool(args.allow_llm_match),
                )
            )

        print(
            json.dumps(
                {
                    "mode": "apply" if args.apply else "dry_run",
                    "clients": clients,
                    "results": results,
                },
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()
