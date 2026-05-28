from __future__ import annotations

import argparse
import json
from pathlib import Path

from client_profiler import ClientProfiler, ProfilerConfig
from client_profiler.embeddings import VectorRetriever


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Client profiler for mixed document collections")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_file = sub.add_parser("ingest-file", help="Ingest one file")
    ingest_file.add_argument("path", type=Path)
    ingest_file.add_argument("--force", action="store_true", help="Force re-ingest even if content hash matches")

    ingest_dir = sub.add_parser("ingest-dir", help="Ingest a directory")
    ingest_dir.add_argument("path", type=Path)
    ingest_dir.add_argument("--non-recursive", action="store_true")
    ingest_dir.add_argument("--force", action="store_true", help="Force re-ingest all files")

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

    merge_client = sub.add_parser("merge-client", help="Merge one client into another")
    merge_client.add_argument("source", type=str, help="Source client to merge from")
    merge_client.add_argument("target", type=str, help="Target client to merge into")

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
        merge_client,
    ]:
        p.add_argument("--db", type=Path, default=Path("./data/profiler.db"))
        p.add_argument("--model", type=str, default="llama3.1")
        p.add_argument("--ollama-url", type=str, default="http://localhost:11434")

    return parser


def _build_profiler(args: argparse.Namespace) -> ClientProfiler:
    config = ProfilerConfig(
        db_path=args.db,
        data_dir=args.db.parent,
        llm_model=args.model,
        ollama_base_url=args.ollama_url,
    )
    return ClientProfiler(config)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    profiler = _build_profiler(args)

    if args.command == "ingest-file":
        result = profiler.ingest_file(args.path, force_reingest=args.force)
        print(json.dumps(result, indent=2))
        return

    if args.command == "ingest-dir":
        results = profiler.ingest_directory(args.path, recursive=not args.non_recursive, force_reingest=args.force)
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

    if args.command == "merge-client":
        result = profiler.storage.merge_clients(args.source, args.target)
        print(json.dumps({"source": args.source, "target": args.target, **result}, indent=2))
        return


if __name__ == "__main__":
    main()
