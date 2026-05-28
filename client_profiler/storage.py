from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import ProfileNode, TimelineItem


class SqliteStorage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_name TEXT NOT NULL,
                    node_path TEXT NOT NULL,
                    facts_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(client_name, node_path)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS timeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_name TEXT NOT NULL,
                    event_date TEXT,
                    summary TEXT NOT NULL,
                    source_document TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_name TEXT,
                    source_document TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_node_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_name TEXT NOT NULL,
                    node_path TEXT NOT NULL,
                    facts_json TEXT NOT NULL,
                    superseded_by_document TEXT NOT NULL,
                    superseded_at TEXT NOT NULL,
                    archived_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS report_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    report_date TEXT,
                    ingested_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    change_summary_json TEXT NOT NULL,
                    UNIQUE(source_path, content_hash)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_name TEXT NOT NULL,
                    project_key TEXT NOT NULL,
                    project_name TEXT,
                    summary_text TEXT NOT NULL,
                    summary_method TEXT NOT NULL,
                    questionnaire_answers_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(client_name, project_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_summary_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_name TEXT NOT NULL,
                    project_key TEXT NOT NULL,
                    project_name TEXT,
                    error_code TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_key_fields (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_name TEXT NOT NULL,
                    project_key TEXT NOT NULL,
                    project_name TEXT,
                    fields_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(client_name, project_key)
                )
                """
            )

    def _path_candidates(self, source_path: str) -> tuple[str, str]:
        normalized = str(source_path).strip()
        return normalized, normalized.replace("\\", "/").replace("/", "\\")

    def _name_key(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = Path(text).stem
        return "".join(ch for ch in text if ch.isalnum())

    def _merge_unique_text_list(self, left: Any, right: Any) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in [*(left if isinstance(left, list) else []), *(right if isinstance(right, list) else [])]:
            text = str(value).strip()
            if text and text not in seen:
                result.append(text)
                seen.add(text)
        return result

    def _merge_source_reports(self, left: Any, right: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in [left, right]:
            for entry in group if isinstance(group, list) else []:
                if not isinstance(entry, dict):
                    continue
                path = str(entry.get("path") or "").strip()
                if not path:
                    continue
                if path in seen:
                    continue
                rows.append({"path": path, "date": entry.get("date")})
                seen.add(path)
        return rows

    def _merge_facts(self, source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
        merged = dict(target if isinstance(target, dict) else {})
        source = source if isinstance(source, dict) else {}

        for key in ["document_kind", "report_type", "report_date", "project_name", "project_code", "project_key", "project_summary"]:
            if not merged.get(key) and source.get(key):
                merged[key] = source.get(key)

        for key in ["authors", "contacts", "key_findings", "recommendations", "document_kinds", "related_references"]:
            merged[key] = self._merge_unique_text_list(source.get(key), merged.get(key))

        merged["source_reports"] = self._merge_source_reports(source.get("source_reports"), merged.get("source_reports"))

        source_additional = source.get("additional_fields", {}) if isinstance(source.get("additional_fields", {}), dict) else {}
        target_additional = merged.get("additional_fields", {}) if isinstance(merged.get("additional_fields", {}), dict) else {}
        merged["additional_fields"] = {**source_additional, **target_additional}
        return merged

    def save_document_record(
        self,
        source_path: str,
        source_type: str,
        content_hash: str,
        metadata: dict[str, Any],
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO documents (source_path, source_type, content_hash, ingested_at, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source_path,
                    source_type,
                    content_hash,
                    datetime.utcnow().isoformat(),
                    json.dumps(metadata, ensure_ascii=True),
                ),
            )
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("Failed to persist document row (missing lastrowid).")
        return int(row_id)

    def document_already_ingested(self, source_path: str, content_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM documents
                WHERE source_path = ? AND content_hash = ?
                LIMIT 1
                """,
                (source_path, content_hash),
            ).fetchone()
        return row is not None

    def upsert_profile_node(self, client_name: str, node: ProfileNode) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO profiles (client_name, node_path, facts_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(client_name, node_path)
                DO UPDATE SET
                    facts_json=excluded.facts_json,
                    updated_at=excluded.updated_at
                """,
                (
                    client_name,
                    node.path,
                    json.dumps(node.facts, ensure_ascii=True),
                    node.updated_at.isoformat(),
                ),
            )

    def add_timeline_item(self, item: TimelineItem) -> None:
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT 1 FROM timeline
                WHERE client_name = ?
                  AND COALESCE(event_date, '') = COALESCE(?, '')
                  AND summary = ?
                  AND source_document = ?
                LIMIT 1
                """,
                (item.client_name, item.date, item.summary, item.source_document),
            ).fetchone()
            if existing is not None:
                return

            conn.execute(
                """
                INSERT INTO timeline (client_name, event_date, summary, source_document, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    item.client_name,
                    item.date,
                    item.summary,
                    item.source_document,
                    item.created_at.isoformat(),
                ),
            )

    def add_vector(
        self,
        source_document: str,
        chunk_text: str,
        embedding: list[float],
        metadata: dict[str, Any],
        client_name: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO vectors (client_name, source_document, chunk_text, embedding_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    client_name,
                    source_document,
                    chunk_text,
                    json.dumps(embedding),
                    json.dumps(metadata, ensure_ascii=True),
                    datetime.utcnow().isoformat(),
                ),
            )

    def upsert_vector(
        self,
        source_document: str,
        chunk_text: str,
        embedding: list[float],
        metadata: dict[str, Any],
        client_name: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM vectors
                WHERE source_document = ?
                  AND COALESCE(client_name, '') = COALESCE(?, '')
                """,
                (source_document, client_name),
            )
            conn.execute(
                """
                INSERT INTO vectors (client_name, source_document, chunk_text, embedding_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    client_name,
                    source_document,
                    chunk_text,
                    json.dumps(embedding),
                    json.dumps(metadata, ensure_ascii=True),
                    datetime.utcnow().isoformat(),
                ),
            )

    def fetch_vectors(
        self,
        client_name: str | None = None,
        source_documents: list[str] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT client_name, source_document, chunk_text, embedding_json, metadata_json FROM vectors"
        where_clauses: list[str] = []
        params: list[Any] = []

        if client_name:
            where_clauses.append("client_name = ?")
            params.append(client_name)

        if source_documents:
            normalized = [str(p).strip() for p in source_documents if str(p).strip()]
            if normalized:
                placeholders = ",".join(["?"] * len(normalized))
                where_clauses.append(f"source_document IN ({placeholders})")
                params.extend(normalized)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY source_document, chunk_text, rowid"

        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()

        result = []
        for row in rows:
            metadata = json.loads(row[4])
            if metadata_filters and not self._metadata_matches(metadata, metadata_filters):
                continue
            result.append(
                {
                    "client_name": row[0],
                    "source_document": row[1],
                    "chunk_text": row[2],
                    "embedding": json.loads(row[3]),
                    "metadata": metadata,
                }
            )
        return result

    def _metadata_matches(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, wanted in (filters or {}).items():
            actual = metadata.get(key)
            if isinstance(wanted, (list, tuple, set)):
                wanted_values = {str(v).strip().lower() for v in wanted if str(v).strip()}
                if not wanted_values:
                    continue
                if isinstance(actual, list):
                    actual_values = {str(v).strip().lower() for v in actual if str(v).strip()}
                    if not (actual_values & wanted_values):
                        return False
                else:
                    if str(actual or "").strip().lower() not in wanted_values:
                        return False
                continue

            if str(actual or "").strip().lower() != str(wanted or "").strip().lower():
                return False

        return True

    def list_clients(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT client_name FROM profiles ORDER BY client_name").fetchall()
        return [r[0] for r in rows if r[0]]

    def list_client_documents(self, client_name: str) -> list[dict[str, Any]]:
        records = self.list_document_records()
        seen_paths: set[str] = set()
        docs: list[dict[str, Any]] = []
        for record in records:
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            if metadata.get("client_name") != client_name:
                continue
            source_path = str(record.get("source_path") or "").strip()
            if not source_path or source_path in seen_paths:
                continue
            seen_paths.add(source_path)
            docs.append(
                {
                    "source_path": source_path,
                    "document_name": Path(source_path).name,
                    "document_stem": Path(source_path).stem,
                    "metadata": metadata,
                }
            )
        return docs

    def find_clients_by_references(self, references: list[str]) -> list[str]:
        wanted = {str(value).strip().upper() for value in references if str(value).strip()}
        if not wanted:
            return []

        found: set[str] = set()
        for record in self.list_document_records():
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            client_name = metadata.get("client_name")
            if not isinstance(client_name, str) or not client_name.strip():
                continue

            values = set()
            for key in [
                "project_code",
                "quote_number",
                "purchase_order_number",
                "access_reference",
            ]:
                raw = metadata.get(key)
                if isinstance(raw, str) and raw.strip():
                    values.add(raw.strip().upper())
            for raw in metadata.get("related_references", []) or []:
                if isinstance(raw, str) and raw.strip():
                    values.add(raw.strip().upper())

            if wanted & values:
                found.add(client_name.strip())

        return sorted(found)

    def list_profile_nodes(self, client_name: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT node_path, facts_json, updated_at
                FROM profiles
                WHERE client_name = ?
                ORDER BY node_path
                """,
                (client_name,),
            ).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "node_path": row[0],
                    "facts": json.loads(row[1]),
                    "updated_at": row[2],
                }
            )
        return result

    def list_timeline(self, client_name: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        query = (
            "SELECT client_name, event_date, summary, source_document, created_at "
            "FROM timeline"
        )
        params: tuple[Any, ...] = ()
        if client_name:
            query += " WHERE client_name = ?"
            params = (client_name,)
        query += " ORDER BY COALESCE(event_date, created_at) DESC LIMIT ?"
        params = params + (limit,)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "client_name": row[0],
                "event_date": row[1],
                "summary": row[2],
                "source_document": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def get_client_summary(self, client_name: str) -> dict[str, int]:
        with self._connect() as conn:
            profile_nodes = conn.execute(
                "SELECT COUNT(*) FROM profiles WHERE client_name = ?",
                (client_name,),
            ).fetchone()[0]
            timeline_events = conn.execute(
                "SELECT COUNT(*) FROM timeline WHERE client_name = ?",
                (client_name,),
            ).fetchone()[0]
            vector_chunks = conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE client_name = ?",
                (client_name,),
            ).fetchone()[0]

        return {
            "profile_nodes": int(profile_nodes),
            "timeline_events": int(timeline_events),
            "vector_chunks": int(vector_chunks),
        }

    def list_document_records(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_path, source_type, content_hash, ingested_at, metadata_json
                FROM documents
                ORDER BY ingested_at DESC
                """
            ).fetchall()

        return [
            {
                "source_path": row[0],
                "source_type": row[1],
                "content_hash": row[2],
                "ingested_at": row[3],
                "metadata": json.loads(row[4]),
            }
            for row in rows
        ]

    def list_project_documents(self, client_name: str, project_key: str | None = None) -> list[dict[str, Any]]:
        records = self.list_document_records()
        result: list[dict[str, Any]] = []
        for record in records:
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            if metadata.get("client_name") != client_name:
                continue
            if project_key and metadata.get("project_key") != project_key:
                continue
            result.append(
                {
                    "source_path": record.get("source_path"),
                    "source_type": record.get("source_type"),
                    "ingested_at": record.get("ingested_at"),
                    "document_kind": metadata.get("document_kind"),
                    "report_date": metadata.get("report_date"),
                    "title": metadata.get("title"),
                    "project_key": metadata.get("project_key"),
                    "project_name": metadata.get("project_name"),
                    "project_code": metadata.get("project_code"),
                    "related_references": metadata.get("related_references", []),
                }
            )
        return result

    def get_project_summary(self, client_name: str, project_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT project_name, summary_text, summary_method, questionnaire_answers_json, created_at, updated_at
                FROM project_summaries
                WHERE client_name = ? AND project_key = ?
                LIMIT 1
                """,
                (client_name, project_key),
            ).fetchone()

        if row is None:
            return None

        answers_raw = row[3]
        try:
            answers = json.loads(answers_raw) if answers_raw else {}
        except json.JSONDecodeError:
            answers = {}

        return {
            "project_name": row[0],
            "summary_text": row[1],
            "summary_method": row[2],
            "questionnaire_answers": answers,
            "created_at": row[4],
            "updated_at": row[5],
        }

    def upsert_project_summary(
        self,
        client_name: str,
        project_key: str,
        project_name: str,
        summary_text: str,
        summary_method: str,
        questionnaire_answers: dict[str, str] | None = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO project_summaries (
                    client_name, project_key, project_name, summary_text, summary_method,
                    questionnaire_answers_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_name, project_key)
                DO UPDATE SET
                    project_name = excluded.project_name,
                    summary_text = excluded.summary_text,
                    summary_method = excluded.summary_method,
                    questionnaire_answers_json = excluded.questionnaire_answers_json,
                    updated_at = excluded.updated_at
                """,
                (
                    client_name,
                    project_key,
                    project_name,
                    summary_text,
                    summary_method,
                    json.dumps(questionnaire_answers or {}, ensure_ascii=True),
                    now,
                    now,
                ),
            )

    def add_project_summary_failure(
        self,
        client_name: str,
        project_key: str,
        project_name: str,
        error_code: str,
        error_message: str,
        payload: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO project_summary_failures (
                    client_name, project_key, project_name, error_code, error_message, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_name,
                    project_key,
                    project_name,
                    error_code,
                    error_message,
                    json.dumps(payload, ensure_ascii=True),
                    datetime.utcnow().isoformat(),
                ),
            )

    def list_project_summary_failures(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT client_name, project_key, project_name, error_code, error_message, payload_json, created_at
                FROM project_summary_failures
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row[5]) if row[5] else {}
            except json.JSONDecodeError:
                payload = {}
            result.append(
                {
                    "client_name": row[0],
                    "project_key": row[1],
                    "project_name": row[2],
                    "error_code": row[3],
                    "error_message": row[4],
                    "payload": payload,
                    "created_at": row[6],
                }
            )
        return result

    def get_project_key_fields(self, client_name: str, project_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT project_name, fields_json, created_at, updated_at
                FROM project_key_fields
                WHERE client_name = ? AND project_key = ?
                LIMIT 1
                """,
                (client_name, project_key),
            ).fetchone()

        if row is None:
            return None
        try:
            fields = json.loads(row[1]) if row[1] else {}
        except json.JSONDecodeError:
            fields = {}
        return {
            "project_name": row[0],
            "fields": fields if isinstance(fields, dict) else {},
            "created_at": row[2],
            "updated_at": row[3],
        }

    def upsert_project_key_field(
        self,
        client_name: str,
        project_key: str,
        project_name: str,
        field_key: str,
        value: str,
        status: str,
        evidence: str = "",
        method: str = "ai",
    ) -> dict[str, Any]:
        existing = self.get_project_key_fields(client_name, project_key) or {"fields": {}}
        fields = dict(existing.get("fields") or {})
        timestamp = datetime.utcnow().isoformat()
        fields[str(field_key)] = {
            "value": str(value or "").strip(),
            "status": str(status or "").strip(),
            "evidence": str(evidence or "").strip(),
            "method": str(method or "").strip() or "unknown",
            "updated_at": timestamp,
        }

        created_at = str(existing.get("created_at") or timestamp)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO project_key_fields (
                    client_name, project_key, project_name, fields_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_name, project_key)
                DO UPDATE SET
                    project_name = excluded.project_name,
                    fields_json = excluded.fields_json,
                    updated_at = excluded.updated_at
                """,
                (
                    client_name,
                    project_key,
                    project_name,
                    json.dumps(fields, ensure_ascii=True),
                    created_at,
                    timestamp,
                ),
            )

        return {
            "field_key": field_key,
            "value": fields[str(field_key)]["value"],
            "status": fields[str(field_key)]["status"],
            "evidence": fields[str(field_key)]["evidence"],
            "method": fields[str(field_key)].get("method") or "unknown",
            "updated_at": timestamp,
        }

    def reassign_document_projects(self, client_name: str, changes: list[dict[str, Any]]) -> dict[str, int]:
        assignment_by_path: dict[str, dict[str, Any]] = {}
        for row in changes:
            source_path = str(row.get("source_path") or "").strip()
            if not source_path:
                continue
            assignment_by_path[source_path] = {
                "project_key": str(row.get("new_project_key") or "").strip(),
                "project_name": str(row.get("new_project_name") or "").strip(),
                "project_code": str(row.get("new_project_code") or "").strip(),
                "related_references": [str(value).strip() for value in (row.get("new_related_references") or []) if str(value).strip()],
                "old_project_key": str(row.get("old_project_key") or "").strip(),
            }

        if not assignment_by_path:
            return {
                "updated_documents": 0,
                "updated_vectors": 0,
                "updated_versions": 0,
                "cleared_project_summaries": 0,
                "cleared_project_key_fields": 0,
            }

        updated_documents = 0
        updated_vectors = 0
        updated_versions = 0
        cleared_project_summaries = 0
        cleared_project_key_fields = 0
        touched_project_keys: set[str] = set()

        with self._connect() as conn:
            document_rows = conn.execute(
                """
                SELECT id, source_path, metadata_json
                FROM documents
                ORDER BY ingested_at DESC
                """
            ).fetchall()
            for doc_id, source_path, metadata_json in document_rows:
                assignment = assignment_by_path.get(str(source_path or "").strip())
                if assignment is None:
                    continue
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}
                if metadata.get("client_name") != client_name:
                    continue

                metadata["project_key"] = assignment["project_key"] or None
                metadata["project_name"] = assignment["project_name"] or None
                metadata["project_code"] = assignment["project_code"] or None
                metadata["related_references"] = assignment["related_references"]
                conn.execute(
                    """
                    UPDATE documents
                    SET metadata_json = ?
                    WHERE id = ?
                    """,
                    (json.dumps(metadata, ensure_ascii=True), doc_id),
                )
                updated_documents += 1
                if assignment["project_key"]:
                    touched_project_keys.add(assignment["project_key"])
                if assignment["old_project_key"]:
                    touched_project_keys.add(assignment["old_project_key"])

            vector_rows = conn.execute(
                """
                SELECT id, source_document, metadata_json
                FROM vectors
                """
            ).fetchall()
            for vector_id, source_document, metadata_json in vector_rows:
                assignment = assignment_by_path.get(str(source_document or "").strip())
                if assignment is None:
                    continue
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata["project_key"] = assignment["project_key"] or None
                metadata["project_name"] = assignment["project_name"] or None
                metadata["project_code"] = assignment["project_code"] or None
                metadata["related_references"] = assignment["related_references"]
                conn.execute(
                    """
                    UPDATE vectors
                    SET metadata_json = ?
                    WHERE id = ?
                    """,
                    (json.dumps(metadata, ensure_ascii=True), vector_id),
                )
                updated_vectors += 1

            version_rows = conn.execute(
                """
                SELECT id, source_path, metadata_json, snapshot_json
                FROM report_versions
                ORDER BY ingested_at DESC
                """
            ).fetchall()
            for row_id, source_path, metadata_json, snapshot_json in version_rows:
                assignment = assignment_by_path.get(str(source_path or "").strip())
                if assignment is None:
                    continue
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}
                try:
                    snapshot = json.loads(snapshot_json)
                except json.JSONDecodeError:
                    snapshot = {}
                if not isinstance(metadata, dict):
                    metadata = {}
                if not isinstance(snapshot, dict):
                    snapshot = {}

                metadata["project_key"] = assignment["project_key"] or None
                metadata["project_name"] = assignment["project_name"] or None
                metadata["project_code"] = assignment["project_code"] or None
                snapshot["project_key"] = assignment["project_key"] or None
                snapshot["project_name"] = assignment["project_name"] or None
                snapshot["project_code"] = assignment["project_code"] or None

                conn.execute(
                    """
                    UPDATE report_versions
                    SET metadata_json = ?, snapshot_json = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(metadata, ensure_ascii=True),
                        json.dumps(snapshot, ensure_ascii=True),
                        row_id,
                    ),
                )
                updated_versions += 1

            for project_key in sorted(touched_project_keys):
                if not project_key:
                    continue
                cleared_project_summaries += conn.execute(
                    """
                    DELETE FROM project_summaries
                    WHERE client_name = ? AND project_key = ?
                    """,
                    (client_name, project_key),
                ).rowcount
                cleared_project_key_fields += conn.execute(
                    """
                    DELETE FROM project_key_fields
                    WHERE client_name = ? AND project_key = ?
                    """,
                    (client_name, project_key),
                ).rowcount

        return {
            "updated_documents": int(updated_documents),
            "updated_vectors": int(updated_vectors),
            "updated_versions": int(updated_versions),
            "cleared_project_summaries": int(cleared_project_summaries),
            "cleared_project_key_fields": int(cleared_project_key_fields),
        }

    def get_latest_document_record(self, source_path: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT source_path, source_type, content_hash, ingested_at, metadata_json
                FROM documents
                WHERE source_path = ?
                ORDER BY ingested_at DESC
                LIMIT 1
                """,
                (source_path,),
            ).fetchone()

        if row is None:
            return None
        return {
            "source_path": row[0],
            "source_type": row[1],
            "content_hash": row[2],
            "ingested_at": row[3],
            "metadata": json.loads(row[4]),
        }

    def set_report_date(self, source_path: str, report_date: str) -> bool:
        record = self.get_latest_document_record(source_path)
        if record is None:
            return False

        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["report_date"] = report_date

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET metadata_json = ?
                WHERE source_path = ? AND content_hash = ?
                """,
                (json.dumps(metadata, ensure_ascii=True), source_path, record["content_hash"]),
            )
            conn.execute(
                """
                UPDATE timeline
                SET event_date = COALESCE(event_date, ?)
                WHERE source_document = ?
                """,
                (report_date, source_path),
            )

        self._apply_manual_date_to_profile_nodes(source_path, report_date)
        return True

    def delete_profile_node(self, client_name: str, node_path: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM profiles
                WHERE client_name = ? AND node_path = ?
                """,
                (client_name, node_path),
            )
            conn.execute(
                """
                DELETE FROM profile_node_history
                WHERE client_name = ? AND node_path = ?
                """,
                (client_name, node_path),
            )
        return int(cursor.rowcount)

    def delete_document(self, source_path: str) -> dict[str, int]:
        path_a, path_b = self._path_candidates(source_path)
        with self._connect() as conn:
            deleted_documents = conn.execute(
                """
                DELETE FROM documents
                WHERE source_path = ? OR source_path = ?
                """,
                (path_a, path_b),
            ).rowcount
            deleted_timeline = conn.execute(
                """
                DELETE FROM timeline
                WHERE source_document = ? OR source_document = ?
                """,
                (path_a, path_b),
            ).rowcount
            deleted_vectors = conn.execute(
                """
                DELETE FROM vectors
                WHERE source_document = ? OR source_document = ?
                """,
                (path_a, path_b),
            ).rowcount
            deleted_versions = conn.execute(
                """
                DELETE FROM report_versions
                WHERE source_path = ? OR source_path = ?
                """,
                (path_a, path_b),
            ).rowcount

            rows = conn.execute(
                """
                SELECT client_name, node_path, facts_json
                FROM profiles
                """
            ).fetchall()
            for client_name, node_path, facts_json in rows:
                try:
                    facts = json.loads(facts_json)
                except json.JSONDecodeError:
                    continue
                source_reports = facts.get("source_reports", [])
                if not isinstance(source_reports, list):
                    continue
                kept = []
                changed = False
                for entry in source_reports:
                    if not isinstance(entry, dict):
                        continue
                    entry_path = str(entry.get("path") or "").strip()
                    if entry_path in {path_a, path_b}:
                        changed = True
                        continue
                    kept.append(entry)
                if changed:
                    facts["source_reports"] = kept
                    conn.execute(
                        """
                        UPDATE profiles
                        SET facts_json = ?, updated_at = ?
                        WHERE client_name = ? AND node_path = ?
                        """,
                        (json.dumps(facts, ensure_ascii=True), datetime.utcnow().isoformat(), client_name, node_path),
                    )

        return {
            "documents": int(deleted_documents),
            "timeline": int(deleted_timeline),
            "vectors": int(deleted_vectors),
            "report_versions": int(deleted_versions),
        }

    def delete_client(self, client_name: str, delete_documents: bool = False) -> dict[str, int]:
        docs = self.list_client_documents(client_name)
        deleted_docs = 0
        if delete_documents:
            for doc in docs:
                result = self.delete_document(doc["source_path"])
                deleted_docs += result["documents"]

        with self._connect() as conn:
            deleted_profiles = conn.execute(
                """
                DELETE FROM profiles
                WHERE client_name = ?
                """,
                (client_name,),
            ).rowcount
            deleted_history = conn.execute(
                """
                DELETE FROM profile_node_history
                WHERE client_name = ?
                """,
                (client_name,),
            ).rowcount
            deleted_timeline = conn.execute(
                """
                DELETE FROM timeline
                WHERE client_name = ?
                """,
                (client_name,),
            ).rowcount
            cleared_vectors = conn.execute(
                """
                UPDATE vectors
                SET client_name = NULL
                WHERE client_name = ?
                """,
                (client_name,),
            ).rowcount

            if not delete_documents:
                rows = conn.execute(
                    """
                    SELECT id, metadata_json
                    FROM documents
                    ORDER BY ingested_at DESC
                    """
                ).fetchall()
                for row_id, metadata_json in rows:
                    try:
                        metadata = json.loads(metadata_json)
                    except json.JSONDecodeError:
                        continue
                    if metadata.get("client_name") != client_name:
                        continue
                    metadata["client_name"] = None
                    for key in ["project_key", "project_name", "project_code"]:
                        metadata[key] = None
                    metadata["related_references"] = []
                    conn.execute(
                        """
                        UPDATE documents
                        SET metadata_json = ?
                        WHERE id = ?
                        """,
                        (json.dumps(metadata, ensure_ascii=True), row_id),
                    )

        return {
            "profiles": int(deleted_profiles),
            "history": int(deleted_history),
            "timeline": int(deleted_timeline),
            "vectors_cleared": int(cleared_vectors),
            "documents_deleted": int(deleted_docs),
        }

    def merge_clients(self, source_client: str, target_client: str) -> dict[str, int]:
        source_name = str(source_client or "").strip()
        target_name = str(target_client or "").strip()
        if not source_name or not target_name:
            raise ValueError("Both source_client and target_client are required")
        if source_name == target_name:
            raise ValueError("source_client and target_client must be different")

        merged_nodes = 0
        moved_nodes = 0
        moved_history = 0
        moved_timeline = 0
        moved_vectors = 0
        updated_documents = 0

        with self._connect() as conn:
            source_rows = conn.execute(
                """
                SELECT node_path, facts_json
                FROM profiles
                WHERE client_name = ?
                """,
                (source_name,),
            ).fetchall()

            for node_path, source_facts_json in source_rows:
                target_row = conn.execute(
                    """
                    SELECT facts_json
                    FROM profiles
                    WHERE client_name = ? AND node_path = ?
                    LIMIT 1
                    """,
                    (target_name, node_path),
                ).fetchone()
                source_facts = json.loads(source_facts_json)
                if target_row is None:
                    conn.execute(
                        """
                        UPDATE profiles
                        SET client_name = ?, updated_at = ?
                        WHERE client_name = ? AND node_path = ?
                        """,
                        (target_name, datetime.utcnow().isoformat(), source_name, node_path),
                    )
                    moved_nodes += 1
                    continue

                target_facts = json.loads(target_row[0])
                merged_facts = self._merge_facts(source_facts, target_facts)
                conn.execute(
                    """
                    UPDATE profiles
                    SET facts_json = ?, updated_at = ?
                    WHERE client_name = ? AND node_path = ?
                    """,
                    (json.dumps(merged_facts, ensure_ascii=True), datetime.utcnow().isoformat(), target_name, node_path),
                )
                conn.execute(
                    """
                    DELETE FROM profiles
                    WHERE client_name = ? AND node_path = ?
                    """,
                    (source_name, node_path),
                )
                merged_nodes += 1

            moved_history += conn.execute(
                """
                UPDATE profile_node_history
                SET client_name = ?
                WHERE client_name = ?
                """,
                (target_name, source_name),
            ).rowcount

            moved_timeline += conn.execute(
                """
                UPDATE timeline
                SET client_name = ?
                WHERE client_name = ?
                """,
                (target_name, source_name),
            ).rowcount

            moved_vectors += conn.execute(
                """
                UPDATE vectors
                SET client_name = ?
                WHERE client_name = ?
                """,
                (target_name, source_name),
            ).rowcount

            document_rows = conn.execute(
                """
                SELECT id, metadata_json
                FROM documents
                ORDER BY ingested_at DESC
                """
            ).fetchall()
            for doc_id, metadata_json in document_rows:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    continue
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("client_name") != source_name:
                    continue
                metadata["client_name"] = target_name
                conn.execute(
                    """
                    UPDATE documents
                    SET metadata_json = ?
                    WHERE id = ?
                    """,
                    (json.dumps(metadata, ensure_ascii=True), doc_id),
                )
                updated_documents += 1

            version_rows = conn.execute(
                """
                SELECT id, metadata_json, snapshot_json
                FROM report_versions
                ORDER BY ingested_at DESC
                """
            ).fetchall()
            for row_id, metadata_json, snapshot_json in version_rows:
                changed = False
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = None
                try:
                    snapshot = json.loads(snapshot_json)
                except json.JSONDecodeError:
                    snapshot = None

                if isinstance(metadata, dict) and metadata.get("client_name") == source_name:
                    metadata["client_name"] = target_name
                    changed = True
                if isinstance(snapshot, dict) and snapshot.get("client_name") == source_name:
                    snapshot["client_name"] = target_name
                    changed = True

                if changed:
                    conn.execute(
                        """
                        UPDATE report_versions
                        SET metadata_json = ?, snapshot_json = ?
                        WHERE id = ?
                        """,
                        (
                            json.dumps(metadata if isinstance(metadata, dict) else {}, ensure_ascii=True),
                            json.dumps(snapshot if isinstance(snapshot, dict) else {}, ensure_ascii=True),
                            row_id,
                        ),
                    )

        return {
            "moved_nodes": int(moved_nodes),
            "merged_nodes": int(merged_nodes),
            "moved_history": int(moved_history),
            "moved_timeline": int(moved_timeline),
            "moved_vectors": int(moved_vectors),
            "updated_documents": int(updated_documents),
        }

    def find_suspicious_single_doc_clients(self) -> list[dict[str, str]]:
        findings: list[dict[str, str]] = []
        for client_name in self.list_clients():
            docs = self.list_client_documents(client_name)
            if len(docs) != 1:
                continue
            doc = docs[0]
            client_key = self._name_key(client_name)
            doc_key = self._name_key(doc.get("document_stem", ""))
            if client_key and doc_key and client_key == doc_key:
                findings.append(
                    {
                        "client_name": client_name,
                        "source_path": doc["source_path"],
                        "document_name": doc["document_name"],
                    }
                )
        return findings

    def cleanup_suspicious_single_doc_clients(self, delete_documents: bool = False) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for finding in self.find_suspicious_single_doc_clients():
            outcome = self.delete_client(finding["client_name"], delete_documents=delete_documents)
            results.append({**finding, **outcome})
        return results

    def find_high_confidence_merge_candidates(self, min_confidence: float = 0.95) -> list[dict[str, Any]]:
        threshold = max(0.0, min(1.0, float(min_confidence)))
        records = self.list_document_records()

        signals: dict[str, dict[str, Any]] = {}
        for record in records:
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            client_name = str(metadata.get("client_name") or "").strip()
            if not client_name:
                continue

            bucket = signals.setdefault(
                client_name,
                {
                    "doc_count": 0,
                    "project_codes": set(),
                    "project_keys": set(),
                    "references": set(),
                    "source_stems": set(),
                },
            )

            bucket["doc_count"] += 1

            project_code = str(metadata.get("project_code") or "").strip().lower()
            if project_code:
                bucket["project_codes"].add(project_code)

            project_key = str(metadata.get("project_key") or "").strip().lower()
            if project_key:
                bucket["project_keys"].add(project_key)

            for key in ["quote_number", "purchase_order_number", "access_reference"]:
                value = str(metadata.get(key) or "").strip().lower()
                if value:
                    bucket["references"].add(value)

            for value in metadata.get("related_references") or []:
                text = str(value).strip().lower()
                if text:
                    bucket["references"].add(text)

            source_path = str(record.get("source_path") or "").strip()
            if source_path:
                bucket["source_stems"].add(self._name_key(Path(source_path).stem))

        groups: dict[str, list[str]] = {}
        for client_name in self.list_clients():
            key = self._name_key(client_name)
            if key:
                groups.setdefault(key, []).append(client_name)

        candidates: list[dict[str, Any]] = []
        for _, names in groups.items():
            if len(names) < 2:
                continue

            ordered = sorted(
                names,
                key=lambda name: (
                    -int((signals.get(name) or {}).get("doc_count") or 0),
                    name.lower(),
                ),
            )
            target = ordered[0]
            target_signals = signals.get(target, {})
            target_docs = int(target_signals.get("doc_count") or 0)

            for source in ordered[1:]:
                source_signals = signals.get(source, {})
                source_docs = int(source_signals.get("doc_count") or 0)
                if source_docs <= 0 or target_docs <= 0:
                    continue

                score = 0.90
                reasons = ["name_key_match"]

                if source.lower().strip() == target.lower().strip():
                    score += 0.03
                    reasons.append("case_variant")

                shared_codes = set(source_signals.get("project_codes") or set()) & set(target_signals.get("project_codes") or set())
                if shared_codes:
                    score += 0.02
                    reasons.append("shared_project_code")

                shared_project_keys = set(source_signals.get("project_keys") or set()) & set(target_signals.get("project_keys") or set())
                if shared_project_keys:
                    score += 0.02
                    reasons.append("shared_project_key")

                shared_refs = set(source_signals.get("references") or set()) & set(target_signals.get("references") or set())
                if shared_refs:
                    score += 0.03
                    reasons.append("shared_references")

                shared_stems = set(source_signals.get("source_stems") or set()) & set(target_signals.get("source_stems") or set())
                if shared_stems:
                    score += 0.03
                    reasons.append("shared_document_stem")

                if min(source_docs, target_docs) >= 3:
                    score += 0.01
                    reasons.append("sufficient_documents")

                confidence = min(1.0, score)
                if confidence < threshold:
                    continue

                candidates.append(
                    {
                        "source_client": source,
                        "target_client": target,
                        "confidence": confidence,
                        "reasons": reasons,
                        "source_doc_count": source_docs,
                        "target_doc_count": target_docs,
                    }
                )

        candidates.sort(
            key=lambda row: (
                -float(row.get("confidence") or 0.0),
                str(row.get("target_client") or "").lower(),
                str(row.get("source_client") or "").lower(),
            )
        )
        return candidates

    def cleanup_high_confidence_client_merges(self, min_confidence: float = 0.95, dry_run: bool = False) -> dict[str, Any]:
        candidates = self.find_high_confidence_merge_candidates(min_confidence=min_confidence)
        if dry_run:
            return {
                "dry_run": True,
                "min_confidence": float(min_confidence),
                "candidate_count": len(candidates),
                "candidates": candidates,
                "merged": [],
            }

        merged: list[dict[str, Any]] = []
        for candidate in candidates:
            source_client = str(candidate.get("source_client") or "").strip()
            target_client = str(candidate.get("target_client") or "").strip()
            if not source_client or not target_client or source_client == target_client:
                continue

            result = self.merge_clients(source_client, target_client)
            merged.append({**candidate, **result})

        return {
            "dry_run": False,
            "min_confidence": float(min_confidence),
            "candidate_count": len(candidates),
            "merged_count": len(merged),
            "candidates": candidates,
            "merged": merged,
        }

    def _apply_manual_date_to_profile_nodes(self, source_path: str, report_date: str) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT client_name, node_path, facts_json
                FROM profiles
                """
            ).fetchall()

            for client_name, node_path, facts_json in rows:
                try:
                    facts = json.loads(facts_json)
                except json.JSONDecodeError:
                    continue
                source_reports = facts.get("source_reports", [])
                if not isinstance(source_reports, list):
                    continue
                changed = False
                for entry in source_reports:
                    if isinstance(entry, dict) and entry.get("path") == source_path:
                        entry["date"] = report_date
                        changed = True
                if changed:
                    facts["source_reports"] = source_reports
                    if not facts.get("report_date"):
                        facts["report_date"] = report_date
                    conn.execute(
                        """
                        UPDATE profiles
                        SET facts_json = ?, updated_at = ?
                        WHERE client_name = ? AND node_path = ?
                        """,
                        (
                            json.dumps(facts, ensure_ascii=True),
                            datetime.utcnow().isoformat(),
                            client_name,
                            node_path,
                        ),
                    )

    def list_document_kinds(self) -> list[str]:
        records = self.list_document_records()
        kinds = set()
        for record in records:
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict):
                kind = metadata.get("document_kind")
                if isinstance(kind, str) and kind.strip():
                    kinds.add(kind.strip())
        return sorted(kinds)

    def list_reports(
        self,
        client_name: str | None = None,
        doc_kind: str | None = None,
        contact: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        records = self.list_document_records()
        reports: list[dict[str, Any]] = []

        for record in records:
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            report = {
                "source_path": record.get("source_path"),
                "source_type": record.get("source_type"),
                "ingested_at": record.get("ingested_at"),
                "client_name": metadata.get("client_name"),
                "document_kind": metadata.get("document_kind", "unknown"),
                "report_date": metadata.get("report_date"),
                "report_type": metadata.get("report_type"),
                "authors": metadata.get("authors", []),
                "contacts": metadata.get("contacts", []),
                "title": metadata.get("title"),
            }

            if report["document_kind"] != "report":
                continue

            if client_name and report["client_name"] != client_name:
                continue
            if doc_kind and report["document_kind"] != doc_kind:
                continue
            if contact:
                contacts = report.get("contacts", [])
                if not isinstance(contacts, list):
                    contacts = []
                joined = " ".join(str(c) for c in contacts)
                if contact.lower() not in joined.lower():
                    continue
            if date_from and report.get("report_date") and str(report["report_date"]) < date_from:
                continue
            if date_to and report.get("report_date") and str(report["report_date"]) > date_to:
                continue

            reports.append(report)

        return reports[: max(1, min(limit, 2000))]

    def list_client_reports(self, client_name: str, limit: int = 500) -> list[dict[str, Any]]:
        return self.list_reports(client_name=client_name, limit=limit)

    def all_report_contacts(self) -> list[str]:
        contacts: set[str] = set()
        for report in self.list_reports(limit=20000):
            for contact in report.get("contacts", []) or []:
                if isinstance(contact, str) and contact.strip():
                    contacts.add(contact.strip())
        return sorted(contacts)

    def list_client_contacts(self, client_name: str) -> list[str]:
        rows = self.list_profile_nodes(client_name)
        contacts: set[str] = set()
        for row in rows:
            facts = row.get("facts", {})
            if not isinstance(facts, dict):
                continue
            raw_contacts = facts.get("contacts", [])
            if isinstance(raw_contacts, list):
                for contact in raw_contacts:
                    if isinstance(contact, str) and contact.strip():
                        contacts.add(contact.strip())
        return sorted(contacts)

    def get_profile_node(self, client_name: str, node_path: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT facts_json, updated_at
                FROM profiles
                WHERE client_name = ? AND node_path = ?
                LIMIT 1
                """,
                (client_name, node_path),
            ).fetchone()
        if row is None:
            return None
        return {
            "facts": json.loads(row[0]),
            "updated_at": row[1],
        }

    def archive_profile_node_version(
        self,
        client_name: str,
        node_path: str,
        facts: dict[str, Any],
        superseded_by_document: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO profile_node_history (
                    client_name, node_path, facts_json, superseded_by_document, superseded_at, archived_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    client_name,
                    node_path,
                    json.dumps(facts, ensure_ascii=True),
                    superseded_by_document,
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )

    def list_node_history(self, client_name: str) -> dict[str, list[dict[str, Any]]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT node_path, facts_json, superseded_by_document, superseded_at, archived_at
                FROM profile_node_history
                WHERE client_name = ?
                ORDER BY node_path, archived_at DESC
                """,
                (client_name,),
            ).fetchall()

        result: dict[str, list[dict[str, Any]]] = {}
        for node_path, facts_json, superseded_by_document, superseded_at, archived_at in rows:
            result.setdefault(node_path, []).append(
                {
                    "facts": json.loads(facts_json),
                    "superseded_by_document": superseded_by_document,
                    "superseded_at": superseded_at,
                    "archived_at": archived_at,
                }
            )
        return result

    def add_report_version(
        self,
        source_path: str,
        content_hash: str,
        report_date: str | None,
        metadata: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> None:
        now = datetime.utcnow().isoformat()
        path_a, path_b = self._path_candidates(source_path)
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT 1 FROM report_versions
                WHERE (source_path = ? OR source_path = ?) AND content_hash = ?
                LIMIT 1
                """,
                (path_a, path_b, content_hash),
            ).fetchone()
            if existing is not None:
                return

            previous = conn.execute(
                """
                SELECT snapshot_json
                FROM report_versions
                WHERE source_path = ? OR source_path = ?
                ORDER BY ingested_at DESC
                LIMIT 1
                """,
                (path_a, path_b),
            ).fetchone()

            prev_snapshot = json.loads(previous[0]) if previous else None
            change_summary = self._build_version_diff(prev_snapshot, snapshot)

            conn.execute(
                """
                INSERT INTO report_versions (
                    source_path, content_hash, report_date, ingested_at, metadata_json, snapshot_json, change_summary_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path_a,
                    content_hash,
                    report_date,
                    now,
                    json.dumps(metadata, ensure_ascii=True),
                    json.dumps(snapshot, ensure_ascii=True),
                    json.dumps(change_summary, ensure_ascii=True),
                ),
            )

    def list_report_versions(self, source_path: str) -> list[dict[str, Any]]:
        path_a, path_b = self._path_candidates(source_path)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_path, content_hash, report_date, ingested_at, metadata_json, snapshot_json, change_summary_json
                FROM report_versions
                WHERE source_path = ? OR source_path = ?
                ORDER BY ingested_at DESC
                """,
                (path_a, path_b),
            ).fetchall()

        return [
            {
                "source_path": row[0],
                "content_hash": row[1],
                "report_date": row[2],
                "ingested_at": row[3],
                "metadata": json.loads(row[4]),
                "snapshot": json.loads(row[5]),
                "change_summary": json.loads(row[6]),
            }
            for row in rows
        ]

    def _build_version_diff(self, previous: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
        if previous is None:
            return ["Initial version captured."]

        changes: list[str] = []

        for key in ["report_type", "document_kind", "client_name", "report_date"]:
            old = previous.get(key)
            new = current.get(key)
            if old != new:
                changes.append(f"{key} changed from '{old}' to '{new}'.")

        if previous.get("content_hash") != current.get("content_hash"):
            changes.append("Document content changed from previous version.")

        for list_key, label in [
            ("authors", "Authors"),
            ("contacts", "Contacts"),
            ("key_findings", "Key findings"),
            ("recommendations", "Recommendations"),
        ]:
            old_list = set(str(v) for v in previous.get(list_key, []) or [])
            new_list = set(str(v) for v in current.get(list_key, []) or [])
            added = sorted(new_list - old_list)
            removed = sorted(old_list - new_list)
            if added:
                changes.append(f"{label} added: {', '.join(added[:4])}.")
            if removed:
                changes.append(f"{label} removed: {', '.join(removed[:4])}.")

        if not changes:
            changes.append("No significant structured changes detected.")
        return changes
