from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from client_profiler.classification import DocumentClassifier
from client_profiler.config import ProfilerConfig
from client_profiler.embeddings import LocalEmbedder, VectorRetriever
from client_profiler.extraction import OllamaClient, ProfileExtractor
from client_profiler.ingestion import DocumentReader
from client_profiler.profiling import ProfileBuilder
from client_profiler.projects import ProjectAssociator, ProjectSummaryService
from client_profiler.storage import SqliteStorage


StatusCallback = Callable[[dict[str, Any]], None]


class ClientProfiler:
    def __init__(self, config: ProfilerConfig) -> None:
        self.config = config
        self.config.ensure_dirs()

        self.storage = SqliteStorage(config.db_path)
        self.reader = DocumentReader()
        self.classifier = DocumentClassifier()
        self.embedder = LocalEmbedder(config.embedding_model)
        self.retriever = VectorRetriever(self.storage)
        self.profile_builder = ProfileBuilder(self.storage)

        llm = None
        if config.llm_provider == "ollama":
            llm = OllamaClient(config.ollama_base_url, config.llm_model, timeout=config.ollama_timeout_seconds)

        extraction_llm = llm if config.ingest_use_llm_extraction else None
        self.extractor = ProfileExtractor(extraction_llm, config)
        self.project_associator = ProjectAssociator(self.storage, llm)
        self.summary_service = ProjectSummaryService(
            storage=self.storage,
            embedder=self.embedder,
            retriever=self.retriever,
            llm=llm,
            questionnaire_path=config.project_summary_questionnaire_path,
        )

    def _rebind_storage(self) -> None:
        self.storage = SqliteStorage(self.config.db_path)
        self.retriever = VectorRetriever(self.storage)
        self.profile_builder.storage = self.storage
        self.project_associator.storage = self.storage
        self.summary_service.storage = self.storage
        self.summary_service.retriever = self.retriever

    def reset_db(self, backup: bool = True) -> dict[str, Any]:
        db_path = self.config.db_path
        backup_path: Path | None = None

        if db_path.exists():
            if backup:
                stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                candidate = db_path.with_name(f"{db_path.name}.bak_{stamp}")
                suffix = 1
                while candidate.exists():
                    candidate = db_path.with_name(f"{db_path.name}.bak_{stamp}_{suffix}")
                    suffix += 1
                db_path.replace(candidate)
                backup_path = candidate
            else:
                db_path.unlink()

        self._rebind_storage()
        return {
            "reset": True,
            "db_path": str(db_path),
            "backup_path": str(backup_path) if backup_path is not None else "",
        }

    def cleanup_high_confidence_client_merges(self, min_confidence: float | None = None, dry_run: bool = False) -> dict[str, Any]:
        threshold = float(min_confidence) if min_confidence is not None else float(self.config.merge_cleanup_min_confidence)
        result = self.storage.cleanup_high_confidence_client_merges(min_confidence=threshold, dry_run=dry_run)

        if (not dry_run) and self.config.ingest_reconcile_projects:
            targets = sorted(
                {
                    str(row.get("target_client") or "").strip()
                    for row in result.get("merged", [])
                    if isinstance(row, dict)
                }
                - {""}
            )
            for client_name in targets:
                self.project_associator.reconcile_client_projects(client_name, apply_changes=True)
            result["reconciled_clients"] = targets

        return result

    def ingest_file(
        self,
        path: Path,
        force_reingest: bool = False,
        status_callback: StatusCallback | None = None,
        run_reconciliation: bool = True,
    ) -> dict:
        _emit_status(status_callback, event="file_started", path=str(path), force_reingest=force_reingest)
        doc = self.reader.read(path)
        _emit_status(
            status_callback,
            event="file_read",
            path=str(path),
            source_type=doc.source_type,
            text_chars=len(doc.text),
        )
        content_hash = hashlib.sha256(doc.text.encode("utf-8", errors="ignore")).hexdigest()
        if (not force_reingest) and self.storage.document_already_ingested(str(doc.source_path), content_hash):
            _emit_status(
                status_callback,
                event="file_skipped_duplicate",
                path=str(path),
                content_hash=content_hash,
            )
            return {
                "path": str(path),
                "status": "skipped_duplicate",
                "reason": "Matching source path and content hash already ingested.",
                "content_hash": content_hash,
            }

        classification = self.classifier.classify(doc.text)
        _emit_status(
            status_callback,
            event="classification_completed",
            path=str(path),
            document_kind=classification.document_kind,
            is_client_related=classification.is_client_related,
            confidence=classification.confidence,
        )
        extracted = self.extractor.extract(doc.text, classification)
        _emit_status(
            status_callback,
            event="extraction_completed",
            path=str(path),
            events_extracted=len(extracted.events),
            key_findings=len(extracted.insight.key_findings),
            recommendations=len(extracted.insight.recommendations),
        )

        explicit_client_name = self._guess_client_name(path, doc.text)
        inferred_by_references = False
        if extracted.client_name is None:
            if explicit_client_name and classification.is_client_related:
                extracted.client_name = explicit_client_name
            else:
                inferred_client = self._infer_client_from_references(extracted)
                if inferred_client:
                    inferred_by_references = True
                    extracted.client_name = inferred_client

        if extracted.client_name and not self._is_confident_client_name(
            extracted.client_name,
            path,
            doc.text,
            classification.confidence,
            explicit_client_name=explicit_client_name,
            inferred_by_references=inferred_by_references,
        ):
            extracted.client_name = None

        project_details = self.project_associator.resolve_project(
            str(doc.source_path),
            doc.text,
            extracted,
            allow_llm_match=self.config.ingest_use_llm_project_matching,
        )
        _emit_status(
            status_callback,
            event="project_resolution_completed",
            path=str(path),
            project_key=project_details.get("project_key"),
            project_name=project_details.get("project_name"),
        )
        extracted.additional_fields.update(project_details)

        report_date = extracted.additional_fields.get("report_date")
        self.storage.save_document_record(
            source_path=str(doc.source_path),
            source_type=doc.source_type,
            content_hash=content_hash,
            metadata={
                **doc.metadata,
                "title": _guess_title(doc.text, path),
                "document_kind": classification.document_kind,
                "is_client_related": classification.is_client_related,
                "client_name": extracted.client_name,
                "report_type": extracted.insight.report_type,
                "authors": extracted.insight.authors,
                "contacts": extracted.insight.contacts,
                "report_date": report_date,
                "project_key": extracted.additional_fields.get("project_key"),
                "project_name": extracted.additional_fields.get("project_name"),
                "project_code": extracted.additional_fields.get("project_code"),
                "quote_number": extracted.additional_fields.get("quote_number"),
                "purchase_order_number": extracted.additional_fields.get("purchase_order_number"),
                "access_reference": extracted.additional_fields.get("access_reference"),
                "related_references": extracted.additional_fields.get("related_references", []),
            },
        )
        _emit_status(status_callback, event="document_record_saved", path=str(path))

        version_snapshot = {
            "content_hash": content_hash,
            "client_name": extracted.client_name,
            "report_type": extracted.insight.report_type,
            "document_kind": classification.document_kind,
            "report_date": report_date,
            "project_name": extracted.additional_fields.get("project_name"),
            "project_code": extracted.additional_fields.get("project_code"),
            "authors": extracted.insight.authors,
            "contacts": extracted.insight.contacts,
            "key_findings": extracted.insight.key_findings,
            "recommendations": extracted.insight.recommendations,
        }
        self.storage.add_report_version(
            source_path=str(doc.source_path),
            content_hash=content_hash,
            report_date=report_date,
            metadata={
                "title": _guess_title(doc.text, path),
                "client_name": extracted.client_name,
                "document_kind": classification.document_kind,
                "report_type": extracted.insight.report_type,
            },
            snapshot=version_snapshot,
        )
        _emit_status(status_callback, event="report_version_saved", path=str(path))

        if classification.is_client_related and extracted.client_name:
            project_key = extracted.additional_fields.get("project_key")
            project_name = extracted.additional_fields.get("project_name")
            self.profile_builder.apply(str(path), extracted)
            _emit_status(
                status_callback,
                event="profile_updated",
                path=str(path),
                client_name=extracted.client_name,
                project_key=project_key,
                project_name=project_name,
            )

        chunks = _chunk_text(doc.text, self.config.default_chunk_size, self.config.default_chunk_overlap)
        _emit_status(
            status_callback,
            event="chunk_embedding_started",
            path=str(path),
            total_chunks=len(chunks),
        )
        for idx, chunk in enumerate(chunks):
            embedding = self.embedder.embed_text(chunk)
            self.storage.add_vector(
                source_document=str(path),
                chunk_text=chunk,
                embedding=embedding,
                metadata={
                    "document_kind": classification.document_kind,
                    "is_client_related": classification.is_client_related,
                    "source_type": doc.source_type,
                    "report_date": report_date,
                    "title": _guess_title(doc.text, path),
                    "report_type": extracted.insight.report_type,
                    "project_key": extracted.additional_fields.get("project_key"),
                    "project_name": extracted.additional_fields.get("project_name"),
                    "project_code": extracted.additional_fields.get("project_code"),
                    "related_references": extracted.additional_fields.get("related_references", []),
                    "chunk_index": idx,
                },
                client_name=extracted.client_name,
            )
            _emit_status(
                status_callback,
                event="chunk_embedded",
                path=str(path),
                chunk_index=idx + 1,
                total_chunks=len(chunks),
            )
        _emit_status(
            status_callback,
            event="chunk_embedding_completed",
            path=str(path),
            total_chunks=len(chunks),
        )

        if (
            self.config.ingest_generate_project_summaries
            and classification.is_client_related
            and extracted.client_name
        ):
            project_key = str(extracted.additional_fields.get("project_key") or "").strip()
            project_name = str(extracted.additional_fields.get("project_name") or "").strip()
            if project_key and project_name:
                existing = self.storage.get_project_summary(extracted.client_name, project_key)
                if existing is None:
                    _emit_status(
                        status_callback,
                        event="summary_generation_started",
                        path=str(path),
                        client_name=extracted.client_name,
                        project_key=project_key,
                        project_name=project_name,
                    )
                    project_docs = self.storage.list_project_documents(extracted.client_name, project_key)
                    self.summary_service.generate_and_store(
                        client_name=extracted.client_name,
                        project_key=project_key,
                        project_name=project_name,
                        documents=project_docs,
                    )
                    _emit_status(
                        status_callback,
                        event="summary_generation_completed",
                        path=str(path),
                        client_name=extracted.client_name,
                        project_key=project_key,
                        project_name=project_name,
                    )

        if self.config.ingest_reconcile_projects and run_reconciliation and extracted.client_name:
            _emit_status(
                status_callback,
                event="project_reconciliation_started",
                path=str(path),
                client_name=extracted.client_name,
            )
            self.project_associator.reconcile_client_projects(
                extracted.client_name,
                apply_changes=True,
            )
            _emit_status(
                status_callback,
                event="project_reconciliation_completed",
                path=str(path),
                client_name=extracted.client_name,
            )

        cleanup_result: dict[str, Any] | None = None
        if self.config.ingest_auto_merge_cleanup and run_reconciliation:
            _emit_status(
                status_callback,
                event="merge_cleanup_started",
                path=str(path),
                min_confidence=self.config.merge_cleanup_min_confidence,
            )
            cleanup_result = self.cleanup_high_confidence_client_merges(
                min_confidence=self.config.merge_cleanup_min_confidence,
                dry_run=False,
            )
            _emit_status(
                status_callback,
                event="merge_cleanup_completed",
                path=str(path),
                candidate_count=int(cleanup_result.get("candidate_count") or 0),
                merged_count=int(cleanup_result.get("merged_count") or 0),
            )

        result = {
            "path": str(path),
            "status": "reingested" if force_reingest else "ingested",
            "document_kind": classification.document_kind,
            "is_client_related": classification.is_client_related,
            "client_name": extracted.client_name,
            "report_date": report_date,
            "confidence": classification.confidence,
            "rationale": classification.rationale,
            "events_extracted": len(extracted.events),
            "key_findings": len(extracted.insight.key_findings),
            "recommendations": len(extracted.insight.recommendations),
            "ingest_llm_enabled": self.config.ingest_use_llm_extraction,
            "ingest_summary_generated": self.config.ingest_generate_project_summaries,
            "merge_cleanup_candidate_count": int((cleanup_result or {}).get("candidate_count") or 0),
            "merge_cleanup_merged_count": int((cleanup_result or {}).get("merged_count") or 0),
        }
        _emit_status(
            status_callback,
            event="file_completed",
            path=str(path),
            status=result.get("status"),
            document_kind=result.get("document_kind"),
            client_name=result.get("client_name"),
        )
        return result

    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
        force_reingest: bool = False,
        status_callback: StatusCallback | None = None,
    ) -> list[dict]:
        iterator = directory.rglob("*") if recursive else directory.glob("*")
        file_paths = [file_path for file_path in iterator if file_path.is_file()]
        _emit_status(
            status_callback,
            event="directory_scanned",
            directory=str(directory),
            recursive=recursive,
            total_files=len(file_paths),
            force_reingest=force_reingest,
        )
        results: list[dict] = []
        for file_path in file_paths:
            try:
                results.append(
                    self.ingest_file(
                        file_path,
                        force_reingest=force_reingest,
                        status_callback=status_callback,
                        run_reconciliation=False,
                    )
                )
            except Exception as exc:
                _emit_status(
                    status_callback,
                    event="file_error",
                    path=str(file_path),
                    error=str(exc),
                )
                results.append(
                    {
                        "path": str(file_path),
                        "error": str(exc),
                    }
                )

        if self.config.ingest_reconcile_projects:
            touched_clients = sorted(
                {
                    str(row.get("client_name") or "").strip()
                    for row in results
                    if isinstance(row, dict)
                }
                - {""}
            )
            for client_name in touched_clients:
                _emit_status(
                    status_callback,
                    event="project_reconciliation_started",
                    directory=str(directory),
                    client_name=client_name,
                )
                self.project_associator.reconcile_client_projects(client_name, apply_changes=True)
                _emit_status(
                    status_callback,
                    event="project_reconciliation_completed",
                    directory=str(directory),
                    client_name=client_name,
                )

        cleanup_result: dict[str, Any] | None = None
        if self.config.ingest_auto_merge_cleanup:
            _emit_status(
                status_callback,
                event="merge_cleanup_started",
                directory=str(directory),
                min_confidence=self.config.merge_cleanup_min_confidence,
            )
            cleanup_result = self.cleanup_high_confidence_client_merges(
                min_confidence=self.config.merge_cleanup_min_confidence,
                dry_run=False,
            )
            _emit_status(
                status_callback,
                event="merge_cleanup_completed",
                directory=str(directory),
                candidate_count=int(cleanup_result.get("candidate_count") or 0),
                merged_count=int(cleanup_result.get("merged_count") or 0),
            )

        if cleanup_result is not None:
            for row in results:
                if isinstance(row, dict) and "error" not in row:
                    row["merge_cleanup_candidate_count"] = int(cleanup_result.get("candidate_count") or 0)
                    row["merge_cleanup_merged_count"] = int(cleanup_result.get("merged_count") or 0)

        _emit_status(
            status_callback,
            event="directory_completed",
            directory=str(directory),
            total_files=len(file_paths),
            succeeded=sum(1 for row in results if "error" not in row),
            failed=sum(1 for row in results if "error" in row),
        )
        return results

    def _guess_client_name(self, path: Path, text: str) -> str | None:
        pattern = re.compile(r"(?im)^\s*(?:client\s*name|client|for\s+client)\s*[:\-]\s*(.+)$")
        match = pattern.search(text[:8000])
        if match:
            candidate = self._clean_candidate_name(match.group(1))
            if candidate and not self._looks_like_document_name(candidate, path):
                return candidate

        for line in text.splitlines()[:40]:
            lower = line.lower()
            if "client" in lower and ":" in line:
                candidate = self._clean_candidate_name(line.split(":", 1)[1])
                if candidate and not self._looks_like_document_name(candidate, path):
                    return candidate
        return None

    def _infer_client_from_references(self, extracted) -> str | None:
        refs = []
        project = extracted.project_context
        for value in [
            project.project_code,
            project.quote_number,
            project.purchase_order_number,
            project.access_reference,
            *project.related_references,
        ]:
            if isinstance(value, str) and value.strip():
                refs.append(value.strip())
        clients = self.storage.find_clients_by_references(refs)
        if len(clients) == 1:
            return clients[0]
        return None

    def _is_confident_client_name(
        self,
        candidate: str,
        path: Path,
        text: str,
        classifier_confidence: float,
        explicit_client_name: str | None,
        inferred_by_references: bool,
    ) -> bool:
        name = str(candidate or "").strip()
        if len(name) < 3 or not any(ch.isalpha() for ch in name):
            return False

        if self._looks_like_document_name(name, path):
            return False

        existing_clients = {c.lower() for c in self.storage.list_clients()}
        if explicit_client_name and name.lower() == explicit_client_name.lower():
            return True
        if inferred_by_references:
            return True
        if name.lower() in existing_clients:
            return True

        has_strong_marker = bool(re.search(r"(?im)^\s*(?:client\s*name|client|for\s+client)\s*[:\-]", text[:8000]))
        if has_strong_marker and classifier_confidence >= 0.7:
            return True

        return False

    def _looks_like_document_name(self, value: str, path: Path) -> bool:
        key = self._name_key(value)
        if not key:
            return True
        doc_key = self._name_key(path.stem)
        parent_key = self._name_key(path.parent.name)
        return key in {doc_key, parent_key}

    def _name_key(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = Path(text).stem
        return "".join(ch for ch in text if ch.isalnum())

    def _clean_candidate_name(self, value: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"^[\-*#\s`_]+", "", text)
        text = re.sub(r"[\-*`_\s]+$", "", text)
        text = text.replace("**", "").replace("__", "")
        text = re.sub(r"\s+", " ", text).strip(" :;,.\t")
        return text

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [re.sub(r"\s+", " ", block).strip() for block in re.split(r"\n\s*\n+", normalized)]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return []

    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
            continue

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        if not sentences:
            sentences = [para]

        current = ""
        for sentence in sentences:
            if not current:
                current = sentence
                continue
            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            units.append(current)
            current = sentence
        if current:
            units.append(current)

    chunks: list[str] = []
    current = ""
    overlap = max(0, int(overlap))
    for unit in units:
        candidate = unit if not current else f"{current}\n{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            tail = current[-overlap:].strip() if overlap else ""
            current = f"{tail} {unit}".strip() if tail else unit
            if len(current) <= chunk_size:
                continue

        # Fallback for rare very long units.
        start = 0
        step = max(1, chunk_size - overlap)
        while start < len(unit):
            piece = unit[start : start + chunk_size].strip()
            if piece:
                chunks.append(piece)
            start += step
        current = ""

    if current:
        chunks.append(current.strip())
    return chunks


def _guess_title(text: str, path: Path) -> str:
    for line in text.splitlines()[:20]:
        candidate = line.strip()
        if len(candidate) > 8:
            return candidate[:200]
    return path.stem.replace("_", " ")


def _emit_status(callback: StatusCallback | None, **payload: Any) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        return
