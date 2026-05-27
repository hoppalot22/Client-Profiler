from __future__ import annotations

import hashlib
import re
from pathlib import Path

from client_profiler.classification import DocumentClassifier
from client_profiler.config import ProfilerConfig
from client_profiler.embeddings import LocalEmbedder, VectorRetriever
from client_profiler.extraction import OllamaClient, ProfileExtractor
from client_profiler.ingestion import DocumentReader
from client_profiler.profiling import ProfileBuilder
from client_profiler.projects import ProjectAssociator, ProjectSummaryService
from client_profiler.storage import SqliteStorage


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

        self.extractor = ProfileExtractor(llm, config)
        self.project_associator = ProjectAssociator(self.storage, llm)
        self.summary_service = ProjectSummaryService(
            storage=self.storage,
            embedder=self.embedder,
            retriever=self.retriever,
            llm=llm,
            questionnaire_path=config.project_summary_questionnaire_path,
        )

    def ingest_file(self, path: Path, force_reingest: bool = False) -> dict:
        doc = self.reader.read(path)
        content_hash = hashlib.sha256(doc.text.encode("utf-8", errors="ignore")).hexdigest()
        if (not force_reingest) and self.storage.document_already_ingested(str(doc.source_path), content_hash):
            return {
                "path": str(path),
                "status": "skipped_duplicate",
                "reason": "Matching source path and content hash already ingested.",
                "content_hash": content_hash,
            }

        classification = self.classifier.classify(doc.text)
        extracted = self.extractor.extract(doc.text, classification)

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

        project_details = self.project_associator.resolve_project(str(doc.source_path), doc.text, extracted)
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

        if classification.is_client_related and extracted.client_name:
            project_key = extracted.additional_fields.get("project_key")
            project_name = extracted.additional_fields.get("project_name")
            self.profile_builder.apply(str(path), extracted)

        for chunk in _chunk_text(doc.text, self.config.default_chunk_size, self.config.default_chunk_overlap):
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
                },
                client_name=extracted.client_name,
            )

        if classification.is_client_related and extracted.client_name:
            project_key = str(extracted.additional_fields.get("project_key") or "").strip()
            project_name = str(extracted.additional_fields.get("project_name") or "").strip()
            if project_key and project_name:
                existing = self.storage.get_project_summary(extracted.client_name, project_key)
                if existing is None:
                    project_docs = self.storage.list_project_documents(extracted.client_name, project_key)
                    self.summary_service.generate_and_store(
                        client_name=extracted.client_name,
                        project_key=project_key,
                        project_name=project_name,
                        documents=project_docs,
                    )

        return {
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
        }

    def ingest_directory(self, directory: Path, recursive: bool = True, force_reingest: bool = False) -> list[dict]:
        files = directory.rglob("*") if recursive else directory.glob("*")
        results: list[dict] = []
        for file_path in files:
            if not file_path.is_file():
                continue
            try:
                results.append(self.ingest_file(file_path, force_reingest=force_reingest))
            except Exception as exc:
                results.append(
                    {
                        "path": str(file_path),
                        "error": str(exc),
                    }
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
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(cleaned):
        chunk = cleaned[start : start + chunk_size]
        chunks.append(chunk)
        start += step
    return chunks


def _guess_title(text: str, path: Path) -> str:
    for line in text.splitlines()[:20]:
        candidate = line.strip()
        if len(candidate) > 8:
            return candidate[:200]
    return path.stem.replace("_", " ")
