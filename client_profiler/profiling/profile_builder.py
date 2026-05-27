from __future__ import annotations

from client_profiler.models import ExtractedProfileData, ProfileNode, TimelineItem
from client_profiler.storage import SqliteStorage


class ProfileBuilder:
    def __init__(self, storage: SqliteStorage) -> None:
        self.storage = storage

    def apply(self, source_document: str, extracted: ExtractedProfileData) -> None:
        if not extracted.client_name:
            return

        base_path = "General"
        project_path = self._project_node_path(extracted)
        if project_path:
            self._upsert_project_node(extracted.client_name, project_path, extracted, source_document)

        scoped_paths = self._scoped_paths(project_path, extracted.hierarchy_paths)
        if scoped_paths:
            for path in scoped_paths:
                self._upsert_node(extracted.client_name, path, extracted, source_document)
        else:
            self._upsert_node(extracted.client_name, project_path or base_path, extracted, source_document)

        report_date = extracted.additional_fields.get("report_date")
        if not extracted.events:
            extracted.events.append(
                TimelineItem(
                    client_name=extracted.client_name,
                    date=report_date,
                    summary="Report ingested",
                    source_document=source_document,
                )
            )

        for event in extracted.events:
            if isinstance(event, TimelineItem):
                self.storage.add_timeline_item(event)
                continue
            summary = f"{event.title}: {event.details}".strip(": ")
            self.storage.add_timeline_item(
                TimelineItem(
                    client_name=extracted.client_name,
                    date=event.date or report_date,
                    summary=summary,
                    source_document=source_document,
                )
            )

    def _upsert_node(self, client_name: str, path: str, extracted: ExtractedProfileData, source_document: str) -> None:
        report_date = extracted.additional_fields.get("report_date")
        existing = self.storage.get_profile_node(client_name, path)
        existing_facts = existing.get("facts", {}) if existing else {}

        source_reports = self._merge_source_reports(existing_facts.get("source_reports", []), source_document, report_date)

        facts = {
            "document_kind": extracted.classification.document_kind,
            "report_type": extracted.insight.report_type,
            "authors": extracted.insight.authors,
            "contacts": extracted.insight.contacts,
            "key_findings": extracted.insight.key_findings,
            "recommendations": extracted.insight.recommendations,
            "report_date": report_date,
            "source_reports": source_reports,
            "project_name": extracted.additional_fields.get("project_name"),
            "project_code": extracted.additional_fields.get("project_code"),
            "project_key": extracted.additional_fields.get("project_key"),
            "related_references": extracted.additional_fields.get("related_references", []),
            "additional_fields": extracted.additional_fields,
        }

        if existing_facts:
            if self._has_meaningful_change(existing_facts, facts):
                self.storage.archive_profile_node_version(
                    client_name=client_name,
                    node_path=path,
                    facts=existing_facts,
                    superseded_by_document=source_document,
                )
            else:
                facts["key_findings"] = existing_facts.get("key_findings", facts["key_findings"])
                facts["recommendations"] = existing_facts.get("recommendations", facts["recommendations"])

        self.storage.upsert_profile_node(client_name, ProfileNode(path=path, facts=facts))

    def _upsert_project_node(
        self,
        client_name: str,
        path: str,
        extracted: ExtractedProfileData,
        source_document: str,
    ) -> None:
        report_date = extracted.additional_fields.get("report_date")
        existing = self.storage.get_profile_node(client_name, path)
        existing_facts = existing.get("facts", {}) if existing else {}
        source_reports = self._merge_source_reports(existing_facts.get("source_reports", []), source_document, report_date)

        facts = {
            "node_type": "project",
            "project_name": extracted.additional_fields.get("project_name"),
            "project_code": extracted.additional_fields.get("project_code"),
            "project_key": extracted.additional_fields.get("project_key"),
            "project_summary": extracted.additional_fields.get("project_summary"),
            "document_kind": extracted.classification.document_kind,
            "document_kinds": self._merge_unique(existing_facts.get("document_kinds", []), [extracted.classification.document_kind]),
            "authors": self._merge_unique(existing_facts.get("authors", []), extracted.insight.authors),
            "contacts": self._merge_unique(existing_facts.get("contacts", []), extracted.insight.contacts),
            "related_references": self._merge_unique(
                existing_facts.get("related_references", []),
                extracted.additional_fields.get("related_references", []),
            ),
            "report_date": report_date or existing_facts.get("report_date"),
            "source_reports": source_reports,
            "additional_fields": extracted.additional_fields,
        }

        if existing_facts and self._has_meaningful_change(existing_facts, facts):
            self.storage.archive_profile_node_version(
                client_name=client_name,
                node_path=path,
                facts=existing_facts,
                superseded_by_document=source_document,
            )

        self.storage.upsert_profile_node(client_name, ProfileNode(path=path, facts=facts))

    def _project_node_path(self, extracted: ExtractedProfileData) -> str | None:
        project_name = extracted.additional_fields.get("project_name")
        if not project_name:
            return None
        return f"Projects/{project_name}"

    def _scoped_paths(self, project_path: str | None, hierarchy_paths: list[str]) -> list[str]:
        if not hierarchy_paths:
            return []
        if not project_path:
            return hierarchy_paths

        scoped = []
        for raw_path in hierarchy_paths:
            path = str(raw_path).strip().strip("/")
            if not path:
                continue
            if path == "General":
                scoped.append(f"{project_path}/Workscope")
                continue
            if path.startswith("General/"):
                path = path[len("General/") :]
            scoped.append(f"{project_path}/{path}")
        return scoped

    def _merge_source_reports(self, existing: list, source_document: str, report_date: str | None) -> list[dict]:
        rows: list[dict] = []
        for entry in existing if isinstance(existing, list) else []:
            if isinstance(entry, dict) and entry.get("path"):
                rows.append({"path": str(entry["path"]), "date": entry.get("date")})

        found = False
        for row in rows:
            if row.get("path") == source_document:
                row["date"] = report_date or row.get("date")
                found = True
                break
        if not found:
            rows.append({"path": source_document, "date": report_date})
        return rows

    def _has_meaningful_change(self, old: dict, new: dict) -> bool:
        keys = [
            "document_kind",
            "report_type",
            "authors",
            "contacts",
            "key_findings",
            "recommendations",
            "report_date",
            "project_name",
            "project_code",
            "project_summary",
            "document_kinds",
            "related_references",
        ]
        for key in keys:
            if old.get(key) != new.get(key):
                return True
        return False

    def _merge_unique(self, existing: list, incoming: list) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for value in [*(existing if isinstance(existing, list) else []), *(incoming if isinstance(incoming, list) else [])]:
            text = str(value).strip()
            if text and text not in seen:
                merged.append(text)
                seen.add(text)
        return merged
