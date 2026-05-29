from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from client_profiler.config import ProfilerConfig
from client_profiler.models import (
    DocumentClassification,
    ExtractedEvent,
    ExtractedInsight,
    ExtractedProfileData,
    ProjectContext,
)

from .llm import LLMClient


class ProfileExtractor:
    def __init__(self, llm: LLMClient | None, config: ProfilerConfig) -> None:
        self.llm = llm
        self.config = config

    def extract(self, text: str, classification: DocumentClassification) -> ExtractedProfileData:
        llm_text = text[: self.config.max_text_chars_for_llm]
        regex_text = text[: self.config.max_text_chars_for_regex]
        regex_result = self._extract_with_regex(regex_text, classification)

        # Small local models are more reliable when reserved for low-confidence cases.
        should_use_llm = (
            self.llm is not None
            and (
                classification.document_kind == "unknown"
                or classification.confidence < 0.8
                or classification.document_kind == "report"
            )
        )
        if should_use_llm:
            llm_data = self._extract_with_llm(llm_text, classification)
            if llm_data:
                llm_result = self._from_dict(llm_data, classification)
                return self._merge_with_regex_fallback(llm_result, regex_result)

        return regex_result

    def _merge_with_regex_fallback(
        self,
        llm_result: ExtractedProfileData,
        regex_result: ExtractedProfileData,
    ) -> ExtractedProfileData:
        if not llm_result.client_name and regex_result.client_name:
            llm_result.client_name = regex_result.client_name

        if not llm_result.events and regex_result.events:
            llm_result.events = regex_result.events

        if not llm_result.insight.key_findings and regex_result.insight.key_findings:
            llm_result.insight.key_findings = regex_result.insight.key_findings
        if not llm_result.insight.recommendations and regex_result.insight.recommendations:
            llm_result.insight.recommendations = regex_result.insight.recommendations
        if not llm_result.insight.contacts and regex_result.insight.contacts:
            llm_result.insight.contacts = regex_result.insight.contacts
        if not llm_result.insight.authors and regex_result.insight.authors:
            llm_result.insight.authors = regex_result.insight.authors
        if not llm_result.insight.report_type and regex_result.insight.report_type:
            llm_result.insight.report_type = regex_result.insight.report_type

        llm_project = llm_result.project_context
        regex_project = regex_result.project_context
        if not llm_project.project_name and regex_project.project_name:
            llm_project.project_name = regex_project.project_name
        if not llm_project.project_code and regex_project.project_code:
            llm_project.project_code = regex_project.project_code
        if not llm_project.quote_number and regex_project.quote_number:
            llm_project.quote_number = regex_project.quote_number
        if not llm_project.purchase_order_number and regex_project.purchase_order_number:
            llm_project.purchase_order_number = regex_project.purchase_order_number
        if not llm_project.access_reference and regex_project.access_reference:
            llm_project.access_reference = regex_project.access_reference
        if not llm_project.related_references and regex_project.related_references:
            llm_project.related_references = regex_project.related_references

        if not llm_result.additional_fields.get("report_date") and regex_result.additional_fields.get("report_date"):
            llm_result.additional_fields["report_date"] = regex_result.additional_fields.get("report_date")

        if not llm_result.hierarchy_paths and regex_result.hierarchy_paths:
            llm_result.hierarchy_paths = regex_result.hierarchy_paths

        return llm_result

    def _extract_with_llm(self, text: str, classification: DocumentClassification) -> dict[str, Any]:
        llm = self.llm
        if llm is None:
            return {}
        prompt = self._build_prompt(text, classification)
        try:
            return llm.extract_structured(prompt)
        except Exception:
            return {}

    def _build_prompt(self, text: str, classification: DocumentClassification) -> str:
        return (
            "You are a document profiling assistant. Extract structured fields as JSON with keys: "
            "client_name, events (list of {date,title,details}), insight ({key_findings,recommendations,contacts,report_type,authors,project_areas}), "
            "project_context ({project_name,project_code,quote_number,purchase_order_number,access_reference,related_references}), "
            "hierarchy_paths (list), additional_fields (object). "
            f"Document kind hint: {classification.document_kind}. "
            "Only return valid JSON and no markdown.\n\n"
            f"Document text:\n{text}"
        )

    def _from_dict(self, data: dict[str, Any], classification: DocumentClassification) -> ExtractedProfileData:
        events_raw = data.get("events", []) if isinstance(data.get("events", []), list) else []
        events = [
            ExtractedEvent(
                date=e.get("date"),
                title=e.get("title", "Event"),
                details=e.get("details", ""),
            )
            for e in events_raw
            if isinstance(e, dict)
        ]

        insight_raw = data.get("insight", {}) if isinstance(data.get("insight", {}), dict) else {}
        insight = ExtractedInsight(
            key_findings=_safe_list(insight_raw.get("key_findings")),
            recommendations=_safe_list(insight_raw.get("recommendations")),
            contacts=_safe_list(insight_raw.get("contacts")),
            report_type=insight_raw.get("report_type"),
            authors=_safe_list(insight_raw.get("authors")),
            project_areas=_safe_list(insight_raw.get("project_areas")),
        )

        project_raw = data.get("project_context", {}) if isinstance(data.get("project_context", {}), dict) else {}
        project_context = ProjectContext(
            project_name=_string_or_none(project_raw.get("project_name")),
            project_code=_string_or_none(project_raw.get("project_code")),
            quote_number=_string_or_none(project_raw.get("quote_number")),
            purchase_order_number=_string_or_none(project_raw.get("purchase_order_number")),
            access_reference=_string_or_none(project_raw.get("access_reference")),
            related_references=_safe_list(project_raw.get("related_references")),
        )

        hierarchy_paths = _safe_list(data.get("hierarchy_paths"))
        if not hierarchy_paths and insight.project_areas:
            hierarchy_paths = [f"General/{area}" for area in insight.project_areas]

        additional_fields = data.get("additional_fields", {})
        if not isinstance(additional_fields, dict):
            additional_fields = {}
        report_date = data.get("report_date") or additional_fields.get("report_date")
        if report_date:
            normalized = _normalize_date(str(report_date))
            additional_fields["report_date"] = normalized or str(report_date)

        return ExtractedProfileData(
            client_name=_clean_client_name(data.get("client_name")),
            classification=classification,
            events=events,
            insight=insight,
            project_context=project_context,
            hierarchy_paths=hierarchy_paths,
            additional_fields=additional_fields,
        )

    def _extract_with_regex(self, text: str, classification: DocumentClassification) -> ExtractedProfileData:
        client_name = _first_match(
            text,
            [
                r"(?im)^\s*client\s*name\s*[:\-]\s*(.+)$",
                r"(?im)^\s*client\s*[:\-]\s*(.+)$",
                r"(?im)^\s*for\s+client\s*[:\-]\s*(.+)$",
            ],
        )
        client_name = _clean_client_name(client_name)

        authors = _all_matches(text, [r"(?im)^\s*author\s*[:\-]\s*(.+)$", r"(?im)^\s*prepared\s+by\s*[:\-]\s*(.+)$"])

        contacts = _all_matches(
            text,
            [r"(?im)^\s*contact\s*[:\-]\s*(.+)$", r"(?im)^\s*client\s+contact\s*[:\-]\s*(.+)$"],
        )

        report_type = _first_match(text, [r"(?im)^\s*report\s*type\s*[:\-]\s*(.+)$", r"(?im)^\s*document\s*type\s*[:\-]\s*(.+)$"])
        report_date = _first_match(
            text,
            [
                r"(?im)^\s*report\s*date\s*[:\-]\s*(\d{4}-\d{2}-\d{2})",
                r"(?im)^\s*date\s*[:\-]\s*(\d{4}-\d{2}-\d{2})",
                r"(?im)^\s*date\s*[:\-]\s*(\d{1,2}/\d{1,2}/\d{2,4})",
                r"(?im)^\s*report\s*date\s*[:\-]\s*([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})",
                r"(?im)^\s*date\s*[:\-]\s*([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})",
                r"(?im)^\s*report\s*date\s*[:\-]\s*(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})",
                r"(?im)^\s*date\s*[:\-]\s*(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})",
            ],
        )
        report_date = _normalize_date(report_date)

        raw_date_candidates = _all_matches(
            text,
            [
                r"\b(\d{4}-\d{2}-\d{2})\b",
                r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
                r"\b([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})\b",
                r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b",
            ],
        )
        date_candidates = [d for d in (_normalize_date(value) for value in raw_date_candidates) if d]
        date_candidates = list(dict.fromkeys(date_candidates))
        if report_date is None and date_candidates:
            report_date = date_candidates[0]
        events = [
            ExtractedEvent(date=d, title="Document date reference", details="Date found in document text")
            for d in date_candidates[:10]
        ]

        key_findings = _section_lines(text, ["findings", "key findings"], limit=8)
        recommendations = _section_lines(text, ["recommendations", "actions"], limit=8)

        project_areas = _infer_areas(text)
        hierarchy_paths = ["General"] if project_areas == ["General"] else [f"General/{a}" for a in project_areas]
        project_context = ProjectContext(
            project_name=_first_match(
                text,
                [
                    r"(?im)^\s*project\s*(?:name)?\s*[:\-]\s*(.+)$",
                    r"(?im)^\s*subject\s*[:\-].*?\b(project|outage|upgrade|review)\b[:\-\s]*(.+)$",
                ],
            ),
            project_code=_first_match(
                text,
                [
                    r"(?im)^\s*project\s*code\s*[:\-]\s*([A-Z0-9\-/]+)$",
                    r"(?im)\b(?:project\s*#|job\s*#|job\s*code|reference)\s*[:#\- ]\s*([A-Z]{2,}[A-Z0-9\-/]+)\b",
                ],
            ),
            quote_number=_first_match(
                text,
                [r"(?im)\b(?:quote|quotation|estimate)(?:\s+(?:number|no\.?))?\s*(?:[:#\-]|\b)\s*([A-Z0-9\-/]+)\b"],
            ),
            purchase_order_number=_first_match(
                text,
                [r"(?im)\b(?:purchase\s*order|po)(?:\s+(?:number|no\.?))?\s*(?:[:#\-]|\b)\s*([A-Z0-9\-/]+)\b"],
            ),
            access_reference=_first_match(
                text,
                [r"(?im)\b(?:access\s*request|permit\s*reference|permit\s*to\s*work)(?:\s+(?:number|no\.?))?\s*(?:[:#\-]|\b)\s*([A-Z0-9\-/]+)\b"],
            ),
            related_references=_collect_references(text),
        )

        insight = ExtractedInsight(
            key_findings=key_findings,
            recommendations=recommendations,
            contacts=contacts,
            report_type=report_type,
            authors=authors,
            project_areas=project_areas,
        )

        return ExtractedProfileData(
            client_name=client_name,
            classification=classification,
            events=events,
            insight=insight,
            project_context=project_context,
            hierarchy_paths=hierarchy_paths,
            additional_fields={"extraction_mode": "regex_fallback", "report_date": report_date},
        )


def _safe_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def _all_matches(text: str, patterns: list[str]) -> list[str]:
    values: list[str] = []
    for pattern in patterns:
        values.extend([m.strip() for m in re.findall(pattern, text)])
    unique = []
    seen = set()
    for v in values:
        if v and v not in seen:
            unique.append(v)
            seen.add(v)
    return unique


def _section_lines(text: str, headers: list[str], limit: int = 6) -> list[str]:
    lines = [l.strip() for l in text.splitlines()]
    collected: list[str] = []
    header_set = {h.lower() for h in headers}

    for idx, line in enumerate(lines):
        normalized = line.lower().strip(" :")
        if normalized in header_set:
            for next_line in lines[idx + 1 : idx + 1 + limit * 2]:
                if not next_line:
                    continue
                if re.match(r"^[A-Z][A-Za-z\s]{2,}$", next_line) and next_line.lower() not in header_set:
                    break
                collected.append(next_line.lstrip("-•* "))
                if len(collected) >= limit:
                    return collected
    return collected


def _infer_areas(text: str) -> list[str]:
    area_patterns = {
        "Piping": r"\b(pipe|piping|pipeline|weld)\b",
        "Electrical": r"\b(electrical|switchboard|cable|transformer)\b",
        "Mechanical": r"\b(pump|compressor|bearing|shaft|valve)\b",
        "Civil": r"\b(concrete|foundation|structural|beam|column)\b",
        "Operations": r"\b(operation|outage|shutdown|throughput)\b",
    }
    found = [name for name, pattern in area_patterns.items() if re.search(pattern, text, re.IGNORECASE)]
    return found or ["General"]


def _collect_references(text: str) -> list[str]:
    patterns = [
        r"(?im)\b(?:quote|quotation|estimate)(?:\s+(?:number|no\.?))?\s*(?:[:#\-]|\b)\s*([A-Z0-9\-/]+)\b",
        r"(?im)\b(?:purchase\s*order|po)(?:\s+(?:number|no\.?))?\s*(?:[:#\-]|\b)\s*([A-Z0-9\-/]+)\b",
        r"(?im)\b(?:access\s*request|permit\s*reference|permit\s*to\s*work)(?:\s+(?:number|no\.?))?\s*(?:[:#\-]|\b)\s*([A-Z0-9\-/]+)\b",
        r"(?im)\b(?:project\s*code|job\s*code|job\s*#|reference)\s*[:#\- ]\s*([A-Z]{2,}[A-Z0-9\-/]+)\b",
    ]
    refs: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, text):
            value = str(match).strip()
            if value and value not in seen:
                refs.append(value)
                seen.add(value)
    return refs


def _normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw

    formats = [
        "%m/%d/%Y",
        "%m/%d/%y",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _clean_client_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^[\-*#\s`_]+", "", text)
    text = re.sub(r"[\-*`_\s]+$", "", text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"\s+", " ", text).strip(" :;,.\t")
    return text or None
