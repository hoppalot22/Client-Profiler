from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from client_profiler.models import ExtractedProfileData
from client_profiler.storage import SqliteStorage

from client_profiler.extraction.llm import LLMClient


class ProjectAssociator:
    def __init__(self, storage: SqliteStorage, llm: LLMClient | None = None) -> None:
        self.storage = storage
        self.llm = llm

    def resolve_project(self, source_path: str, text: str, extracted: ExtractedProfileData) -> dict[str, Any]:
        if not extracted.client_name:
            return {}

        inferred = self._infer_project(source_path, extracted)
        candidates = self.storage.list_project_documents(extracted.client_name)
        current_refs = set(inferred.get("related_references", []))

        matched = self._match_existing_project(inferred, candidates)
        if self.llm and candidates:
            llm_match = self._match_with_llm(text, extracted, inferred, candidates)
            if llm_match:
                matched = llm_match

        if matched:
            inferred["project_key"] = matched.get("project_key") or inferred["project_key"]
            inferred["project_name"] = matched.get("project_name") or inferred["project_name"]
            inferred["project_code"] = matched.get("project_code") or inferred["project_code"]
            refs = set(matched.get("related_references", []))
            inferred["related_references"] = sorted(refs | current_refs)

        return inferred

    def summarize_project(
        self,
        client_name: str,
        project_name: str,
        documents: list[dict[str, Any]],
        skip_llm: bool = False,
    ) -> str:
        if self.llm and not skip_llm:
            summary = self._summarize_with_llm(client_name, project_name, documents)
            if summary:
                return summary

        return self._summarize_with_rules(client_name, project_name, documents)

    def _infer_project(self, source_path: str, extracted: ExtractedProfileData) -> dict[str, Any]:
        context = extracted.project_context
        project_name = context.project_name or self._derive_name_from_path(source_path, extracted)
        project_code = context.project_code or self._derive_code(project_name, extracted)
        related_references = [
            value
            for value in [
                context.quote_number,
                context.purchase_order_number,
                context.access_reference,
                *context.related_references,
            ]
            if value
        ]
        project_key = self._slugify(project_code or project_name or Path(source_path).stem)

        return {
            "project_key": project_key,
            "project_name": project_name,
            "project_code": project_code,
            "quote_number": context.quote_number,
            "purchase_order_number": context.purchase_order_number,
            "access_reference": context.access_reference,
            "related_references": self._unique(related_references),
        }

    def _match_existing_project(self, inferred: dict[str, Any], documents: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not documents:
            return None

        project_key = inferred.get("project_key")
        project_name = self._normalize_name(inferred.get("project_name"))
        project_code = (inferred.get("project_code") or "").upper()
        references = {str(value).upper() for value in inferred.get("related_references", [])}

        best_match: dict[str, Any] | None = None
        best_score = 0
        for doc in documents:
            score = 0
            if project_key and doc.get("project_key") == project_key:
                score += 6
            if project_code and (doc.get("project_code") or "").upper() == project_code:
                score += 5
            if project_name and self._normalize_name(doc.get("project_name")) == project_name:
                score += 4
            existing_refs = {str(value).upper() for value in doc.get("related_references", [])}
            score += len(references & existing_refs) * 3
            if score > best_score:
                best_score = score
                best_match = doc
        return best_match if best_score >= 4 else None

    def _match_with_llm(
        self,
        text: str,
        extracted: ExtractedProfileData,
        inferred: dict[str, Any],
        documents: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        candidates = []
        seen: set[str] = set()
        for doc in documents:
            key = str(doc.get("project_key") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "project_key": key,
                    "project_name": doc.get("project_name"),
                    "project_code": doc.get("project_code"),
                    "related_references": doc.get("related_references", []),
                }
            )

        if not candidates:
            return None

        prompt = (
            "You associate incoming project documents with existing projects. "
            "Use project names, dates, quote numbers, PO numbers, access references, and client context. "
            "Return JSON with keys: match_project_key, match_project_name, match_project_code, confidence. "
            "If no candidate clearly matches, return null for the match fields.\n\n"
            f"Client: {extracted.client_name}\n"
            f"Inferred current project: {inferred}\n"
            f"Candidate projects: {candidates}\n\n"
            f"Document excerpt:\n{text[:4000]}"
        )
        try:
            result = self.llm.extract_structured(prompt)
        except Exception:
            return None

        match_key = str(result.get("match_project_key") or "").strip()
        if not match_key:
            return None
        for candidate in candidates:
            if candidate["project_key"] == match_key:
                return candidate
        return None

    def _summarize_with_llm(self, client_name: str, project_name: str, documents: list[dict[str, Any]]) -> str | None:
        digest = []
        for doc in documents[:6]:
            digest.append(
                {
                    "kind": doc.get("document_kind"),
                    "date": doc.get("report_date"),
                    "title": doc.get("title"),
                    "work_type": doc.get("report_type"),
                    "contacts": doc.get("contacts", []),
                    "refs": (doc.get("related_references") or [])[:5],
                    "amounts": (doc.get("currency_amounts") or [])[:4],
                    "excerpt": str(doc.get("excerpt") or "")[:220],
                }
            )
        prompt = (
            "Return STRICT JSON only with this schema: {\"summary\":\"...\"}. "
            "Write 4-5 concise sentences using only provided evidence. "
            "Order: plant/work area and scope; key findings/issues; recommendations (one full sentence minimum); budget/invoice/logistics status. "
            "Do not add unsupported details.\n\n"
            f"Client: {client_name}; Project: {project_name}; Evidence: {digest}"
        )
        try:
            result = self.llm.extract_structured(prompt)
        except Exception:
            return None
        summary = result.get("summary")
        if isinstance(summary, str) and summary.strip():
            clean = summary.strip()
            if len(clean) >= 120:
                return clean
        return None

    def _summarize_with_rules(self, client_name: str, project_name: str, documents: list[dict[str, Any]]) -> str:
        if not documents:
            return f"{project_name} has been identified, but there are no related documents stored yet."

        kinds = self._unique(str(doc.get("document_kind") or "unknown") for doc in documents)
        refs: list[str] = []
        contacts: list[str] = []
        authors: list[str] = []
        findings: list[str] = []
        recommendations: list[str] = []
        currency_amounts: list[str] = []
        for doc in documents:
            for ref in doc.get("related_references", []) or []:
                if ref not in refs:
                    refs.append(ref)
            for contact in doc.get("contacts", []) or []:
                if contact not in contacts:
                    contacts.append(contact)
            for author in doc.get("authors", []) or []:
                if author not in authors:
                    authors.append(author)
            for amount in doc.get("currency_amounts", []) or []:
                if amount not in currency_amounts:
                    currency_amounts.append(amount)

            excerpt = str(doc.get("excerpt") or "")
            findings.extend(self._extract_statement(excerpt, ["finding", "issue", "observation", "risk"]))
            recommendations.extend(
                self._extract_statement(
                    excerpt,
                    ["recommend", "action", "next step", "mitigation", "should", "priority", "required"],
                )
            )

        date_values = [str(doc.get("report_date") or "").strip() for doc in documents if doc.get("report_date")]
        earliest_date = min(date_values) if date_values else "undated"
        latest_date = max(date_values) if date_values else "undated"
        logistics_present = any(
            str(doc.get("document_kind")) in {"quote", "purchase_order", "access_request", "email_chain"}
            for doc in documents
        )
        invoice_present = any(str(doc.get("document_kind")) == "invoice" for doc in documents)
        report_types = self._unique(str(doc.get("report_type") or "").strip() for doc in documents if doc.get("report_type"))
        refs_text = ", ".join(refs[:4]) if refs else "no commercial references yet"
        participants = ", ".join((contacts or authors)[:4]) if (contacts or authors) else "project stakeholders are not named in metadata"
        work_scope = ", ".join(report_types[:3]) if report_types else ", ".join(kinds[:3])
        findings_text = "; ".join(findings[:2]) if findings else "No explicit issues or findings were extracted from the available documents."
        recommendations_text = "; ".join(recommendations[:2]) if recommendations else "Recommendations are not explicitly captured in the available metadata."
        commercial_bits: list[str] = []
        if logistics_present:
            commercial_bits.append("logistics and access coordination are documented")
        if invoice_present:
            commercial_bits.append("invoice records are present")
        if currency_amounts:
            commercial_bits.append(f"noted commercial values include {', '.join(currency_amounts[:3])}")
        if not commercial_bits:
            commercial_bits.append("commercial status is only partially evidenced in the stored documents")
        return (
            f"{project_name} covers {work_scope} work for {client_name} across {len(documents)} related documents dated between {earliest_date} and {latest_date}. "
            f"Key participants include {participants}, and the document set links the work through references such as {refs_text}. "
            f"Recommendations: {recommendations_text} "
            f"Issues and key findings: {findings_text} "
            f"The available records indicate that {', '.join(commercial_bits)}. "
            "Conclusions should be based on the documented findings and recommended actions above."
        )

    def _extract_statement(self, text: str, keywords: list[str]) -> list[str]:
        if not text.strip():
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result: list[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            snippet = sentence.strip()
            if len(snippet) < 24:
                continue
            lowered = snippet.lower()
            if not any(keyword in lowered for keyword in keywords):
                continue
            snippet = re.sub(r"\s+", " ", snippet)
            if snippet in seen:
                continue
            seen.add(snippet)
            result.append(snippet[:220])
        return result[:2]

    def _derive_name_from_path(self, source_path: str, extracted: ExtractedProfileData) -> str:
        stem = Path(source_path).stem.replace("_", " ")
        stem = re.sub(r"\breport\b", "", stem, flags=re.IGNORECASE)
        stem = re.sub(r"\bv\d+\b", "", stem, flags=re.IGNORECASE)
        if extracted.client_name:
            stem = re.sub(re.escape(extracted.client_name.replace("_", " ")), "", stem, flags=re.IGNORECASE)
        stem = re.sub(r"\b\d{2,4}\b", " ", stem)
        stem = re.sub(r"\s+", " ", stem).strip(" -")
        if extracted.insight.report_type:
            return stem.title() or str(extracted.insight.report_type).strip()
        return stem.title() or Path(source_path).stem.replace("_", " ").title()

    def _derive_code(self, project_name: str | None, extracted: ExtractedProfileData) -> str | None:
        if not project_name and not extracted.client_name:
            return None
        client = "".join(word[0] for word in str(extracted.client_name or "").split()[:3]).upper()
        words = [word[:3].upper() for word in str(project_name or "General Project").split()[:3]]
        year = str(extracted.additional_fields.get("report_date") or "")[:4]
        pieces = [piece for piece in [client, *words, year] if piece]
        return "-".join(pieces) or None

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
        return slug or "general-project"

    def _normalize_name(self, value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    def _unique(self, values: Any) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                result.append(text)
                seen.add(text)
        return result