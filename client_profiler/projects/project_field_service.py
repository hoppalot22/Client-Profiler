from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from client_profiler.embeddings import LocalEmbedder, VectorRetriever
from client_profiler.extraction import OllamaClient
from client_profiler.storage import SqliteStorage


ABSENT_VALUE = "Information not present"
FIELD_CONTEXT_BUDGETS: dict[str, tuple[int, int]] = {
    "title": (900, 360),
    "scope": (1200, 420),
    "participants": (1100, 420),
    "date": (1000, 380),
    "quoted": (900, 340),
    "invoice": (900, 340),
    "findings": (1400, 520),
    "recommendations": (1400, 520),
    "gaps": (1200, 460),
}

FIELD_PROFILES: dict[str, dict[str, Any]] = {
    "title": {
        "terms": ["title", "project", "assessment", "audit", "review"],
    },
    "scope": {
        "terms": ["scope", "work", "assessment", "audit", "review", "inspection"],
    },
    "participants": {
        "terms": ["participant", "contact", "author", "stakeholder", "team"],
    },
    "date": {
        "terms": ["date", "submitted", "issued", "report", "timeline", "milestone", "inspection", "approval"],
    },
    "quoted": {
        "terms": ["quote", "$", "price", "cost", "estimate"],
    },
    "invoice": {
        "terms": ["invoice", "$", "amount", "billing", "payment"],
    },
    "findings": {
        "terms": ["finding", "issue", "risk", "observation", "defect"],
    },
    "recommendations": {
        "terms": ["recommend", "action", "should", "mitigation", "next"],
    },
    "gaps": {
        "terms": ["missing", "unknown", "gap", "uncertain", "not provided"],
    },
}


class ProjectFieldService:
    def __init__(
        self,
        storage: SqliteStorage,
        embedder: LocalEmbedder,
        retriever: VectorRetriever,
        llm: OllamaClient | None,
        key_fields_path: Path,
        debug_enabled: bool = False,
        debug_log_path: Path | None = None,
    ) -> None:
        self.storage = storage
        self.embedder = embedder
        self.retriever = retriever
        self.llm = llm
        self.key_fields_path = key_fields_path
        self.debug_enabled = bool(debug_enabled)
        self.debug_log_path = debug_log_path or (key_fields_path.parent / "project_field_debug.jsonl")
        self._ensure_default_key_fields_doc()

    def field_definitions(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        text = self.key_fields_path.read_text(encoding="utf-8", errors="ignore")
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                key, label, prompt = parts[0], parts[1], "|".join(parts[2:]).strip()
            elif len(parts) == 2:
                key, prompt = parts
                label = key.replace("_", " ").title()
            else:
                key = parts[0]
                label = key.replace("_", " ").title()
                prompt = f"What is known about {label.lower()} for this project?"
            if not key or not prompt:
                continue
            rows.append({"key": key, "label": label or key, "prompt": prompt})
        return rows or self._default_key_fields()

    def project_field_values(self, client_name: str, project_key: str) -> dict[str, dict[str, str]]:
        payload = self.storage.get_project_key_fields(client_name, project_key) or {}
        fields = payload.get("fields", {}) if isinstance(payload, dict) else {}
        return fields if isinstance(fields, dict) else {}

    def generate_field(
        self,
        client_name: str,
        project_key: str,
        project_name: str,
        field_key: str,
        field_prompt: str,
        documents: list[dict[str, Any]],
        existing_fields: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        if self.llm is None:
            return {"ok": False, "error_code": "llm_not_configured", "error_message": "LLM provider is not configured."}
        if not documents:
            return {"ok": False, "error_code": "no_project_documents", "error_message": "No project documents found."}

        profile = self._field_profile(field_key)
        max_words_raw = profile.get("max_words")
        max_words = int(max_words_raw) if isinstance(max_words_raw, (int, float)) else None
        local_evidence = self._build_local_field_evidence(field_key, field_prompt, documents)

        # Only use broad RAG when local project evidence is thin.
        project_hints = self._project_hints(project_key, documents)
        rag_hits: list[dict[str, Any]] = []
        if len(local_evidence) < 2:
            query_text = self._build_query_text(project_name, field_key, field_prompt, documents)
            query_embedding = self.embedder.embed_text(query_text)
            rag_hits = self._search_field_rag(
                field_key=field_key,
                query_embedding=query_embedding,
                client_name=client_name,
                project_hints=project_hints,
            )
        rag_context = self._build_rag_context(field_key, rag_hits)
        local_evidence, rag_context = self._fit_context_budget(field_key, local_evidence, rag_context)

        context_fields = {
            key: str((value or {}).get("value") or "").strip()
            for key, value in (existing_fields or {}).items()
            if str((value or {}).get("value") or "").strip()
        }
        compact_fields = self._compact_known_fields(context_fields, field_key)

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
            f"Field-specific rules: {self._field_specific_guardrails(field_key)}\n\n"
            f"Project: {project_name}\n"
            f"Field: {field_key}\n"
            f"Instruction: {field_prompt}\n"
            f"Known fields: {compact_fields}\n"
            f"Project evidence:\n{self._format_lines(local_evidence)}\n"
            f"Related evidence:\n{self._format_lines(rag_context)}"
        )

        debug_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_name": client_name,
            "project_key": project_key,
            "project_name": project_name,
            "field_key": field_key,
            "prompt_chars": len(prompt),
            "local_evidence_count": len(local_evidence),
            "rag_hit_count": len(rag_hits),
            "rag_context_chars": len(rag_context),
            "known_fields_count": len(compact_fields.split(";")) if compact_fields != "none" else 0,
        }

        result = self.llm.extract_structured(prompt)
        if not isinstance(result, dict) or not result:
            code = self._llm_error_code()
            self._write_debug_log(
                {
                    **debug_payload,
                    "ok": False,
                    "error_code": code,
                    "error_message": self._llm_error_message(code),
                }
            )
            return {"ok": False, "error_code": code, "error_message": self._llm_error_message(code)}

        is_present = bool(result.get("is_present"))
        value = self._short_value(str(result.get("value") or "").strip(), max_words)
        evidence = str(result.get("evidence") or "").strip()
        if (not is_present) or (not value):
            stored = self.storage.upsert_project_key_field(
                client_name=client_name,
                project_key=project_key,
                project_name=project_name,
                field_key=field_key,
                value=ABSENT_VALUE,
                status="absent",
                evidence=evidence,
                method="ai",
            )
            self._refresh_project_fields_embedding(client_name, project_key, project_name)
            self._write_debug_log({**debug_payload, "ok": True, "status": "absent", "value_words": 0})
            return {"ok": True, **stored}

        stored = self.storage.upsert_project_key_field(
            client_name=client_name,
            project_key=project_key,
            project_name=project_name,
            field_key=field_key,
            value=value,
            status="filled",
            evidence=evidence,
            method="ai",
        )
        self._refresh_project_fields_embedding(client_name, project_key, project_name)
        self._write_debug_log(
            {
                **debug_payload,
                "ok": True,
                "status": "filled",
                "value_words": len(value.split()),
            }
        )
        return {"ok": True, **stored}

    def _refresh_project_fields_embedding(self, client_name: str, project_key: str, project_name: str) -> None:
        fields = self.project_field_values(client_name, project_key)
        if not isinstance(fields, dict) or not fields:
            return

        ordered_keys = [row["key"] for row in self.field_definitions()]
        lines = [
            f"Client: {client_name}",
            f"Project: {project_name}",
            f"Project key: {project_key}",
            "Key fields:",
        ]
        for key in ordered_keys:
            payload = fields.get(key, {}) if isinstance(fields.get(key, {}), dict) else {}
            value = str(payload.get("value") or ABSENT_VALUE).strip() or ABSENT_VALUE
            status = str(payload.get("status") or "unknown").strip()
            method = str(payload.get("method") or "unknown").strip()
            lines.append(f"- {key}: {value} (status={status}, method={method})")

        chunk_text = "\n".join(lines)
        embedding = self.embedder.embed_text(chunk_text)
        source_document = f"__project_fields__/{client_name}/{project_key}"
        metadata = {
            "client_name": client_name,
            "project_key": project_key,
            "project_name": project_name,
            "document_kind": "project_fields",
            "is_client_related": True,
            "source_type": "lazy_llm_fields",
        }
        self.storage.upsert_vector(
            source_document=source_document,
            chunk_text=chunk_text,
            embedding=embedding,
            metadata=metadata,
            client_name=client_name,
        )

    def _build_query_text(self, project_name: str, field_key: str, field_prompt: str, documents: list[dict[str, Any]]) -> str:
        profile = self._field_profile(field_key)
        parts: list[str] = [project_name, field_key, field_prompt, " ".join(profile["terms"])]
        for doc in documents[:4]:
            for key in ["title", "report_type", "project_code"]:
                value = str(doc.get(key) or "").strip()
                if value:
                    parts.append(value)
        return " ".join(dict.fromkeys(parts))

    def _search_field_rag(
        self,
        field_key: str,
        query_embedding: list[float],
        client_name: str,
        project_hints: dict[str, Any],
    ) -> list[dict[str, Any]]:
        profile = self._field_profile(field_key)
        terms = [str(t).lower() for t in profile["terms"]]
        project_sources = sorted(str(p).strip() for p in (project_hints.get("source_documents") or []) if str(p).strip())
        metadata_filters = self._metadata_filters_for_field(field_key)

        hits = self.retriever.search(
            query_embedding,
            top_k=10,
            client_name=client_name,
            source_documents=project_sources or None,
            metadata_filters=metadata_filters,
            query_text=" ".join(terms),
            hybrid_alpha=0.74,
            candidate_pool=70,
            use_mmr=True,
            mmr_lambda=0.72,
        )
        if len(hits) < 3:
            client_hits = self.retriever.search(
                query_embedding,
                top_k=14,
                client_name=client_name,
                metadata_filters=metadata_filters,
                query_text=" ".join(terms),
                hybrid_alpha=0.74,
                candidate_pool=90,
                use_mmr=True,
                mmr_lambda=0.72,
            )
            seen_sources = {str(hit.get("source_document") or "") for hit in hits}
            for hit in client_hits:
                src = str(hit.get("source_document") or "")
                if src in seen_sources:
                    continue
                hits.append(hit)
                seen_sources.add(src)

        scored: list[tuple[float, dict[str, Any]]] = []
        hint_refs = {str(v).lower() for v in project_hints.get("references", set())}
        hint_code = str(project_hints.get("project_code") or "").strip().lower()
        hint_key = str(project_hints.get("project_key") or "").strip().lower()
        hint_years = {str(v) for v in project_hints.get("years", set())}
        hit_dates = [self._to_datetime((hit.get("metadata") or {}).get("report_date")) for hit in hits]
        valid_dates = [dt for dt in hit_dates if dt is not None]
        newest = max(valid_dates) if valid_dates else None
        oldest = min(valid_dates) if valid_dates else None

        for hit in hits:
            text = str(hit.get("chunk_text") or "").lower()
            keyword_score = sum(1 for term in terms if term in text)
            if keyword_score == 0:
                continue
            score = float(hit.get("score") or 0.0) + (0.04 * keyword_score)

            metadata_raw = hit.get("metadata")
            metadata: dict[str, Any] = metadata_raw if isinstance(metadata_raw, dict) else {}
            if hint_key and str(metadata.get("project_key") or "").strip().lower() == hint_key:
                score += 0.12
            if hint_code and str(metadata.get("project_code") or "").strip().lower() == hint_code:
                score += 0.08

            metadata_refs = metadata.get("related_references", [])
            if isinstance(metadata_refs, list):
                refs = {str(v).strip().lower() for v in metadata_refs if str(v).strip()}
                if refs & hint_refs:
                    score += 0.08

            if field_key == "date":
                report_date = str(metadata.get("report_date") or "")
                if any(year in report_date for year in hint_years):
                    score += 0.05

            temporal = self._temporal_boost(field_key, self._to_datetime(metadata.get("report_date")), oldest, newest)
            score += temporal

            scored.append((score, hit))
        scored.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("source_document") or ""),
                str(item[1].get("chunk_text") or ""),
            )
        )
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _, hit in scored:
            src = str(hit.get("source_document") or "")
            if src in seen:
                continue
            seen.add(src)
            out.append(hit)
            if len(out) >= 3:
                break
        return out

    def _metadata_filters_for_field(self, field_key: str) -> dict[str, Any]:
        filters: dict[str, Any] = {"is_client_related": True}
        if field_key == "date":
            filters["document_kind"] = ["report", "email"]
        elif field_key == "quoted":
            filters["document_kind"] = ["quote", "proposal", "email", "report"]
        elif field_key == "invoice":
            filters["document_kind"] = ["invoice", "email", "report"]
        return filters

    def _fit_context_budget(self, field_key: str, local_evidence: list[str], rag_context: str) -> tuple[list[str], str]:
        local_budget, rag_budget = FIELD_CONTEXT_BUDGETS.get(field_key, (760, 280))

        kept_local: list[str] = []
        used = 0
        for row in local_evidence:
            row_len = len(str(row))
            if kept_local and (used + row_len) > local_budget:
                break
            kept_local.append(row)
            used += row_len
            if used >= local_budget:
                break

        rag_lines = [line for line in str(rag_context or "").splitlines() if line.strip()]
        kept_rag: list[str] = []
        rag_used = 0
        for line in rag_lines:
            line_len = len(line)
            if kept_rag and (rag_used + line_len) > rag_budget:
                break
            kept_rag.append(line)
            rag_used += line_len
            if rag_used >= rag_budget:
                break

        return kept_local, "\n".join(kept_rag)

    def _temporal_boost(
        self,
        field_key: str,
        current: datetime | None,
        oldest: datetime | None,
        newest: datetime | None,
    ) -> float:
        if current is None or oldest is None or newest is None:
            return 0.0
        span = max(1.0, (newest - oldest).total_seconds())
        pos = (current - oldest).total_seconds() / span

        if field_key in {"date", "quoted", "invoice", "participants"}:
            return 0.05 * pos
        if field_key in {"title", "scope"}:
            return 0.05 * (1.0 - pos)
        return 0.0

    def _to_datetime(self, value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _project_hints(self, project_key: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        sources: set[str] = set()
        refs: set[str] = set()
        years: set[str] = set()
        project_code = ""

        for doc in documents:
            src = str(doc.get("source_path") or "").strip()
            if src:
                sources.add(src)

            if not project_code:
                project_code = str(doc.get("project_code") or "").strip()

            for value in doc.get("related_references") or []:
                text = str(value).strip()
                if text:
                    refs.add(text)

            for value in [doc.get("report_date"), doc.get("ingested_at")]:
                match = re.search(r"\b(19\d{2}|20\d{2})\b", str(value or ""))
                if match:
                    years.add(match.group(1))

        return {
            "project_key": str(project_key or "").strip(),
            "project_code": project_code,
            "references": refs,
            "years": years,
            "source_documents": sources,
        }

    def _build_rag_context(self, field_key: str, rag_hits: list[dict[str, Any]]) -> str:
        if not rag_hits:
            return ""
        terms = [str(t).lower() for t in self._field_profile(field_key)["terms"]]
        lines: list[str] = []
        for hit in rag_hits[:3]:
            source = str(hit.get("source_document") or "")
            text = str(hit.get("chunk_text") or "")
            focused = self._extract_focus_snippet(text, terms, max_chars=180)
            if not focused:
                continue
            label = f"[{hit.get('client_name') or 'unknown'}|{Path(source).name}]"
            lines.append(f"- {label} {focused}")
        return "\n".join(lines)

    def _build_local_field_evidence(self, field_key: str, field_prompt: str, documents: list[dict[str, Any]]) -> list[str]:
        profile = self._field_profile(field_key)
        terms = [str(t).lower() for t in profile["terms"]]
        rows: list[tuple[int, str]] = []
        for doc in documents[:6]:
            title = str(doc.get("title") or doc.get("document_name") or "").strip()
            excerpt = str(doc.get("excerpt") or "")
            refs = ",".join((doc.get("related_references") or [])[:3])
            date = str(doc.get("report_date") or doc.get("ingested_at") or "")
            kind = str(doc.get("document_kind") or "")
            snippet = self._extract_focus_snippet(excerpt, terms, max_chars=150)
            meta = f"[{kind} {date} {title}]"

            match_score = 0
            text_for_score = f"{title} {excerpt}".lower()
            for term in terms:
                if term in text_for_score:
                    match_score += 1
            if field_key == "recommendations" and not self._contains_recommendation_signal(text_for_score):
                match_score -= 2
            if field_key == "findings" and not self._contains_finding_signal(text_for_score):
                match_score -= 2
            if field_key in {"quoted", "invoice"} and doc.get("currency_amounts"):
                amounts = ",".join((doc.get("currency_amounts") or [])[:2])
                snippet = f"amounts: {amounts}; {snippet}".strip("; ")
                match_score += 2
            if field_key == "participants" and (doc.get("contacts") or doc.get("authors")):
                people = ",".join((doc.get("contacts") or doc.get("authors") or [])[:4])
                snippet = f"people: {people}; {snippet}".strip("; ")
                match_score += 2
            if field_key in {"recommendations", "findings"} and self._is_commercial_only_text(text_for_score):
                match_score -= 2

            if match_score <= 0:
                continue
            evidence = f"- {meta} refs:{refs} {snippet}".strip()
            rows.append((match_score, evidence))

        rows.sort(key=lambda item: (-item[0], item[1]))
        return [row for _, row in rows[:4]]

    def _extract_focus_snippet(self, text: str, terms: list[str], max_chars: int = 160) -> str:
        raw = re.sub(r"\s+", " ", str(text or "").strip())
        if not raw:
            return ""
        lower = raw.lower()
        candidate_positions: list[int] = []
        for term in terms:
            for match in re.finditer(re.escape(term), lower):
                candidate_positions.append(match.start())

        if not candidate_positions:
            return raw[:max_chars]

        signal_terms = [
            "finding",
            "issue",
            "risk",
            "defect",
            "recommend",
            "action",
            "should",
            "repair",
            "replace",
            "monitor",
            "mitigate",
            "compliance",
        ]
        best_idx = candidate_positions[0]
        best_score = -1
        for idx in candidate_positions:
            start = max(0, idx - 60)
            end = min(len(raw), start + max_chars)
            window = lower[start:end]
            score = sum(1 for cue in signal_terms if cue in window)
            if score > best_score:
                best_score = score
                best_idx = idx

        start = max(0, best_idx - 40)
        if start > 0:
            prev_space = raw.rfind(" ", 0, start)
            if prev_space > 0:
                start = prev_space + 1
        end = min(len(raw), start + max_chars)
        if end < len(raw):
            next_space = raw.find(" ", end)
            if next_space > 0:
                end = next_space
        return raw[start:end].strip()

    def _compact_known_fields(self, context_fields: dict[str, str], current_field: str) -> str:
        if not context_fields:
            return "none"
        priority = [
            "title",
            "scope",
            "date",
            "quoted",
            "invoice",
            "participants",
            "findings",
            "recommendations",
            "gaps",
        ]
        ordered = [key for key in priority if key in context_fields and key != current_field]
        if len(ordered) < 3:
            ordered.extend([k for k in context_fields.keys() if k not in ordered and k != current_field])
        picked = ordered[:3]
        return "; ".join(f"{k}={self._short_value(context_fields[k], 10)}" for k in picked) or "none"

    def _format_lines(self, value: list[str] | str) -> str:
        if isinstance(value, list):
            return "\n".join(value) if value else "- none"
        text = str(value or "").strip()
        return text if text else "- none"

    def _field_specific_guardrails(self, field_key: str) -> str:
        if field_key == "date":
            return (
                "Return all key project dates as concise labeled entries in the format `label: YYYY-MM-DD` when possible, "
                "separated by `; `. Include milestones/report/inspection/approval dates when explicitly evidenced. "
                "Do not omit key dates that are explicitly present."
            )
        if field_key == "recommendations":
            return (
                "Return only explicit technical actions/recommendations from report narrative. "
                "Reject answers that are only quote/invoice/price/PO status without an action."
            )
        if field_key == "findings":
            return (
                "Return observed conditions, defects, risks, or compliance issues. "
                "Do not return recommendations or commercial values unless tied to a finding."
            )
        if field_key == "quoted":
            return "Return only quoted/estimated amount; do not return invoice totals or recommendation text."
        if field_key == "invoice":
            return "Return only invoiced/billed amount; do not return quote estimates or recommendation text."
        return "Prioritize direct report evidence for this field and avoid cross-field substitution."

    def _contains_recommendation_signal(self, text: str) -> bool:
        cues = [
            "recommend",
            "should",
            "action",
            "mitigate",
            "repair",
            "replace",
            "monitor",
            "follow-up",
            "implement",
        ]
        return any(cue in text for cue in cues)

    def _contains_finding_signal(self, text: str) -> bool:
        cues = [
            "finding",
            "issue",
            "risk",
            "defect",
            "non-compliance",
            "observed",
            "condition",
            "failure",
            "gap",
        ]
        return any(cue in text for cue in cues)

    def _is_commercial_only_text(self, text: str) -> bool:
        commercial_cues = ["quote", "quoted", "invoice", "price", "cost", "purchase order", "po", "billing"]
        technical_cues = ["recommend", "issue", "risk", "defect", "repair", "replace", "monitor", "compliance"]
        has_commercial = any(cue in text for cue in commercial_cues)
        has_technical = any(cue in text for cue in technical_cues)
        return has_commercial and not has_technical

    def _field_profile(self, field_key: str) -> dict[str, Any]:
        return FIELD_PROFILES.get(field_key, {"terms": [field_key], "max_words": 16})

    def _short_value(self, value: str, max_words: int | None) -> str:
        words = str(value or "").split()
        if max_words is None:
            return " ".join(words)
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words]).strip()

    def _write_debug_log(self, payload: dict[str, Any]) -> None:
        if not self.debug_enabled:
            return
        try:
            self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.debug_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except OSError:
            pass

    def _llm_error_code(self) -> str:
        raw = str(getattr(self.llm, "last_error", "") or "").strip()
        return raw or "no_response"

    def _llm_error_message(self, code: str) -> str:
        mapping = {
            "request_timed_out": "Request to local model timed out.",
            "empty_model_response": "Model returned an empty response.",
            "invalid_json_response": "Model returned invalid JSON.",
            "client_disabled_after_previous_failure": "Model client is disabled after a previous request failure.",
            "request_error": "Could not reach local model service.",
            "no_response": "Model returned no response.",
        }
        if code.startswith("http_error_"):
            return f"Local model service returned HTTP error ({code.replace('http_error_', '')})."
        return mapping.get(code, f"Model request failed ({code}).")

    def _ensure_default_key_fields_doc(self) -> None:
        self.key_fields_path.parent.mkdir(parents=True, exist_ok=True)
        if self.key_fields_path.exists():
            return
        lines = [
            "# Project key fields",
            "# Format: key|Label|Prompt for extraction",
            "scope|Project Scope|What is this project about and what is the scope of work?",
            "participants|Participants|Who are the key people, teams, or stakeholders involved?",
            "timeline|Timeline|What dates, milestones, or sequencing are known?",
            "commercial|Commercial|What budget, quote, PO, invoice, or logistics details are known?",
            "findings|Findings|What key findings, issues, or risks are documented?",
            "recommendations|Recommendations|What actions or recommendations are documented?",
            "gaps|Known Gaps|What important information is missing or uncertain?",
        ]
        self.key_fields_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _default_key_fields(self) -> list[dict[str, str]]:
        return [
            {"key": "scope", "label": "Project Scope", "prompt": "What is this project about and what is the scope of work?"},
            {"key": "participants", "label": "Participants", "prompt": "Who are the key people, teams, or stakeholders involved?"},
            {"key": "timeline", "label": "Timeline", "prompt": "What dates, milestones, or sequencing are known?"},
            {"key": "commercial", "label": "Commercial", "prompt": "What budget, quote, PO, invoice, or logistics details are known?"},
            {"key": "findings", "label": "Findings", "prompt": "What key findings, issues, or risks are documented?"},
            {"key": "recommendations", "label": "Recommendations", "prompt": "What actions or recommendations are documented?"},
            {"key": "gaps", "label": "Known Gaps", "prompt": "What important information is missing or uncertain?"},
        ]
