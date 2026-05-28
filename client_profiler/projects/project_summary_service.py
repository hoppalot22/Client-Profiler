from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from client_profiler.embeddings import LocalEmbedder, VectorRetriever
from client_profiler.extraction import OllamaClient
from client_profiler.storage import SqliteStorage


ABSENT_ANSWER = "Information not present in stored documents."
SUMMARY_RAG_CONTEXT_BUDGET = 2200


class ProjectSummaryService:
    def __init__(
        self,
        storage: SqliteStorage,
        embedder: LocalEmbedder,
        retriever: VectorRetriever,
        llm: OllamaClient | None,
        questionnaire_path: Path,
    ) -> None:
        self.storage = storage
        self.embedder = embedder
        self.retriever = retriever
        self.llm = llm
        self.questionnaire_path = questionnaire_path
        self.failure_log_path = questionnaire_path.parent / "project_summary_failures.jsonl"
        self.question_retry_count = 2
        self.summary_retry_count = 1
        self.batch_size = 3
        self._ensure_default_questionnaire()

    def generate_and_store(
        self,
        client_name: str,
        project_key: str,
        project_name: str,
        documents: list[dict[str, Any]],
        *,
        strategy: str = "batched",
        store: bool = True,
    ) -> dict[str, Any]:
        if self.llm is None:
            return self._failure(
                client_name,
                project_key,
                project_name,
                documents,
                "llm_not_configured",
                "LLM provider is not configured.",
                strategy=strategy,
            )

        if not documents:
            return self._failure(
                client_name,
                project_key,
                project_name,
                documents,
                "no_project_documents",
                "No documents are linked to this project yet.",
                strategy=strategy,
            )

        query_text = self._build_query_text(project_name, documents)
        query_embedding = self.embedder.embed_text(query_text)
        source_documents = [str(doc.get("source_path") or "").strip() for doc in documents if str(doc.get("source_path") or "").strip()]
        rag_hits = self.retriever.search(
            query_embedding,
            top_k=12,
            client_name=client_name,
            source_documents=source_documents or None,
            query_text=query_text,
            hybrid_alpha=0.8,
            use_mmr=True,
            mmr_lambda=0.75,
            candidate_pool=90,
        )
        if len(rag_hits) < 8:
            extra_hits = self.retriever.search(
                query_embedding,
                top_k=18,
                client_name=client_name,
                query_text=query_text,
                hybrid_alpha=0.8,
                use_mmr=True,
                mmr_lambda=0.75,
                candidate_pool=120,
            )
            seen_sources = {str(hit.get("source_document") or "") for hit in rag_hits}
            for hit in extra_hits:
                src = str(hit.get("source_document") or "")
                if src in seen_sources:
                    continue
                rag_hits.append(hit)
                seen_sources.add(src)

        rag_context = self._build_rag_context(rag_hits)
        digest = self._build_project_digest(documents)
        questions = self._load_questionnaire()
        if not questions:
            return self._failure(
                client_name,
                project_key,
                project_name,
                documents,
                "questionnaire_empty",
                "Questionnaire file has no usable questions.",
                strategy=strategy,
                query_text=query_text,
                rag_hits=rag_hits,
                digest=digest,
            )

        question_result = self._ask_questions(
            client_name,
            project_name,
            digest,
            rag_context,
            questions,
            strategy=strategy,
        )
        answers = question_result["answers"]
        question_runs = question_result["question_runs"]

        summary_result = self._build_summary(client_name, project_name, digest, rag_context, answers)
        if not summary_result["ok"]:
            return self._failure(
                client_name,
                project_key,
                project_name,
                documents,
                str(summary_result.get("error_code") or "summary_failed"),
                str(summary_result.get("error_message") or "Summary generation failed."),
                strategy=strategy,
                query_text=query_text,
                rag_hits=rag_hits,
                digest=digest,
                question_runs=question_runs,
                summary_attempt=summary_result,
            )
        summary = summary_result["summary"]

        updated_at = None
        if store:
            summary_method = f"ai_rag_questionnaire_{strategy}_v1"
            self.storage.upsert_project_summary(
                client_name=client_name,
                project_key=project_key,
                project_name=project_name,
                summary_text=summary,
                summary_method=summary_method,
                questionnaire_answers=answers,
            )
            self.refresh_summary_embedding(
                client_name=client_name,
                project_key=project_key,
                project_name=project_name,
                summary_text=summary,
                summary_method=summary_method,
                questionnaire_answers=answers,
            )
            stored = self.storage.get_project_summary(client_name, project_key) or {}
            updated_at = stored.get("updated_at")

        return {
            "ok": True,
            "strategy": strategy,
            "summary": summary,
            "answers": answers,
            "question_runs": question_runs,
            "updated_at": updated_at,
            "summary_attempt": summary_result,
        }

    def _build_query_text(self, project_name: str, documents: list[dict[str, Any]]) -> str:
        parts: list[str] = [project_name]
        for doc in documents[:10]:
            for key in ["report_type", "title", "document_name", "project_code"]:
                value = str(doc.get(key) or "").strip()
                if value:
                    parts.append(value)
            for ref in (doc.get("related_references") or [])[:5]:
                value = str(ref).strip()
                if value:
                    parts.append(value)
        deduped = list(dict.fromkeys(parts))
        return " ".join(deduped)

    def _build_rag_context(self, rag_hits: list[dict[str, Any]]) -> str:
        seen_sources: set[str] = set()
        lines: list[str] = []
        used = 0
        for hit in rag_hits:
            source = str(hit.get("source_document") or "")
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)
            label = f"[{hit.get('client_name') or 'unknown client'} | {Path(source).name}]"
            snippet = str(hit.get("chunk_text") or "")[:420]
            candidate = f"{label}\n{snippet}"
            if lines and (used + len(candidate)) > SUMMARY_RAG_CONTEXT_BUDGET:
                break
            lines.append(candidate)
            used += len(candidate)
            if used >= SUMMARY_RAG_CONTEXT_BUDGET:
                break
            if len(lines) >= 12:
                break
        return "\n\n".join(lines)

    def _build_project_digest(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        digest: list[dict[str, Any]] = []
        for doc in documents[:8]:
            digest.append(
                {
                    "kind": doc.get("document_kind"),
                    "date": doc.get("report_date") or doc.get("ingested_at"),
                    "title": doc.get("title") or doc.get("document_name"),
                    "work_type": doc.get("report_type"),
                    "contacts": (doc.get("contacts") or [])[:5],
                    "authors": (doc.get("authors") or [])[:4],
                    "refs": (doc.get("related_references") or [])[:6],
                    "amounts": (doc.get("currency_amounts") or [])[:5],
                    "excerpt": str(doc.get("excerpt") or "")[:220],
                }
            )
        return digest

    def _ask_questions(
        self,
        client_name: str,
        project_name: str,
        digest: list[dict[str, Any]],
        rag_context: str,
        questions: list[tuple[str, str]],
        *,
        strategy: str,
    ) -> dict[str, Any]:
        if self.llm is None:
            return {"answers": {}, "question_runs": []}

        if strategy == "non_batched":
            return self._ask_questions_non_batched(client_name, project_name, digest, rag_context, questions)
        return self._ask_questions_batched(client_name, project_name, digest, rag_context, questions)

    def _ask_questions_non_batched(
        self,
        client_name: str,
        project_name: str,
        digest: list[dict[str, Any]],
        rag_context: str,
        questions: list[tuple[str, str]],
    ) -> dict[str, Any]:
        answers: dict[str, str] = {}
        question_runs: list[dict[str, Any]] = []

        for key, question in questions:
            prompt = self._single_question_prompt(client_name, project_name, digest, rag_context, key, question)
            result, meta = self._extract_with_retry(prompt, retry_count=self.question_retry_count)
            run: dict[str, Any] = {
                "key": key,
                "question": question,
                "attempts": meta["attempts"],
                "last_error_code": meta["last_error_code"],
                "last_error_message": meta["last_error_message"],
                "mode": "single",
            }
            normalized = self._normalize_answer_record(result)
            run.update(normalized)
            answers[key] = normalized["normalized_answer"]
            question_runs.append(run)

        return {"answers": answers, "question_runs": question_runs}

    def _ask_questions_batched(
        self,
        client_name: str,
        project_name: str,
        digest: list[dict[str, Any]],
        rag_context: str,
        questions: list[tuple[str, str]],
    ) -> dict[str, Any]:
        answers: dict[str, str] = {}
        question_runs: list[dict[str, Any]] = []

        for chunk in self._chunk_questions(questions, self.batch_size):
            prompt = self._batch_questions_prompt(client_name, project_name, digest, rag_context, chunk)
            result, meta = self._extract_with_retry(prompt, retry_count=self.question_retry_count)
            records = self._normalize_batch_answer_record(result)
            by_key = {item["key"]: item for item in records}

            for key, question in chunk:
                item = by_key.get(key, {"key": key, "status": "missing_in_batch", "normalized_answer": ABSENT_ANSWER})
                run = {
                    "key": key,
                    "question": question,
                    "attempts": meta["attempts"],
                    "last_error_code": meta["last_error_code"],
                    "last_error_message": meta["last_error_message"],
                    "mode": "batch",
                }
                run.update(item)
                answers[key] = str(item.get("normalized_answer") or ABSENT_ANSWER)
                question_runs.append(run)

        return {"answers": answers, "question_runs": question_runs}

    def _single_question_prompt(
        self,
        client_name: str,
        project_name: str,
        digest: list[dict[str, Any]],
        rag_context: str,
        key: str,
        question: str,
    ) -> str:
        return (
            'Return STRICT JSON only with this schema: {"is_present":true|false,"evidence":"...","answer":"..."}. '
            "You MUST follow this process: "
            "(1) decide if the required information is explicitly present and related to THIS project, "
            "(2) extract brief evidence quote fragments, "
            "(3) answer only if present. "
            "If missing, unrelated, or ambiguous set is_present=false and answer exactly: "
            f"{ABSENT_ANSWER}\n\n"
            f"Client: {client_name}; Project: {project_name}\n"
            f"Question key: {key}\n"
            f"Question: {question}\n\n"
            f"Project digest: {digest}\n\n"
            f"Related context from similar work: {rag_context}"
        )

    def _batch_questions_prompt(
        self,
        client_name: str,
        project_name: str,
        digest: list[dict[str, Any]],
        rag_context: str,
        chunk: list[tuple[str, str]],
    ) -> str:
        question_rows = [{"key": key, "question": question} for key, question in chunk]
        return (
            'Return STRICT JSON only with this schema: {"answers":{"<key>":{"is_present":true|false,"evidence":"...","answer":"..."}}}. '
            "For EACH question key, decide if information is explicitly present and related to THIS project. "
            "If missing, unrelated, or ambiguous set is_present=false and answer exactly: "
            f"{ABSENT_ANSWER}. "
            "Do not skip any keys.\n\n"
            f"Client: {client_name}; Project: {project_name}\n"
            f"Questions: {question_rows}\n\n"
            f"Project digest: {digest}\n\n"
            f"Related context from similar work: {rag_context}"
        )

    def _normalize_answer_record(self, result: dict[str, Any] | Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {
                "status": "non_dict_response",
                "raw_response": result,
                "normalized_answer": ABSENT_ANSWER,
            }

        is_present = bool(result.get("is_present"))
        evidence = str(result.get("evidence") or "").strip()
        answer = str(result.get("answer") or "").strip()
        if (not is_present) or (not answer) or self._looks_absent(answer):
            return {
                "status": "absent_or_rejected",
                "raw_response": result,
                "is_present": is_present,
                "evidence": evidence,
                "normalized_answer": ABSENT_ANSWER,
            }
        if len(evidence) < 8:
            return {
                "status": "insufficient_evidence",
                "raw_response": result,
                "is_present": is_present,
                "evidence": evidence,
                "normalized_answer": ABSENT_ANSWER,
            }
        return {
            "status": "accepted",
            "raw_response": result,
            "is_present": is_present,
            "evidence": evidence,
            "normalized_answer": answer,
        }

    def _normalize_batch_answer_record(self, result: dict[str, Any] | Any) -> list[dict[str, Any]]:
        if not isinstance(result, dict):
            return []
        answers_node = result.get("answers")
        if not isinstance(answers_node, dict):
            return []

        normalized: list[dict[str, Any]] = []
        for key, payload in answers_node.items():
            item = self._normalize_answer_record(payload)
            item["key"] = str(key)
            normalized.append(item)
        return normalized

    def _extract_with_retry(self, prompt: str, *, retry_count: int) -> tuple[dict[str, Any] | Any, dict[str, Any]]:
        attempts = 0
        last_code = ""
        last_message = ""

        for attempt in range(retry_count + 1):
            attempts += 1
            try:
                result = self.llm.extract_structured(prompt) if self.llm else {}
            except Exception:
                result = {}

            if isinstance(result, dict) and result:
                return result, {
                    "attempts": attempts,
                    "last_error_code": "",
                    "last_error_message": "",
                }

            last_code = self._llm_error_code()
            last_message = self._llm_error_message()
            if attempt < retry_count and self._is_retryable_error(last_code):
                continue
            break

        return {}, {
            "attempts": attempts,
            "last_error_code": last_code,
            "last_error_message": last_message,
        }

    def _is_retryable_error(self, code: str) -> bool:
        if code in {"request_timed_out", "request_error", "empty_model_response", "no_response"}:
            return True
        if code.startswith("http_error_"):
            status = code.replace("http_error_", "")
            return status.startswith("5")
        return False

    def _build_summary(
        self,
        client_name: str,
        project_name: str,
        digest: list[dict[str, Any]],
        rag_context: str,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        if self.llm is None:
            return {"ok": False, "error_code": "llm_not_configured", "error_message": "LLM provider is not configured."}

        prompt = (
            'Return STRICT JSON only with this schema: {"summary":"..."}. '
            "Write 8-11 concise sentences. Use the Q&A answers as the primary structure, "
            "and only add details that are supported by evidence. "
            "Do not infer missing facts. If a section has no evidence, acknowledge missing data instead of guessing. "
            "The summary MUST touch each project key-field area with a little context: "
            "title/theme, scope, participants, date/timeline, quoted/commercial, invoice/commercial status, "
            "findings, recommendations, and gaps/unknowns. "
            "Prefer explicit phrasing labels such as 'Scope:', 'Participants:', 'Timeline:', 'Commercial:', "
            "'Findings:', 'Recommendations:', and 'Gaps:' so each area is visibly covered.\n\n"
            f"Client: {client_name}; Project: {project_name}\n"
            f"Questionnaire answers: {answers}\n\n"
            f"Project digest: {digest}\n\n"
            f"Related context from similar work: {rag_context}"
        )

        result, meta = self._extract_with_retry(prompt, retry_count=self.summary_retry_count)

        if not isinstance(result, dict) or not result:
            return {
                "ok": False,
                "error_code": meta.get("last_error_code") or self._llm_error_code(),
                "error_message": meta.get("last_error_message") or self._llm_error_message(),
                "attempts": meta.get("attempts", 1),
            }

        summary = result.get("summary")
        if not isinstance(summary, str):
            return {
                "ok": False,
                "error_code": "summary_field_missing",
                "error_message": "Model response did not include a summary field.",
                "raw_response": result,
                "attempts": meta.get("attempts", 1),
            }

        clean = summary.strip()
        clean = self._ensure_key_field_coverage(clean, project_name, answers)
        if len(clean) < 60:
            return {
                "ok": False,
                "error_code": "summary_too_short",
                "error_message": f"Generated summary was too short ({len(clean)} chars) even after enrichment.",
                "raw_response": result,
                "summary_length": len(clean),
                "attempts": meta.get("attempts", 1),
            }

        return {
            "ok": True,
            "summary": clean,
            "raw_response": result,
            "attempts": meta.get("attempts", 1),
        }

    def _ensure_key_field_coverage(self, summary: str, project_name: str, answers: dict[str, str]) -> str:
        text = str(summary or "").strip()
        if not text:
            text = "Project summary unavailable."
        text = self._normalize_section_breaks(text)

        def answer(key: str) -> str:
            value = str((answers or {}).get(key) or "").strip()
            if not value or self._looks_absent(value):
                return ABSENT_ANSWER
            return value

        sections: list[tuple[str, str]] = [
            ("Title", project_name or "Untitled project"),
            ("Scope", answer("scope")),
            ("Participants", answer("participants")),
            ("Timeline", answer("timeline")),
            ("Quoted / Commercial", answer("commercial")),
            ("Invoice status", answer("commercial")),
            ("Findings", answer("risks_actions")),
            ("Recommendations", answer("risks_actions")),
            ("Gaps", answer("missing_info")),
        ]

        missing_parts: list[str] = []
        lowered = text.lower()
        for label, value in sections:
            token = f"{label.lower()}:"
            if token in lowered:
                continue
            compact = " ".join(str(value).split())
            if len(compact) > 180:
                compact = compact[:177].rstrip() + "..."
            missing_parts.append(f"{label}: {compact}")

        if not missing_parts:
            return self._normalize_section_breaks(text)
        return self._normalize_section_breaks(text + "\n\n" + "\n".join(missing_parts))

    def _normalize_section_breaks(self, summary_text: str) -> str:
        text = " ".join(str(summary_text or "").split())
        labels = [
            "Title",
            "Scope",
            "Participants",
            "Timeline",
            "Quoted / Commercial",
            "Invoice status",
            "Findings",
            "Recommendations",
            "Gaps",
            "Commercial",
        ]
        for label in labels:
            pattern = re.escape(label) + r":"
            text = re.sub(r"\s*" + pattern, f"\n{label}: ", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def refresh_summary_embedding(
        self,
        client_name: str,
        project_key: str,
        project_name: str,
        summary_text: str,
        summary_method: str,
        questionnaire_answers: dict[str, Any] | None = None,
    ) -> None:
        summary = self._normalize_section_breaks(str(summary_text or "").strip())
        if not summary:
            return

        answers = questionnaire_answers or {}
        answer_lines = []
        if isinstance(answers, dict):
            for key in sorted(answers.keys()):
                value = str(answers.get(key) or "").strip()
                if value:
                    answer_lines.append(f"- {key}: {value}")

        chunk_text = "\n".join(
            [
                f"Client: {client_name}",
                f"Project: {project_name}",
                f"Project key: {project_key}",
                f"Summary method: {summary_method}",
                "Summary:",
                summary,
                "Questionnaire:",
                *(answer_lines or ["- none"]),
            ]
        )
        embedding = self.embedder.embed_text(chunk_text)
        metadata = {
            "client_name": client_name,
            "project_key": project_key,
            "project_name": project_name,
            "document_kind": "project_summary",
            "is_client_related": True,
            "summary_method": summary_method,
            "source_type": "lazy_llm_summary",
        }
        source_document = f"__project_summary__/{client_name}/{project_key}"
        self.storage.upsert_vector(
            source_document=source_document,
            chunk_text=chunk_text,
            embedding=embedding,
            metadata=metadata,
            client_name=client_name,
        )

    def _looks_absent(self, text: str) -> bool:
        low = str(text or "").strip().lower()
        if not low:
            return True
        return (
            "insufficient evidence" in low
            or "not present" in low
            or "not enough information" in low
            or low == "unknown"
        )

    def _llm_error_code(self) -> str:
        raw = str(getattr(self.llm, "last_error", "") or "").strip()
        return raw or "no_response"

    def _llm_error_message(self) -> str:
        code = self._llm_error_code()
        mapping = {
            "request_timed_out": "Request to local model timed out.",
            "empty_model_response": "Model returned an empty response.",
            "invalid_json_response": "Model returned invalid JSON.",
            "request_error": "Could not reach local model service.",
            "no_response": "Model returned no response.",
        }
        if code.startswith("http_error_"):
            return f"Local model service returned HTTP error ({code.replace('http_error_', '')})."
        return mapping.get(code, f"Model request failed ({code}).")

    def _failure(
        self,
        client_name: str,
        project_key: str,
        project_name: str,
        documents: list[dict[str, Any]],
        code: str,
        message: str,
        *,
        strategy: str,
        query_text: str | None = None,
        rag_hits: list[dict[str, Any]] | None = None,
        digest: list[dict[str, Any]] | None = None,
        question_runs: list[dict[str, Any]] | None = None,
        summary_attempt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "ok": False,
            "strategy": strategy,
            "error_code": code,
            "error_message": message,
            "client_name": client_name,
            "project_key": project_key,
            "project_name": project_name,
            "question_runs": question_runs or [],
            "summary_attempt": summary_attempt or {},
        }
        self._append_failure_log(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "strategy": strategy,
                "client_name": client_name,
                "project_key": project_key,
                "project_name": project_name,
                "error_code": code,
                "error_message": message,
                "query_text": query_text or "",
                "document_count": len(documents),
                "document_titles": [str(doc.get("title") or doc.get("document_name") or "") for doc in documents[:12]],
                "question_runs": question_runs or [],
                "summary_attempt": summary_attempt or {},
                "rag_hits": [
                    {
                        "score": hit.get("score"),
                        "client_name": hit.get("client_name"),
                        "source_document": hit.get("source_document"),
                        "metadata": hit.get("metadata"),
                        "chunk_preview": str(hit.get("chunk_text") or "")[:180],
                    }
                    for hit in (rag_hits or [])[:12]
                ],
                "digest": digest or [],
            }
        )
        return payload

    def _append_failure_log(self, payload: dict[str, Any]) -> None:
        self.storage.add_project_summary_failure(
            client_name=str(payload.get("client_name") or ""),
            project_key=str(payload.get("project_key") or ""),
            project_name=str(payload.get("project_name") or ""),
            error_code=str(payload.get("error_code") or "unknown_error"),
            error_message=str(payload.get("error_message") or "Unknown error."),
            payload=payload,
        )
        self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.failure_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except OSError:
            pass

    def _load_questionnaire(self) -> list[tuple[str, str]]:
        text = self.questionnaire_path.read_text(encoding="utf-8", errors="ignore")
        rows: list[tuple[str, str]] = []
        idx = 1
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                key, question = line.split("|", 1)
                key = key.strip() or f"q{idx}"
                question = question.strip()
            else:
                key = f"q{idx}"
                question = line
            if not question:
                continue
            rows.append((key, question))
            idx += 1
        return rows or self._default_questions()

    def _chunk_questions(self, questions: list[tuple[str, str]], size: int) -> list[list[tuple[str, str]]]:
        chunked: list[list[tuple[str, str]]] = []
        for index in range(0, len(questions), max(1, size)):
            chunked.append(questions[index : index + max(1, size)])
        return chunked

    def _ensure_default_questionnaire(self) -> None:
        self.questionnaire_path.parent.mkdir(parents=True, exist_ok=True)
        if self.questionnaire_path.exists():
            return
        lines = [
            "# Project summary questionnaire",
            "# Format: key|question",
            "scope|What is this project about and what is the work scope?",
            "participants|Who are the key people or teams involved?",
            "timeline|What dates, milestones, or sequencing are relevant?",
            "commercial|What budget, quote, PO, invoice, or logistics status is evidenced?",
            "risks_actions|What key risks/issues and recommended actions are evidenced?",
            "missing_info|What important information is still missing or uncertain?",
        ]
        self.questionnaire_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _default_questions(self) -> list[tuple[str, str]]:
        return [
            ("scope", "What is this project about and what is the work scope?"),
            ("participants", "Who are the key people or teams involved?"),
            ("timeline", "What dates, milestones, or sequencing are relevant?"),
            ("commercial", "What budget, quote, PO, invoice, or logistics status is evidenced?"),
            ("risks_actions", "What key risks/issues and recommended actions are evidenced?"),
            ("missing_info", "What important information is still missing or uncertain?"),
        ]
