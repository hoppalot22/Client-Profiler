from __future__ import annotations

import re

from client_profiler.models import DocumentClassification


class DocumentClassifier:
    REPORT_PATTERNS = [
        r"\breport\b",
        r"\binspection\b",
        r"\bfindings?\b",
        r"\brecommendations?\b",
        r"\bsummary\b",
    ]

    NON_REPORT_PATTERNS = {
        "quote": [r"\bquote\b", r"\bquotation\b", r"\bestimate\b"],
        "purchase_order": [r"\bpurchase\s+order\b", r"\bpo\s*#?\b"],
        "access_request": [r"\baccess\s+request\b", r"\bpermit\s+to\s+work\b"],
        "email_chain": [
            r"(?m)^\s*-?\s*from:",
            r"(?m)^\s*-?\s*to:",
            r"(?m)^\s*-?\s*subject:",
            r"(?m)^\s*-?\s*sent:",
            r"\bemail\s+chain\b",
            r"\bfrom:\s*\S+@\S+",
            r"\bsubject:\b",
        ],
        "invoice": [r"\binvoice\b", r"\bamount\s+due\b"],
        "expense_report": [r"\bexpense\s+report\b", r"\bexpense\s+summary\b", r"\btravel\s+expense\b", r"\bcost\s+breakdown\b"],
        "timesheet": [r"\btimesheet\b", r"\bhours\s+worked\b"],
    }

    CLIENT_CONTEXT_PATTERNS = [
        r"\bclient\b",
        r"\bproject\b",
        r"\bsite\b",
        r"\bunit\b",
        r"\bline\b",
        r"\basset\b",
        r"\bquote\b",
        r"\bpurchase\s+order\b",
        r"\baccess\s+request\b",
        r"\bcost\b",
        r"\bexpense\b",
        r"\bshutdown\b",
        r"\boutage\b",
    ]

    DOCUMENT_KIND_KEYWORDS = {
        "quote": ["quote", "quotation", "estimate"],
        "purchase_order": ["purchase_order", "purchase-order", "po_", "po-"],
        "access_request": ["access_request", "access-request", "permit"],
        "email_chain": ["email_chain", "email-chain", "email"],
        "invoice": ["invoice", "inv_"],
        "expense_report": ["expense", "cost", "budget"],
        "timesheet": ["timesheet"],
    }

    def classify(self, text: str, source_path: str | None = None) -> DocumentClassification:
        lowered = text.lower()
        path_text = str(source_path or "").lower()

        non_report_score = 0.0
        non_report_kind = None
        for kind, patterns in self.NON_REPORT_PATTERNS.items():
            score = float(sum(1 for p in patterns if re.search(p, lowered)))
            # Fallback hint from filename/path when textual cues are sparse.
            if path_text and any(token in path_text for token in self.DOCUMENT_KIND_KEYWORDS.get(kind, [])):
                score += 0.75
            if kind == "email_chain" and "email_chain" in path_text:
                score += 2.0

            # Access request and email chains were frequently misrouted as purchase orders.
            if kind == "access_request" and re.search(r"\b(approval|workers?|visitor|permit)\b", lowered):
                score += 0.75
            if kind == "email_chain":
                email_marker_count = sum(
                    1 for p in self.NON_REPORT_PATTERNS["email_chain"] if re.search(p, lowered)
                )
                if email_marker_count >= 2:
                    score += 1.0
                if email_marker_count >= 3:
                    score += 0.75

            if score > non_report_score:
                non_report_score = score
                non_report_kind = kind

        report_score = sum(1 for p in self.REPORT_PATTERNS if re.search(p, lowered))
        client_score = sum(1 for p in self.CLIENT_CONTEXT_PATTERNS if re.search(p, lowered))

        is_client_related = client_score >= 2 or re.search(r"\bclient\s*(?:name)?\b", lowered) is not None

        if non_report_score > 0 and non_report_score >= float(report_score) and non_report_kind:
            kind = non_report_kind
            confidence = min(0.97, 0.42 + 0.13 * non_report_score)
            rationale = f"Detected non-report cues for {non_report_kind}."
        elif report_score > 0:
            kind = "report"
            confidence = min(0.95, 0.45 + 0.1 * report_score)
            rationale = "Detected report-like structure and terminology."
        else:
            kind = "unknown"
            confidence = 0.4
            rationale = "Insufficient strong cues for document kind."

        if not is_client_related and kind == "report" and client_score > 0:
            is_client_related = True
        if not is_client_related and kind in {"quote", "purchase_order", "access_request", "email_chain"} and client_score > 0:
            is_client_related = True

        return DocumentClassification(
            document_kind=kind,
            is_client_related=is_client_related,
            confidence=confidence,
            rationale=rationale,
        )
