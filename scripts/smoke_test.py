from __future__ import annotations

import sqlite3
import shutil
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client_profiler.config import ProfilerConfig
from client_profiler.embeddings.embedder import LocalEmbedder
from client_profiler.embeddings import VectorRetriever
from client_profiler.pipeline.profiler import ClientProfiler
from client_profiler.web.app import create_app


def _write_sample_docs(base_dir: Path) -> None:
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    report = docs_dir / "inspection_report.md"
    report.write_text(
        """
Client Name: Acme Refining
Report Type: Inspection Report
Author: Jane Smith
Client Contact: Bob Jones
Date: 2026-05-20

Findings
- Weld 6 on Line 3 shows minor porosity at 12:00 position.
- Corrosion observed on adjacent piping support.

Recommendations
- Perform NDT on Weld 6 within 7 days.
- Recoat support area and reinspect in next outage.
""".strip()
    )

    invoice = docs_dir / "invoice.txt"
    invoice.write_text(
        """
Invoice #INV-1007
Amount Due: 12000
Client: Acme Refining
Date: 2026-05-21
Description: Engineering consulting services
""".strip()
    )

    quote = docs_dir / "quote.txt"
    quote.write_text(
        """
Quote Number: Q-2026-044
Client: Acme Refining
Project: Crude Unit Outage Planning
Project Code: AR-CRU-OUT-2026
Date: 2026-05-18
Scope: Mobilise inspection and shutdown planning team.
""".strip()
    )

    purchase_order = docs_dir / "purchase_order.txt"
    purchase_order.write_text(
        """
Purchase Order Number: PO-2026-044
Client: Acme Refining
Project: Crude Unit Outage Planning
Project Code: AR-CRU-OUT-2026
Date: 2026-05-19
Approved against Quote Number Q-2026-044.
""".strip()
    )

    email_chain = docs_dir / "email_chain.md"
    email_chain.write_text(
        """
# Shutdown Logistics Email Chain

- **Client:** Acme Refining
- **Project:** Crude Unit Outage Planning
- **Project Code:** AR-CRU-OUT-2026
- **Date:** 2026-05-20

## Email Chain

- From: planner@acme.example.com
  To: coordinator@vector.example.com
  Subject: RE: Crude Unit Outage Planning access list
  Sent: 2026-05-20 08:15
  Body: Please confirm the crew list against Quote Number Q-2026-044 and PO-2026-044.
""".strip()
    )

    access_request = docs_dir / "access_request.txt"
    access_request.write_text(
        """
Access Request Number: AR-2026-044
Client: Acme Refining
Project: Crude Unit Outage Planning
Project Code: AR-CRU-OUT-2026
Date: 2026-05-21
Workers: Alyssa Tran, Jordan Pike, Marco Singh
Approval: Approved for escorted entry to the crude unit MCC room.
""".strip()
    )


def _db_counts(db_path: Path) -> tuple[int, int, int, int]:
    with sqlite3.connect(db_path) as conn:
        documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        profiles = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
        timeline = conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0]
        vectors = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    return int(documents), int(profiles), int(timeline), int(vectors)


def run_smoke_test() -> None:
    td = tempfile.mkdtemp(prefix="client_profiler_smoke_")
    root = Path(td)
    try:
        _write_sample_docs(root)

        LocalEmbedder._load_backend = lambda self: setattr(self, "_backend", None)

        db_path = root / "data" / "profiler.db"
        config = ProfilerConfig(
            data_dir=root / "data",
            db_path=db_path,
            llm_provider="none",
        )
        profiler = ClientProfiler(config)

        docs_dir = root / "docs"
        first_results = profiler.ingest_directory(docs_dir)
        assert len(first_results) == 6, f"Expected 6 docs ingested, got {len(first_results)}"
        assert all(r.get("status") == "ingested" for r in first_results), first_results

        kinds = {Path(r["path"]).name: r.get("document_kind") for r in first_results}
        assert kinds.get("quote.txt") == "quote", kinds
        assert kinds.get("purchase_order.txt") == "purchase_order", kinds
        assert kinds.get("email_chain.md") == "email_chain", kinds
        assert kinds.get("access_request.txt") == "access_request", kinds

        counts_after_first = _db_counts(db_path)

        second_results = profiler.ingest_directory(docs_dir)
        assert all(r.get("status") == "skipped_duplicate" for r in second_results), second_results

        counts_after_second = _db_counts(db_path)
        assert counts_after_first == counts_after_second, (
            "Counts changed after duplicate ingest: "
            f"first={counts_after_first}, second={counts_after_second}"
        )

        clients = profiler.storage.list_clients()
        assert "Acme Refining" in clients, f"Expected Acme Refining in clients, got {clients}"

        project_nodes = [
            node
            for node in profiler.storage.list_profile_nodes("Acme Refining")
            if node["node_path"] == "Projects/Crude Unit Outage Planning"
        ]
        assert project_nodes, "Expected top-level project node for Crude Unit Outage Planning"
        # Two-tier ingest defaults to deferred LLM enrichment, so summaries should not be generated during ingest.
        assert not project_nodes[0]["facts"].get("project_summary"), project_nodes[0]["facts"]

        query_embedding = profiler.embedder.embed_text("weld 6 recommendations")
        hits = VectorRetriever(profiler.storage).search(query_embedding, top_k=3, client_name="Acme Refining")
        assert hits, "Expected semantic query hits"

        app = create_app(config)
        with TestClient(app) as client:
            root_res = client.get("/")
            assert root_res.status_code == 200, f"Expected / status 200, got {root_res.status_code}"

            timeline_res = client.get("/timeline")
            assert timeline_res.status_code == 200, f"Expected /timeline status 200, got {timeline_res.status_code}"
            assert "Scaled Project Timeline" in timeline_res.text, "Expected visual timeline card in timeline page"

            api_res = client.get("/api/clients")
            assert api_res.status_code == 200, f"Expected /api/clients status 200, got {api_res.status_code}"
            payload = api_res.json()
            names = [item.get("name") for item in payload]
            assert "Acme Refining" in names, f"Expected Acme Refining in API payload, got {names}"

        print("SMOKE TEST PASSED")
        print(f"DB counts: documents={counts_after_first[0]}, profiles={counts_after_first[1]}, timeline={counts_after_first[2]}, vectors={counts_after_first[3]}")
    finally:
        # Cleanup can be flaky on Windows if background handles linger briefly.
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    run_smoke_test()
