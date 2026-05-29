# Client Profiler (Python)

A modular document profiler for consulting teams that ingests mixed document types, classifies report vs non-report business documents, extracts client-centric knowledge, builds profile/timeline records, and stores searchable embeddings.

## What this baseline does

- Ingests `.txt`, `.md`, `.html`, `.pdf`, `.docx`, `.xlsx`, and `.csv` files.
- Classifies documents as report/non-report/unknown and estimates client relevance.
- Extracts profile data using a local LLM (Ollama) with regex fallback.
- Captures timeline events and profile facts in SQLite.
- Groups related reports, quotes, purchase orders, email chains, and access documents under common project nodes.
- Generates project summaries from related documents, using the local LLM when available with rule-based fallback.
- Stores text chunk embeddings for semantic retrieval.
- Supports tree-like profile paths such as `Unit 1/Line 3/Weld 6/12:00 position`.

## Architecture

The project is intentionally modular:

- `client_profiler/ingestion/readers.py`: file-type parsing and text extraction.
- `client_profiler/classification/document_classifier.py`: report/non-report/client-related classification.
- `client_profiler/extraction/llm.py`: local LLM adapter (Ollama).
- `client_profiler/extraction/extractor.py`: structured extraction and regex fallback.
- `client_profiler/profiling/profile_builder.py`: profile-tree and timeline updates.
- `client_profiler/projects/project_associator.py`: project association and project-summary generation.
- `client_profiler/embeddings/embedder.py`: local embedding backend with deterministic fallback.
- `client_profiler/embeddings/retriever.py`: cosine similarity semantic search.
- `client_profiler/storage.py`: SQLite persistence for docs, profiles, timeline, vectors.
- `client_profiler/pipeline/profiler.py`: end-to-end orchestration.
- `cli.py`: command line interface.

## Install

1. Create environment and install dependencies:
   - `python -m venv .venv`
   - Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
   - `pip install -r requirements.txt`

2. Optional local LLM setup:
   - Install Ollama.
   - Pull a model: `ollama pull llama3.1`
   - Start Ollama service (typically auto-started).

If Ollama is unavailable, extraction falls back to regex heuristics.

## Usage

### Two-tier ingest (default)

- Ingestion now defaults to a fast deterministic path (parse/classify/metadata/vector storage) with LLM-heavy enrichment deferred.
- Project key fields are generated on demand from the UI, field-by-field.
- Project summaries are generated on demand (or when explicitly enabled during ingest).

- Ingest one file:
  - `python cli.py ingest-file "path/to/document.pdf"`
  - Write structured ingest events to JSONL for diagnostics:
    - `python cli.py ingest-file "path/to/document.pdf" --event-log data/diagnostics/ingest_events.jsonl`
  - Force re-ingest (captures updates/drafts becoming final):
    - `python cli.py ingest-file "path/to/document.pdf" --force`
  - Disable live terminal status/progress output:
    - `python cli.py ingest-file "path/to/document.pdf" --no-status`
  - Enable full LLM ingest behavior (slower):
    - `python cli.py ingest-file "path/to/document.pdf" --enable-ingest-llm --generate-project-summaries`

- Ingest folder recursively:
  - `python cli.py ingest-dir "path/to/documents"`
  - Write structured ingest events to JSONL for diagnostics:
    - `python cli.py ingest-dir "path/to/documents" --event-log data/diagnostics/ingest_events.jsonl`
  - Force re-ingest all files:
    - `python cli.py ingest-dir "path/to/documents" --force`
  - Disable live terminal status/progress output:
    - `python cli.py ingest-dir "path/to/documents" --no-status`
  - Enable full LLM ingest behavior (slower):
    - `python cli.py ingest-dir "path/to/documents" --force --enable-ingest-llm --generate-project-summaries`

- Query semantic memory:
  - `python cli.py query "What were the recommendations for Unit 1 piping?" --client "Client A"`

- List profiled clients:
  - `python cli.py list-clients`

- Show per-client operational and financial metrics:
  - `python cli.py client-metrics`
  - Single client: `python cli.py client-metrics --client "NorthRiver Energy"`

- Manually set report date when extraction is missing:
  - `python cli.py set-report-date "path/to/document.docx" 2025-10-15`

- Remove specific components when cleanup is needed:
  - Delete a client profile and related nodes/timeline data:
    - `python cli.py delete-client "Client Name"`
  - Merge duplicate clients (for case variants or alias cleanup):
    - `python cli.py merge-client "Source Client" "Target Client"`
  - Detect and merge only very high-confidence duplicate clients:
    - Preview only: `python cli.py cleanup-merge-clients --dry-run --min-confidence 0.95`
    - Apply merges: `python cli.py cleanup-merge-clients --min-confidence 0.95`
  - Reset the database (with backup by default):
    - `python cli.py reset-db --backup`
  - Delete a specific node:
    - `python cli.py delete-node "Client Name" "Projects/Project A/Workscope"`
  - Delete a report/document path and its timeline/vector/version entries:
    - `python cli.py delete-report "path/to/document.docx"`

- Find and clean suspicious one-document clients (client name equals only document name):
  - `python cli.py list-suspicious-clients`
  - `python cli.py cleanup-bad-clients`

- Post-ingest automatic cleanup:
  - After `ingest-file` and `ingest-dir`, the CLI now runs a high-confidence duplicate-client merge cleanup automatically.
  - This behavior is enabled by default and can be controlled with:
    - `--auto-merge-cleanup / --no-auto-merge-cleanup`
    - `--merge-confidence <threshold>`

- Run automated smoke test:
  - `python scripts/smoke_test.py`

- Analyze ingest logs and DB quality anomalies:
  - `python scripts/analyze_ingest_log.py --event-log data/diagnostics/ingest_events.jsonl --db data/profiler.db --output data/diagnostics/ingest_report.json`

- Evaluate retrieval quality on the labeled benchmark set:
  - `python scripts/evaluate_retrieval_quality.py --top-k 8 --show-failures`

- Generate mixed-format logistics fixtures for all current sample reports:
  - `python scripts/generate_project_logistics_docs.py`
  - Generate and ingest into the active database:
    - `python scripts/generate_project_logistics_docs.py --ingest`

- Generate varied fictional financial/expense documents and optional ingest:
  - `python scripts/generate_financial_variant_docs.py`
  - Generate and ingest into a dedicated DB:
    - `python scripts/generate_financial_variant_docs.py --ingest --force --db data/profiler_financial_variants.db`

- Evaluate financial metrics against expected totals from generated fixtures:
  - `python scripts/evaluate_financial_metrics.py --db data/profiler_financial_variants.db --expected data/financial_variants_expected.json`

## Web UI (FastAPI)

Run the web server:

- `uvicorn client_profiler.web.app:app --reload`

Then open:

- `http://127.0.0.1:8000`

Web features included:

- Dashboard listing all discovered clients with node/event/chunk counts.
- Recent timeline events across all clients.
- Per-client page showing profile tree and timeline history.
- Dedicated timeline explorer page at `/timeline` with filters for client, date range, document type, and contact.
- `/timeline` now includes a long vertical scaled timeline card with orthogonal event leaders and wheel-based zoom around the cursor.
- Maintenance console page at `/admin/maintenance` provides merge, cleanup, and delete actions from the web UI.
- Dedicated report lookup page at `/reports` with filters for client, date range, document type, and contact.
- Report version timeline page at `/report/versions?path=...` with diff-style summaries.
- Clickable document links from timeline cards.
- Clickable source-report links in each profile node.
- Client page report card listing all ingested reports associated with that client.
- Project nodes in the client profile tree with aggregated source documents and a generated project summary.
- Manual date-set forms in timeline/report pages when report date is missing.
- Document viewer route at `/document?path=...`:
  - Browser-friendly files (`.pdf`, `.txt`, `.md`, `.html`, `.csv`) open as original files.
  - Other files (`.docx`, `.xlsx`, etc.) render as markdown based on extracted text.
- JSON endpoints at `/api/clients` and `/api/client/{client_name}`.

## Notes on flexibility

This starter is designed for extension:

- Replace or enrich `DocumentClassifier` with a learned classifier.
- Expand extraction schema in `ProfileExtractor`.
- Add richer hierarchy inference rules to map extracted entities into profile paths.
- Add specialized parsers for invoices, purchase orders, timesheets, and access requests.
- Expand project linking prompts if you want stronger LLM-only resolution for ambiguous commercial references.
- Add deduplication checks by content hash before storing timeline/profile events.

## Client guardrails

- Client names are now treated conservatively during ingestion.
- The profiler only accepts client assignment when one of these holds:
  - explicit client marker is present in text (`Client:` / `Client Name:` style),
  - references (quote/PO/access/project code) resolve uniquely to one known client,
  - or the extracted name already matches a known client.
- Names that look like document filenames/folder names are rejected to prevent accidental pseudo-clients.
- Ingest now normalizes case variants to an existing canonical client name when one already exists in storage.
- If client markers are missing, ingest can infer client from report-folder naming patterns (for example `report_01_2016_northriver_energy_...`) and then applies confidence checks.

## Deduplication behavior

- Re-ingesting the same file with the same content now returns `status: "skipped_duplicate"`.
- Dedup key is `(source_path, content_hash)` to prevent repeated timeline/vector/profile inserts from reruns.
- With `--force`, files are re-ingested even if content hash matches, enabling update workflows.
- When node content changes on re-ingest, prior node facts are archived and shown in UI as superseded history.
- Re-ingest also writes version snapshots for reports, so users can inspect what changed between versions.

## Unsupported files

- Unsupported file types are skipped with `status: "skipped_unsupported"` during directory ingest.
- They are also emitted as `file_unsupported` events in optional ingest event logs, so diagnostics can separate skipped files from true ingest failures.

## Data model idea for tree profiles

The current baseline writes nodes with string paths. To represent your example:

- `Client A -> Unit 1 -> Line 3 -> Weld 6 -> 12:00 position`

you can store the path as:

- `Unit 1/Line 3/Weld 6/12:00 position`

and enrich node facts as more reports arrive.

## Next suggested enhancements

- Add pydantic schemas and strict JSON validation for LLM output.
- Add confidence scoring and human-review queue for low-confidence extraction.
- Add web UI (FastAPI + simple frontend) for profile browsing and timeline visualisation.
- Add automated tests for each module.
