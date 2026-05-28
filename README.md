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

- Ingest one file:
  - `python cli.py ingest-file "path/to/document.pdf"`
  - Force re-ingest (captures updates/drafts becoming final):
    - `python cli.py ingest-file "path/to/document.pdf" --force`

- Ingest folder recursively:
  - `python cli.py ingest-dir "path/to/documents"`
  - Force re-ingest all files:
    - `python cli.py ingest-dir "path/to/documents" --force`

- Query semantic memory:
  - `python cli.py query "What were the recommendations for Unit 1 piping?" --client "Client A"`

- List profiled clients:
  - `python cli.py list-clients`

- Manually set report date when extraction is missing:
  - `python cli.py set-report-date "path/to/document.docx" 2025-10-15`

- Remove specific components when cleanup is needed:
  - Delete a client profile and related nodes/timeline data:
    - `python cli.py delete-client "Client Name"`
  - Merge duplicate clients (for case variants or alias cleanup):
    - `python cli.py merge-client "Source Client" "Target Client"`
  - Delete a specific node:
    - `python cli.py delete-node "Client Name" "Projects/Project A/Workscope"`
  - Delete a report/document path and its timeline/vector/version entries:
    - `python cli.py delete-report "path/to/document.docx"`

- Find and clean suspicious one-document clients (client name equals only document name):
  - `python cli.py list-suspicious-clients`
  - `python cli.py cleanup-bad-clients`

- Run automated smoke test:
  - `python scripts/smoke_test.py`

- Evaluate retrieval quality on the labeled benchmark set:
  - `python scripts/evaluate_retrieval_quality.py --top-k 8 --show-failures`

- Generate mixed-format logistics fixtures for all current sample reports:
  - `python scripts/generate_project_logistics_docs.py`
  - Generate and ingest into the active database:
    - `python scripts/generate_project_logistics_docs.py --ingest`

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

## Deduplication behavior

- Re-ingesting the same file with the same content now returns `status: "skipped_duplicate"`.
- Dedup key is `(source_path, content_hash)` to prevent repeated timeline/vector/profile inserts from reruns.
- With `--force`, files are re-ingested even if content hash matches, enabling update workflows.
- When node content changes on re-ingest, prior node facts are archived and shown in UI as superseded history.
- Re-ingest also writes version snapshots for reports, so users can inspect what changed between versions.

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
