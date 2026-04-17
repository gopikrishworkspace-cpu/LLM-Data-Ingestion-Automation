# Orion Ingestion Pipeline

An LLM-powered entity extraction and knowledge base ingestion system. It processes unstructured documents (PDF, DOCX, PPTX, TXT, URLs) through a multi-step pipeline to extract, normalize, deduplicate, and persist structured entities into a local knowledge base.

## Pipeline Overview

```
Input Document → Chunking → Step 1: Extract → Step 2: Normalize → Step 3: Deduplicate → Step 4: Persist
```

| Step | Prompt | Description |
|------|--------|-------------|
| **Step 1** | `prompt/step1.txt` | Extract entities (23 types) from text chunks |
| **Step 2** | `prompt/step2.txt` | Normalize names, link relationships, assign confidence scores |
| **Step 3** | `prompt/step3.txt` | Cluster duplicates, merge entities, assign statuses |
| **Step 4** | `prompt/step4_merge.txt` | Cross-batch merge for global deduplication and persistence |

## Project Structure

```
├── run.py              # Entry point
├── pipeline.py         # 4-step pipeline orchestrator
├── fetcher.py          # URL fetching and HTML-to-text conversion
├── chunking.py         # PDF/DOCX/PPTX/TXT extraction and text chunking
├── llm_client.py       # LLM clients (Ollama local / Medtronic GPT cloud)
├── storage.py          # Atomic file I/O, conflict detection, confidence-based merging
├── validation.py       # Type correction, scoring, relationship enrichment, dedup clustering
├── prompt/             # System prompts for each pipeline step
├── input/              # Sample input documents
├── knowledge_base/     # Persisted entity store (auto-populated on runs)
│   └── index/          # state.json registry and conflicts.json log
└── requirements.txt    # Python dependencies
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ORION_API_TOKEN` | API token for Medtronic GPT |
| `ORION_SUBSCRIPTION_KEY` | Subscription key for Medtronic GPT |

## Usage

```bash
python run.py input/input2.txt
```

Accepts file paths (PDF, DOCX, PPTX, TXT) or URLs as input.

## Key Features

- **23 entity types** — architecture, metric, product, pattern, terminology, and more
- **Deterministic entity IDs** — SHA-256 hashed from type + name + attributes for stable cross-run identity
- **Parallel extraction** — configurable thread pool (default 4 workers)
- **Conflict resolution** — confidence-based merging with unresolvable conflicts logged
- **Retry & resilience** — exponential backoff, validation retries, optional fail-fast mode

## Dependencies

- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF text extraction
- [python-docx](https://python-docx.readthedocs.io/) — DOCX text extraction
- [python-pptx](https://python-pptx.readthedocs.io/) — PPTX text extraction
