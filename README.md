# docqa-engine

**Upload documents, ask questions -- get cited answers with a prompt engineering lab.**

![CI](https://github.com/ChunkyTortoise/docqa-engine/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-157%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B.svg?logo=streamlit&logoColor=white)](https://ct-document-engine.streamlit.app)

**[Live Demo](https://ct-document-engine.streamlit.app)** -- try it without installing anything.

## What This Solves

- **RAG pipeline from upload to answer** -- Ingest documents (PDF, DOCX, TXT, MD, CSV), chunk them with pluggable strategies, embed with TF-IDF, and retrieve using BM25 + dense hybrid search with Reciprocal Rank Fusion
- **Prompt engineering lab for A/B testing** -- Create prompt templates, run the same question through different strategies side-by-side, compare outputs
- **Citation accuracy matters** -- Faithfulness, coverage, and redundancy scoring for every generated citation

## Architecture

```
Documents (PDF, DOCX, TXT, MD, CSV)
         |
         v
+--------------+    +--------------+    +--------------+
|   Ingest     |--->|   Chunk      |--->|   Embed      |
|  (file I/O,  |    |  (fixed,     |    |  (TF-IDF     |
|   parsing)   |    |   sentence,  |    |   vectors)   |
+--------------+    |   semantic)  |    +------+-------+
                    +--------------+           |
                    +--------------+    +------v-------+
                    |   Answer     |<---|   Retrieve   |
                    |  (LLM gen,   |    |  (BM25 +     |
                    |   citations) |    |   Dense RRF) |
                    +------+-------+    +--------------+
                           |
                +----------+----------+
                |                     |
         +------v-------+    +-------v--------+
         |  Prompt Lab  |    | Citation Scorer |
         |  (A/B test,  |    | (faithfulness,  |
         |   versions)  |    |  coverage)      |
         +--------------+    +----------------+
```

## Modules

| Module | File | Description |
|--------|------|-------------|
| **Ingest** | `ingest.py` | Multi-format document loading (PDF, DOCX, TXT, MD, CSV) |
| **Chunking** | `chunking.py` | Pluggable chunking strategies: fixed-size, sentence-boundary, semantic |
| **Embedder** | `embedder.py` | TF-IDF embedding (5,000 features, no external API calls) |
| **Retriever** | `retriever.py` | BM25 + dense cosine + hybrid RRF fusion |
| **Answer** | `answer.py` | Context-aware answer generation with source citations |
| **Prompt Lab** | `prompt_lab.py` | Prompt versioning and A/B comparison framework |
| **Citation Scorer** | `citation_scorer.py` | Citation faithfulness, coverage, and redundancy scoring |
| **Evaluator** | `evaluator.py` | Retrieval metrics: MRR, NDCG@K, Precision@K, Recall@K, Hit Rate |
| **Batch** | `batch.py` | Parallel batch ingestion and query processing |
| **Exporter** | `exporter.py` | JSON/CSV export for results and metrics |
| **Cost Tracker** | `cost_tracker.py` | Per-query token and cost tracking |
| **Pipeline** | `pipeline.py` | End-to-end DocQAPipeline class |

## Quick Start

```bash
git clone https://github.com/ChunkyTortoise/docqa-engine.git
cd docqa-engine
pip install -r requirements.txt
make test
make demo
```

## Demo Documents

| Document | Topic | Content |
|----------|-------|---------|
| `python_guide.md` | Python Basics | Variables, control flow, functions, classes, error handling |
| `machine_learning.md` | ML Concepts | Supervised/unsupervised, regression, classification, neural networks |
| `startup_playbook.md` | Startup Advice | Product-market fit, MVP, fundraising, team building, metrics |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit (4 tabs) |
| Embeddings | scikit-learn (TF-IDF) |
| Retrieval | BM25 (Okapi) + Dense (cosine) + RRF |
| Document Parsing | PyPDF2, python-docx |
| Testing | pytest, pytest-asyncio (157 tests) |
| CI | GitHub Actions (Python 3.11, 3.12) |
| Linting | Ruff |

## Project Structure

```
docqa-engine/
├── app.py                          # Streamlit application (4 tabs)
├── docqa_engine/
│   ├── ingest.py                   # Document loading + parsing
│   ├── chunking.py                 # Pluggable chunking strategies
│   ├── embedder.py                 # TF-IDF embedding
│   ├── retriever.py                # BM25 + Dense + Hybrid (RRF)
│   ├── answer.py                   # LLM answer generation + citations
│   ├── prompt_lab.py               # Prompt versioning + A/B testing
│   ├── citation_scorer.py          # Citation accuracy scoring
│   ├── evaluator.py                # Retrieval metrics (MRR, NDCG, P@K)
│   ├── batch.py                    # Parallel batch processing
│   ├── exporter.py                 # JSON/CSV export
│   ├── cost_tracker.py             # Token + cost tracking
│   └── pipeline.py                 # End-to-end pipeline
├── demo_docs/                      # 3 sample documents
├── tests/                          # 12 test files, one per module
├── .github/workflows/ci.yml        # CI pipeline
├── Makefile                        # demo, test, lint, setup
└── requirements.txt
```

## Testing

```bash
make test                           # Full suite (157 tests)
python -m pytest tests/ -v          # Verbose output
python -m pytest tests/test_ingest.py  # Single module
```

## Related Projects

- [EnterpriseHub](https://github.com/ChunkyTortoise/EnterpriseHub) -- Real estate AI platform with BI dashboards and CRM integration
- [insight-engine](https://github.com/ChunkyTortoise/insight-engine) -- Upload CSV/Excel, get instant dashboards, predictive models, and reports
- [ai-orchestrator](https://github.com/ChunkyTortoise/ai-orchestrator) -- AgentForge: unified async LLM interface (Claude, Gemini, OpenAI, Perplexity)
- [scrape-and-serve](https://github.com/ChunkyTortoise/scrape-and-serve) -- Web scraping, price monitoring, Excel-to-web apps, and SEO tools
- [prompt-engineering-lab](https://github.com/ChunkyTortoise/prompt-engineering-lab) -- 8 prompt patterns, A/B testing, TF-IDF evaluation
- [llm-integration-starter](https://github.com/ChunkyTortoise/llm-integration-starter) -- Production LLM patterns: completion, streaming, function calling, RAG, hardening
- [Portfolio](https://chunkytortoise.github.io) -- Project showcase and services

## Deploy

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/chunkytortoise/docqa-engine/main/app.py)

## License

MIT -- see [LICENSE](LICENSE) for details.
