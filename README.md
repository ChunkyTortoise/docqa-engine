# docqa-engine

**Upload documents, ask questions -- get cited answers with a prompt engineering lab.**

[![CI](https://img.shields.io/github/actions/workflow/status/ChunkyTortoise/docqa-engine/ci.yml?label=CI)](https://github.com/ChunkyTortoise/docqa-engine/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F1C40F.svg)](LICENSE)

## Problem Statement

Teams waste hours searching through documents for answers. Knowledge is scattered across PDFs, Word docs, text files, and CSVs. When someone asks a question, the answer requires reading multiple sources, cross-referencing sections, and summarizing findings manually. There is no single tool that ingests documents, retrieves the most relevant passages, generates cited answers, and lets you experiment with different prompt strategies to improve quality.

## What This Solves

- **RAG pipeline from upload to answer** -- Ingest documents (PDF, DOCX, TXT, MD, CSV), chunk them intelligently with sentence-boundary detection, embed with TF-IDF, and retrieve using BM25 + dense hybrid search with Reciprocal Rank Fusion.
- **Prompt engineering lab for A/B testing** -- Create multiple prompt templates, run the same question through different strategies side-by-side, and compare outputs to find the most effective prompts for your use case.
- **BM25 + dense hybrid retrieval** -- Keyword search (BM25/Okapi) catches exact term matches while dense vector search (TF-IDF cosine similarity) captures semantic relationships. RRF fusion combines both ranked lists for better recall than either alone.

## Architecture

```
Documents (PDF, DOCX, TXT, MD, CSV)
         |
         v
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Ingest     │───>│   Chunk      │───>│   Embed      │
│  (file I/O,  │    │  (sentence   │    │  (TF-IDF     │
│   parsing)   │    │   boundary)  │    │   vectors)   │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐    ┌───────▼──────┐
                    │   Answer     │<───│   Retrieve   │
                    │  (LLM gen,   │    │  (BM25 +     │
                    │   citations) │    │   Dense RRF) │
                    └──────┬───────┘    └──────────────┘
                           │
                    ┌──────▼───────┐
                    │  Prompt Lab  │
                    │  (A/B test,  │
                    │   versions)  │
                    └──────────────┘
```

## Quick Start

```bash
git clone https://github.com/ChunkyTortoise/docqa-engine.git
cd docqa-engine
pip install -r requirements.txt

# Demo mode -- 3 sample documents, no config needed
make demo
```

## Core Features

### 1. Document Ingestion
Multi-format document loading with configurable chunking. Supports PDF (via PyPDF2), DOCX (via python-docx), plain text, Markdown, and CSV. Chunks use sentence-boundary detection with configurable size and overlap to preserve context across splits.

### 2. Hybrid Retrieval
Dual-index search combining BM25 (Okapi) keyword matching with dense vector cosine similarity. Results are fused using Reciprocal Rank Fusion (RRF) to produce a single ranked list that captures both exact term matches and semantic relationships.

### 3. Answer Generation
Context-aware answer generation with automatic source citations. Builds a context window from top-k retrieved chunks, annotates each with source references, and generates answers via LLM or mock mode. Every answer includes chunk-level citations with relevance scores.

### 4. Prompt Lab
Prompt versioning and A/B comparison framework. Create named prompt templates with configurable temperature and max tokens. Run the same question through two templates side-by-side to evaluate which strategy produces better answers. Tracks experiment history with evaluation scores.

### 5. TF-IDF Embedder
Lightweight embedding using scikit-learn's TfidfVectorizer with unigram + bigram features, English stop word removal, and sublinear TF scaling. No external API calls required -- embeddings are generated locally and support up to 5,000 features.

### 6. Streamlit UI
Four-tab interface: Documents (ingest and overview), Ask Questions (search and answer), Prompt Lab (A/B comparison), and Stats (pipeline metrics and cost tracking). Session state persists the pipeline across Streamlit rerenders.

## Demo Documents

| Document | Topic | Content |
|----------|-------|---------|
| `python_guide.md` | Python Basics | Variables, control flow, functions, classes, comprehensions, error handling |
| `machine_learning.md` | ML Concepts | Supervised/unsupervised, regression, classification, neural networks, overfitting |
| `startup_playbook.md` | Startup Advice | Product-market fit, MVP, fundraising, team building, metrics, pivoting |

## Project Structure

```
docqa-engine/
├── app.py                          # Streamlit application (4 tabs)
├── docqa_engine/
│   ├── __init__.py
│   ├── ingest.py                   # Document loading + chunking
│   ├── embedder.py                 # TF-IDF embedding
│   ├── retriever.py                # BM25 + Dense + Hybrid (RRF)
│   ├── answer.py                   # LLM answer generation + citations
│   ├── prompt_lab.py               # Prompt versioning + A/B testing
│   ├── cost_tracker.py             # Per-query token + cost tracking
│   └── pipeline.py                 # End-to-end DocQAPipeline class
├── demo_docs/
│   ├── python_guide.md             # Python programming guide
│   ├── machine_learning.md         # ML concepts overview
│   └── startup_playbook.md         # Startup strategy playbook
├── tests/                          # One test file per module
├── .github/workflows/ci.yml        # CI pipeline (Python 3.11, 3.12)
├── Makefile                        # demo, test, lint, clean, setup
├── pyproject.toml                  # Project config + ruff + pytest
├── requirements.txt                # Runtime dependencies
└── requirements-dev.txt            # Dev dependencies (pytest, ruff)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Embeddings | scikit-learn (TF-IDF) |
| Retrieval | BM25 (Okapi) + Dense (cosine) + RRF |
| Numerical | NumPy |
| Document Parsing | PyPDF2, python-docx |
| Testing | pytest, pytest-asyncio |
| CI | GitHub Actions (Python 3.11, 3.12) |
| Linting | Ruff |

## Testing

```bash
make test                                   # Full suite
python -m pytest tests/ -v                  # Verbose output
python -m pytest tests/test_ingest.py       # Single module
```

## Related Projects

- [EnterpriseHub](https://github.com/ChunkyTortoise/EnterpriseHub) -- Real estate AI platform with BI dashboards and CRM integration
- [insight-engine](https://github.com/ChunkyTortoise/insight-engine) -- Upload CSV/Excel, get instant dashboards, predictive models, and reports
- [ai-orchestrator](https://github.com/ChunkyTortoise/ai-orchestrator) -- AgentForge: unified async LLM interface (Claude, Gemini, OpenAI, Perplexity)
- [Revenue-Sprint](https://github.com/ChunkyTortoise/Revenue-Sprint) -- AI-powered freelance pipeline: job scanning, proposal generation, prompt injection testing
- [jorge_real_estate_bots](https://github.com/ChunkyTortoise/jorge_real_estate_bots) -- Three-bot lead qualification system (Lead, Buyer, Seller)

## License

MIT -- see [LICENSE](LICENSE) for details.
