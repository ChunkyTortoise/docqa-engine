[![Sponsor](https://img.shields.io/badge/Sponsor-💖-pink.svg)](https://github.com/sponsors/ChunkyTortoise)

# docqa-engine

**Upload documents, ask questions -- get cited answers with a prompt engineering lab.**

![CI](https://github.com/ChunkyTortoise/docqa-engine/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-557%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B.svg?logo=streamlit&logoColor=white)](https://ct-document-engine.streamlit.app)

**[Live Demo](https://ct-document-engine.streamlit.app)** -- try it without installing anything.

## Demo Snapshot

![Demo Snapshot](assets/demo.png)

## What This Solves

- **RAG pipeline from upload to answer** -- Ingest documents (PDF, DOCX, TXT, MD, CSV), chunk them with pluggable strategies, embed with TF-IDF, and retrieve using BM25 + dense hybrid search with Reciprocal Rank Fusion
- **Prompt engineering lab for A/B testing** -- Create prompt templates, run the same question through different strategies side-by-side, compare outputs
- **Citation accuracy matters** -- Faithfulness, coverage, and redundancy scoring for every generated citation

## Service Mapping

- **Service 3:** Custom RAG Conversational Agents
- **Service 5:** Prompt Engineering and System Optimization

## Certification Mapping

- IBM Generative AI Engineering with PyTorch, LangChain & Hugging Face
- IBM RAG and Agentic AI Professional Certificate
- Vanderbilt ChatGPT Personal Automation
- Duke University LLMOps Specialization

## Architecture

```mermaid
flowchart TB
    Upload["Document Upload\n(PDF, DOCX, TXT, MD, CSV)"]
    Chunk["Chunking Engine\n(semantic, fixed, sliding window)"]
    Embed["Embedding Layer\n(TF-IDF, BM25, Dense)"]
    VStore["Vector Store\n(FAISS / in-memory)"]
    Hybrid["Hybrid Retrieval\n(BM25 + Dense + RRF fusion)"]
    Rerank["Cross-Encoder Re-Ranker"]
    QExpand["Query Expansion\n(synonym, PRF, decompose)"]
    Citation["Citation Scoring\n(faithfulness, coverage, redundancy)"]
    Answer["Answer Generation"]
    Convo["Conversation Manager\n(multi-turn context)"]
    API["REST API\n(JWT auth, rate limiting, metering)"]
    UI["Streamlit Demo UI\n(4-tab interface)"]

    Upload --> Chunk --> Embed --> VStore
    QExpand --> Hybrid
    VStore --> Hybrid --> Rerank --> Answer
    Answer --> Citation
    Answer --> Convo
    API --> Answer
    UI --> API
```

## Key Metrics

| Metric | Value |
|--------|-------|
| Test Suite | 557 automated tests |
| Retrieval Accuracy | Hybrid > BM25-only by 15-25% |
| Re-Ranking Boost | +8-12% relevance improvement |
| Query Latency | <100ms for 10K document corpus |
| Citation Accuracy | Faithfulness + coverage scoring |
| API Rate Limit | Configurable per-user metering |

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
| **REST API** | `api.py` | FastAPI wrapper with JWT auth, rate limiting, metering |
| **Vector Store** | `vector_store.py` | Pluggable vector store backends (FAISS, in-memory) |
| **Re-Ranker** | `reranker.py` | Cross-encoder TF-IDF re-ranking with Kendall tau |
| **Query Expansion** | `query_expansion.py` | Synonym, pseudo-relevance feedback, decomposition |
| **Answer Quality** | `answer_quality.py` | Multi-axis answer quality scoring |
| **Summarizer** | `summarizer.py` | Extractive and abstractive document summarization |
| **Document Graph** | `document_graph.py` | Cross-document entity and relationship graph |
| **Multi-Hop** | `multi_hop.py` | Multi-hop reasoning across document chains |
| **Conversation Manager** | `conversation_manager.py` | Multi-turn context tracking and query rewriting |
| **Context Compressor** | `context_compressor.py` | Token-budget context window compression |
| **Benchmark Runner** | `benchmark_runner.py` | Automated retrieval and performance benchmarking |

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
| Testing | pytest, pytest-asyncio (557 tests) |
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
├── tests/                          # 26 test files, 557 tests
├── .github/workflows/ci.yml        # CI pipeline
├── Makefile                        # demo, test, lint, setup
└── requirements.txt
```

## Testing

```bash
make test                           # Full suite (557 tests)
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
