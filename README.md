# DocQA Engine

**Teams waste hours searching through PDFs and docs for answers.** Upload your documents and get instant cited answers with a prompt engineering lab for optimizing your Q&A pipeline.

[![CI](https://img.shields.io/github/actions/workflow/status/ChunkyTortoise/docqa-engine/ci.yml?label=CI)](https://github.com/ChunkyTortoise/docqa-engine/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F1C40F.svg)](LICENSE)

## What This Solves

- **Document search is manual and slow** -- Hybrid retrieval (BM25 keyword + dense vectors) finds the right passages in seconds, with Reciprocal Rank Fusion combining both approaches
- **Answers lack sources** -- Every answer includes page numbers and highlighted passages from the source documents, so you can verify claims
- **Prompt quality is a black box** -- Built-in prompt engineering lab lets you version prompts, A/B test configurations, and score answers on faithfulness, relevance, and completeness

## Architecture

```
Document Upload (PDF, DOCX, TXT, CSV)
      |
      v
┌─────────────┐    ┌───────────────┐    ┌──────────────┐
│  Ingestion   │───>│  Hybrid       │───>│  Answer Gen  │
│  (chunk +    │    │  Retrieval    │    │  (LLM + cite │
│   metadata)  │    │  (BM25 + RRF) │    │   sources)   │
└──────────────┘    └───────────────┘    └──────────────┘
                                                |
                    ┌───────────────┐    ┌──────┴───────┐
                    │  Prompt Lab   │    │ Cost Tracker  │
                    │  (A/B test,   │    │ (per-query,   │
                    │   versioning) │    │  by provider) │
                    └───────────────┘    └──────────────┘
```

## Quick Start

```bash
git clone https://github.com/ChunkyTortoise/docqa-engine.git
cd docqa-engine
pip install -r requirements.txt

# Demo mode -- pre-loaded sample documents, no API keys needed
make demo
```

### What You Get

1. **Document Ingestion** -- PDF, DOCX, TXT, CSV with configurable chunking and sentence-boundary splitting
2. **Hybrid Retrieval** -- BM25 keyword search + dense vector search with Reciprocal Rank Fusion
3. **Cited Answers** -- Every answer includes page numbers and source passages
4. **Prompt Engineering Lab** -- Version prompts, configure temperature/tokens, A/B test, score on faithfulness/relevance/completeness
5. **Cost Tracking** -- Per-query token usage and cumulative cost by provider
6. **Mock Mode** -- Pre-indexed demo docs, works with zero API keys

## Demo Documents

| Document | Content | Questions to Try |
|----------|---------|-----------------|
| Company FAQ | TechCorp pricing, onboarding, security, SLAs | "What is the enterprise SLA?" |
| Employee Handbook | Remote work, PTO, reviews, professional development | "How much PTO do I get after 5 years?" |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Retrieval | BM25 (custom), NumPy (dense), Reciprocal Rank Fusion |
| Ingestion | PyPDF2, python-docx |
| ML | NumPy (cosine similarity) |
| Testing | pytest |
| CI | GitHub Actions (Python 3.11, 3.12) |

## Project Structure

```
docqa-engine/
├── app.py                          # Streamlit application
├── docqa_engine/
│   ├── ingest.py                   # PDF/DOCX/TXT/CSV ingestion + chunking
│   ├── retriever.py                # BM25, dense index, hybrid + RRF
│   ├── answer.py                   # Context building, citations, LLM answer gen
│   ├── prompt_lab.py               # Prompt versioning, A/B testing, eval scoring
│   └── cost_tracker.py             # Per-query cost tracking by provider
├── demo_data/
│   ├── sample_faq.txt              # Company FAQ demo document
│   └── sample_policy.txt           # Employee handbook demo document
├── tests/                          # One test file per module
├── .github/workflows/ci.yml        # CI pipeline
├── Makefile                        # demo, test, lint, setup
└── requirements.txt
```

## Testing

```bash
make test                           # Full suite
python -m pytest tests/ -v          # Verbose output
python -m pytest tests/test_retriever.py  # Single module
```

## Related Projects

- [ai-orchestrator](https://github.com/ChunkyTortoise/ai-orchestrator) -- AgentForge: unified async LLM interface (used as optional multi-LLM backend)
- [EnterpriseHub](https://github.com/ChunkyTortoise/EnterpriseHub) -- Real estate AI platform with advanced RAG system
- [insight-engine](https://github.com/ChunkyTortoise/insight-engine) -- Data analytics platform: auto-profiling, dashboards, predictive modeling
- [jorge_real_estate_bots](https://github.com/ChunkyTortoise/jorge_real_estate_bots) -- Three-bot lead qualification system
- [Revenue-Sprint](https://github.com/ChunkyTortoise/Revenue-Sprint) -- AI-powered freelance pipeline

## License

MIT -- see [LICENSE](LICENSE) for details.
