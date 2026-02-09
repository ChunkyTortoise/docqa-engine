# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Mermaid architecture diagram in README
- Dockerfile and docker-compose.yml for containerized deployment
- Architecture Decision Records (docs/adr/)
- SECURITY.md, CODE_OF_CONDUCT.md, CHANGELOG.md governance files
- Enhanced benchmark suite with re-ranker, citation, and query expansion timing

## [0.4.0] - 2026-02-08

### Added
- Document summarizer with extractive and abstractive modes
- Multi-hop reasoning engine for cross-document question answering
- Conversation manager with multi-turn context tracking and query rewriting
- Document graph for entity and relationship extraction across documents
- Context compressor for token-budget-aware context window management
- Benchmark runner module for automated performance evaluation
- 76 new tests (summarizer, multi-hop, conversation manager, document graph, context compressor, benchmark runner)

## [0.3.0] - 2026-02-07

### Added
- Cross-encoder TF-IDF re-ranker with Kendall tau rank correlation
- Cascade multi-stage re-ranking pipeline
- Query expansion: synonym dictionary, pseudo-relevance feedback (PRF), query decomposition
- Answer quality scoring with multi-axis evaluation
- 60 new tests (re-ranker, query expansion, answer quality)

## [0.2.0] - 2026-02-06

### Added
- REST API wrapper (FastAPI) with JWT/API-key authentication
- Per-user sliding-window rate limiting (configurable, default 100 req/60s)
- Usage metering with per-key request and token tracking
- `/ingest`, `/ask`, `/stats`, `/reset` endpoints
- Pluggable vector store backends (FAISS, in-memory, ChromaDB-compatible)
- Client demo with httpx
- 20 new tests (API auth, rate limiting, metering, vector store backends)

## [0.1.0] - 2026-02-05

### Added
- Multi-format document ingestion (PDF, DOCX, TXT, MD, CSV)
- Pluggable chunking strategies: fixed-size, sentence-boundary, semantic
- TF-IDF embedding (5,000 features, scikit-learn, zero external API cost)
- BM25 (Okapi) retrieval with term-frequency scoring
- Dense retrieval with TF-IDF cosine similarity
- Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- Context-aware answer generation with source citations
- Prompt versioning and A/B comparison framework
- Citation accuracy scoring (faithfulness, coverage, redundancy)
- Retrieval evaluation metrics (MRR, NDCG@K, Precision@K, Recall@K, Hit Rate)
- Parallel batch ingestion and query processing
- JSON/CSV export for results and metrics
- Per-query token and cost tracking
- End-to-end DocQAPipeline class
- Streamlit demo UI with 4-tab interface
- GitHub Actions CI (Python 3.11, 3.12)
- 157 tests across 12 modules
