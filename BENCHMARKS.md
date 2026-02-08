# DocQA Engine -- Benchmarks

Generated: 2026-02-08

## Retrieval Performance

Evaluated on 15 queries against 3 demo documents (Python guide, ML textbook, Startup playbook).

| Method | Precision@5 | Hit Rate@5 | Avg Latency |
|--------|-------------|------------|-------------|
| BM25 | 29.3% | 86.7% | 0.2ms |
| Dense (TF-IDF) | 28.0% | 86.7% | 0.2ms |
| Hybrid (BM25 + Dense + RRF) | 29.3% | 86.7% | 0.4ms |

**Methodology**: Keyword-based relevance judgment. A result is relevant if it contains at least one expected keyword. Precision@5 = relevant results in top 5 / 5. Hit Rate = queries with at least 1 relevant result in top 5.

**Embedder**: TF-IDF (5,000 features, scikit-learn). No external API calls required.

## Test Suite Summary

157 tests across 12 modules. All tests run without network access or external API keys.

| Module | Test File | Tests | Description |
|--------|-----------|-------|-------------|
| Ingest | `test_ingest.py` | ~14 | Document loading, format parsing |
| Chunking | `test_chunking.py` | ~14 | Fixed, sentence, semantic strategies |
| Embedder | `test_embedder.py` | ~12 | TF-IDF vectorization, feature limits |
| Retriever | `test_retriever.py` | ~14 | BM25, dense, hybrid RRF fusion |
| Answer | `test_answer.py` | ~12 | Context window, citations, mock LLM |
| Prompt Lab | `test_prompt_lab.py` | ~14 | Template versioning, A/B comparison |
| Citation Scorer | `test_citation_scorer.py` | ~14 | Faithfulness, coverage, redundancy |
| Evaluator | `test_evaluator.py` | ~14 | MRR, NDCG, Precision/Recall@K |
| Batch | `test_batch.py` | ~12 | Parallel ingestion, query batches |
| Exporter | `test_exporter.py` | ~12 | JSON/CSV export, metadata |
| Cost Tracker | `test_cost_tracker.py` | ~12 | Token counting, cost estimation |
| Pipeline | `test_pipeline.py` | ~13 | End-to-end integration |
| **Total** | **12 files** | **157** | |

## How to Reproduce

```bash
git clone https://github.com/ChunkyTortoise/docqa-engine.git
cd docqa-engine
pip install -r requirements.txt
make test
# or: python -m pytest tests/ -v

# Retrieval benchmarks
python benchmarks/run_benchmarks.py
```

## Notes

- All tests use mock LLM responses for reproducibility
- No external API calls or network access required
- Retrieval benchmarks use the 3 included demo documents
