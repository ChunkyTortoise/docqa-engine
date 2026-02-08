"""Retrieval Benchmarks: BM25 vs Dense vs Hybrid on demo documents.

Run: python benchmarks/run_benchmarks.py
Output: Markdown table to stdout + BENCHMARKS.md
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docqa_engine.ingest import ingest_file
from docqa_engine.retriever import BM25Index, DenseIndex, HybridRetriever, SearchResult
from docqa_engine.embedder import TfidfEmbedder


# Ground truth: (question, expected_chunk_keywords)
BENCHMARK_QUERIES = [
    ("What is a Python decorator?", ["decorator", "function", "wrap"]),
    ("How do you handle exceptions in Python?", ["exception", "try", "except", "error"]),
    ("What is gradient descent?", ["gradient", "descent", "optimization", "loss"]),
    ("Explain overfitting in machine learning", ["overfitting", "training", "generalization"]),
    ("What is a neural network?", ["neural", "network", "layer", "neuron"]),
    ("How do you read a file in Python?", ["file", "open", "read"]),
    ("What is cross-validation?", ["cross", "validation", "fold", "training"]),
    ("Explain list comprehension in Python", ["list", "comprehension", "expression"]),
    ("What is product-market fit?", ["product", "market", "fit", "customer"]),
    ("How to raise funding for a startup?", ["funding", "investor", "raise", "capital"]),
    ("What is regularization?", ["regularization", "l1", "l2", "penalty"]),
    ("How do Python generators work?", ["generator", "yield", "iterator"]),
    ("What is backpropagation?", ["backpropagation", "gradient", "chain", "rule"]),
    ("Explain Python virtual environments", ["virtual", "environment", "venv", "pip"]),
    ("What is a confusion matrix?", ["confusion", "matrix", "precision", "recall"]),
]


def keyword_hit(result: SearchResult, keywords: list[str]) -> bool:
    """Check if at least one keyword appears in the chunk content."""
    content_lower = result.chunk.content.lower()
    return any(kw.lower() in content_lower for kw in keywords)


def evaluate_results(
    results: list[SearchResult], keywords: list[str], k: int = 5
) -> dict:
    """Calculate precision@k and hit@k for a single query."""
    top_k = results[:k]
    hits = sum(1 for r in top_k if keyword_hit(r, keywords))
    hit_at_k = 1 if hits > 0 else 0
    precision = hits / k if k > 0 else 0.0
    return {"precision": precision, "hit": hit_at_k, "hits": hits}


def run_bm25_benchmark(all_chunks, queries, k=5):
    """Benchmark BM25-only retrieval."""
    index = BM25Index()
    index.add_chunks(all_chunks)

    metrics = []
    total_time = 0
    for query, keywords in queries:
        start = time.perf_counter()
        results = index.search(query, top_k=k)
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed
        m = evaluate_results(results, keywords, k)
        m["time_ms"] = elapsed
        metrics.append(m)

    return {
        "precision_at_k": sum(m["precision"] for m in metrics) / len(metrics),
        "hit_rate": sum(m["hit"] for m in metrics) / len(metrics),
        "avg_time_ms": total_time / len(metrics),
        "method": "BM25",
    }


def run_dense_benchmark(all_chunks, embedder, queries, k=5):
    """Benchmark Dense-only retrieval."""
    texts = [c.content for c in all_chunks]
    embeddings = embedder.embed(texts)

    index = DenseIndex()
    index.add_chunks(all_chunks, embeddings)

    metrics = []
    total_time = 0
    for query, keywords in queries:
        start = time.perf_counter()
        query_emb = embedder.embed_query(query)
        results = index.search(query_emb, top_k=k)
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed
        m = evaluate_results(results, keywords, k)
        m["time_ms"] = elapsed
        metrics.append(m)

    return {
        "precision_at_k": sum(m["precision"] for m in metrics) / len(metrics),
        "hit_rate": sum(m["hit"] for m in metrics) / len(metrics),
        "avg_time_ms": total_time / len(metrics),
        "method": "Dense (TF-IDF)",
    }


async def run_hybrid_benchmark(all_chunks, embedder, queries, k=5):
    """Benchmark Hybrid (BM25 + Dense + RRF) retrieval."""
    texts = [c.content for c in all_chunks]
    embeddings = embedder.embed(texts)

    async def embed_fn(txts):
        return embedder.embed(txts)

    retriever = HybridRetriever(embed_fn=embed_fn)
    retriever.add_chunks(all_chunks, embeddings)

    metrics = []
    total_time = 0
    for query, keywords in queries:
        start = time.perf_counter()
        results = await retriever.search(query, top_k=k)
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed
        m = evaluate_results(results, keywords, k)
        m["time_ms"] = elapsed
        metrics.append(m)

    return {
        "precision_at_k": sum(m["precision"] for m in metrics) / len(metrics),
        "hit_rate": sum(m["hit"] for m in metrics) / len(metrics),
        "avg_time_ms": total_time / len(metrics),
        "method": "Hybrid (BM25 + Dense + RRF)",
    }


def format_table(results: list[dict], k: int) -> str:
    """Format results as a markdown table."""
    lines = [
        f"# DocQA Engine Retrieval Benchmarks",
        f"",
        f"Evaluated on {len(BENCHMARK_QUERIES)} queries against 3 demo documents (Python guide, ML textbook, Startup playbook).",
        f"",
        f"| Method | Precision@{k} | Hit Rate@{k} | Avg Latency |",
        f"|--------|{'---' * 5}|{'---' * 5}|{'---' * 5}|",
    ]
    for r in results:
        lines.append(
            f"| {r['method']} | {r['precision_at_k']:.1%} | {r['hit_rate']:.1%} | {r['avg_time_ms']:.1f}ms |"
        )
    lines.extend([
        f"",
        f"**Methodology**: Keyword-based relevance judgment. A result is relevant if it contains at least one expected keyword. "
        f"Precision@{k} = relevant results in top {k} / {k}. Hit Rate = queries with at least 1 relevant result in top {k}.",
        f"",
        f"**Embedder**: TF-IDF (5000 features, scikit-learn). No external API calls required.",
        f"",
        f"Generated by `benchmarks/run_benchmarks.py`.",
    ])
    return "\n".join(lines)


async def main():
    demo_dir = Path(__file__).resolve().parent.parent / "demo_docs"
    if not demo_dir.exists():
        print(f"Error: demo_docs directory not found at {demo_dir}", file=sys.stderr)
        sys.exit(1)

    # Ingest all demo documents
    all_chunks = []
    for doc_file in sorted(demo_dir.glob("*.md")):
        result = ingest_file(str(doc_file))
        all_chunks.extend(result.chunks)
        print(f"Ingested {doc_file.name}: {len(result.chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Queries: {len(BENCHMARK_QUERIES)}")
    print(f"---\n")

    # Fit embedder on all chunk texts
    embedder = TfidfEmbedder(max_features=5000)
    embedder.fit([c.content for c in all_chunks])

    # Run benchmarks
    k = 5
    bm25_results = run_bm25_benchmark(all_chunks, BENCHMARK_QUERIES, k)
    dense_results = run_dense_benchmark(all_chunks, embedder, BENCHMARK_QUERIES, k)
    hybrid_results = await run_hybrid_benchmark(all_chunks, embedder, BENCHMARK_QUERIES, k)

    all_results = [bm25_results, dense_results, hybrid_results]

    # Print to stdout
    table = format_table(all_results, k)
    print(table)

    # Write BENCHMARKS.md
    output_path = Path(__file__).resolve().parent.parent / "BENCHMARKS.md"
    output_path.write_text(table + "\n")
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
