"""Retrieval Benchmark Suite: systematic evaluation with reproducible benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class BenchmarkResult:
    method: str
    mrr: float
    ndcg_at_k: float
    precision_at_k: float
    recall_at_k: float
    latency_ms: float


@dataclass
class QAPair:
    query: str
    relevant_doc_ids: list[str]
    answer: str = ""


# Built-in synthetic benchmark dataset (50 QA pairs)
SYNTHETIC_QA_PAIRS: list[QAPair] = [
    QAPair("What is machine learning?", ["ml-1", "ml-2"], "ML is a subset of AI"),
    QAPair("How does gradient descent work?", ["ml-3"], "Iterative optimization"),
    QAPair("What is a neural network?", ["ml-4", "ml-5"], "Layers of neurons"),
    QAPair("Explain backpropagation", ["ml-6"], "Chain rule for gradients"),
    QAPair("What is overfitting?", ["ml-7"], "Model memorizes training data"),
    QAPair("How to prevent underfitting?", ["ml-8"], "Increase model complexity"),
    QAPair("What is cross-validation?", ["ml-9"], "K-fold data splitting"),
    QAPair("Explain regularization", ["ml-10"], "Penalize complex models"),
    QAPair("What is a decision tree?", ["ml-11", "ml-12"], "Tree-based classifier"),
    QAPair("How does random forest work?", ["ml-13"], "Ensemble of trees"),
    QAPair("What is natural language processing?", ["nlp-1"], "Text understanding by machines"),
    QAPair("How does tokenization work?", ["nlp-2"], "Splitting text into tokens"),
    QAPair("What is word embedding?", ["nlp-3", "nlp-4"], "Dense vector representation"),
    QAPair("Explain attention mechanism", ["nlp-5"], "Weighted focus on input parts"),
    QAPair("What is a transformer?", ["nlp-6"], "Self-attention architecture"),
    QAPair("How does BERT work?", ["nlp-7"], "Masked language model"),
    QAPair("What is sentiment analysis?", ["nlp-8"], "Detecting opinion polarity"),
    QAPair("Explain named entity recognition", ["nlp-9"], "Identifying entities in text"),
    QAPair("What is text classification?", ["nlp-10"], "Categorizing documents"),
    QAPair("How does TF-IDF work?", ["nlp-11", "nlp-12"], "Term frequency weighting"),
    QAPair("What is a database index?", ["db-1"], "Data structure for fast lookup"),
    QAPair("Explain SQL joins", ["db-2", "db-3"], "Combining table rows"),
    QAPair("What is database normalization?", ["db-4"], "Reducing data redundancy"),
    QAPair("How does ACID work?", ["db-5"], "Transaction guarantees"),
    QAPair("What is a primary key?", ["db-6"], "Unique row identifier"),
    QAPair("Explain database sharding", ["db-7"], "Horizontal data partitioning"),
    QAPair("What is an ORM?", ["db-8"], "Object-relational mapping"),
    QAPair("How does connection pooling work?", ["db-9"], "Reusing database connections"),
    QAPair("What is a NoSQL database?", ["db-10", "db-11"], "Non-relational data store"),
    QAPair("Explain CAP theorem", ["db-12"], "Consistency-availability tradeoff"),
    QAPair("What is REST API?", ["api-1"], "Resource-based web interface"),
    QAPair("How does OAuth work?", ["api-2", "api-3"], "Delegated authorization"),
    QAPair("What is rate limiting?", ["api-4"], "Request throttling"),
    QAPair("Explain API versioning", ["api-5"], "Backward-compatible changes"),
    QAPair("What is GraphQL?", ["api-6"], "Query language for APIs"),
    QAPair("How does caching work?", ["api-7"], "Storing computed results"),
    QAPair("What is a webhook?", ["api-8"], "Event-driven HTTP callback"),
    QAPair("Explain CORS", ["api-9"], "Cross-origin resource sharing"),
    QAPair("What is microservices architecture?", ["api-10", "api-11"], "Decomposed services"),
    QAPair("How does load balancing work?", ["api-12"], "Distributing traffic"),
    QAPair("What is Docker?", ["devops-1"], "Container runtime"),
    QAPair("How does Kubernetes work?", ["devops-2", "devops-3"], "Container orchestration"),
    QAPair("What is CI/CD?", ["devops-4"], "Automated build and deploy"),
    QAPair("Explain infrastructure as code", ["devops-5"], "Declarative infra management"),
    QAPair("What is observability?", ["devops-6"], "System behavior understanding"),
    QAPair("How does A/B testing work?", ["devops-7"], "Controlled experiments"),
    QAPair("What is feature flagging?", ["devops-8"], "Toggle features at runtime"),
    QAPair("Explain blue-green deployment", ["devops-9"], "Zero-downtime deploys"),
    QAPair("What is chaos engineering?", ["devops-10", "devops-11"], "Resilience testing"),
    QAPair("How does service mesh work?", ["devops-12"], "Network layer for microservices"),
]


class RetrievalFunction(Protocol):
    """Protocol for retrieval functions to benchmark."""

    def __call__(self, query: str) -> list[str]:
        """Return list of document IDs ranked by relevance."""
        ...


class BenchmarkSuite:
    """Run retrieval benchmarks with reproducible evaluation."""

    def __init__(self, qa_pairs: list[QAPair] | None = None, k: int = 5):
        self.qa_pairs = qa_pairs if qa_pairs is not None else SYNTHETIC_QA_PAIRS
        self.k = k

    def _reciprocal_rank(self, retrieved: list[str], relevant: list[str]) -> float:
        """Compute reciprocal rank of first relevant result."""
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def _precision_at_k(self, retrieved: list[str], relevant: list[str], k: int) -> float:
        """Compute precision@k."""
        if k <= 0:
            return 0.0
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for doc in top_k if doc in relevant_set)
        return hits / k

    def _recall_at_k(self, retrieved: list[str], relevant: list[str], k: int) -> float:
        """Compute recall@k."""
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for doc in top_k if doc in relevant_set)
        return hits / len(relevant)

    def _ndcg_at_k(self, retrieved: list[str], relevant: list[str], k: int) -> float:
        """Compute NDCG@k with binary relevance."""
        import math

        if k <= 0 or not relevant:
            return 0.0

        relevant_set = set(relevant)
        top_k = retrieved[:k]

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(top_k):
            if doc_id in relevant_set:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0

        # Ideal DCG (all relevant docs at top)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def run(
        self,
        retrieval_fn: RetrievalFunction,
        method_name: str = "default",
    ) -> BenchmarkResult:
        """Run the benchmark suite against a retrieval function."""
        mrr_scores: list[float] = []
        ndcg_scores: list[float] = []
        precision_scores: list[float] = []
        recall_scores: list[float] = []
        latencies: list[float] = []

        for qa in self.qa_pairs:
            start = time.perf_counter()
            retrieved = retrieval_fn(qa.query)
            elapsed_ms = (time.perf_counter() - start) * 1000

            latencies.append(elapsed_ms)
            mrr_scores.append(self._reciprocal_rank(retrieved, qa.relevant_doc_ids))
            ndcg_scores.append(self._ndcg_at_k(retrieved, qa.relevant_doc_ids, self.k))
            precision_scores.append(
                self._precision_at_k(retrieved, qa.relevant_doc_ids, self.k)
            )
            recall_scores.append(
                self._recall_at_k(retrieved, qa.relevant_doc_ids, self.k)
            )

        n = len(self.qa_pairs) or 1

        return BenchmarkResult(
            method=method_name,
            mrr=sum(mrr_scores) / n,
            ndcg_at_k=sum(ndcg_scores) / n,
            precision_at_k=sum(precision_scores) / n,
            recall_at_k=sum(recall_scores) / n,
            latency_ms=sum(latencies) / n,
        )


class BenchmarkRegistry:
    """Register, list, and compare benchmarks. Track results over time."""

    def __init__(self) -> None:
        self._results: list[BenchmarkResult] = []
        self._baselines: dict[str, BenchmarkResult] = {}

    def register(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the registry."""
        self._results.append(result)

    def set_baseline(self, result: BenchmarkResult) -> None:
        """Set a baseline for regression detection."""
        self._baselines[result.method] = result

    def list_results(self) -> list[BenchmarkResult]:
        """Return all registered results."""
        return list(self._results)

    def best_by(self, metric: str) -> BenchmarkResult | None:
        """Return the result with the highest value for the given metric."""
        if not self._results:
            return None
        valid = [r for r in self._results if hasattr(r, metric)]
        if not valid:
            return None
        return max(valid, key=lambda r: getattr(r, metric))

    def detect_regression(
        self,
        result: BenchmarkResult,
        threshold: float = 0.05,
    ) -> dict[str, float]:
        """Compare a result against its baseline.

        Returns a dict of metrics that regressed beyond the threshold.
        Positive values mean the metric dropped by that amount.
        """
        baseline = self._baselines.get(result.method)
        if baseline is None:
            return {}

        regressions: dict[str, float] = {}
        for metric in ("mrr", "ndcg_at_k", "precision_at_k", "recall_at_k"):
            baseline_val = getattr(baseline, metric)
            result_val = getattr(result, metric)
            drop = baseline_val - result_val
            if drop > threshold:
                regressions[metric] = round(drop, 4)

        return regressions

    def compare(
        self,
        result_a: BenchmarkResult,
        result_b: BenchmarkResult,
    ) -> dict[str, float]:
        """Compare two results. Positive values mean A is better."""
        deltas: dict[str, float] = {}
        for metric in ("mrr", "ndcg_at_k", "precision_at_k", "recall_at_k", "latency_ms"):
            val_a = getattr(result_a, metric)
            val_b = getattr(result_b, metric)
            if metric == "latency_ms":
                # Lower is better for latency
                deltas[metric] = round(val_b - val_a, 4)
            else:
                deltas[metric] = round(val_a - val_b, 4)
        return deltas

    def summary(self) -> list[dict[str, object]]:
        """Return a summary of all registered results."""
        return [
            {
                "method": r.method,
                "mrr": round(r.mrr, 4),
                "ndcg@k": round(r.ndcg_at_k, 4),
                "p@k": round(r.precision_at_k, 4),
                "r@k": round(r.recall_at_k, 4),
                "latency_ms": round(r.latency_ms, 2),
            }
            for r in self._results
        ]
