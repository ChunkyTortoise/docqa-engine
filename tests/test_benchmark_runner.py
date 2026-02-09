"""Tests for retrieval benchmark suite: metrics, registry, and regression detection."""

from __future__ import annotations

import pytest

from docqa_engine.benchmark_runner import (
    SYNTHETIC_QA_PAIRS,
    BenchmarkRegistry,
    BenchmarkResult,
    BenchmarkSuite,
    QAPair,
)

# ---------------------------------------------------------------------------
# QAPair and synthetic data
# ---------------------------------------------------------------------------


class TestQAPair:
    """Tests for QAPair dataclass and built-in dataset."""

    def test_qa_pair_fields(self) -> None:
        qa = QAPair(query="What is ML?", relevant_doc_ids=["ml-1"], answer="Machine learning")
        assert qa.query == "What is ML?"
        assert qa.relevant_doc_ids == ["ml-1"]
        assert qa.answer == "Machine learning"

    def test_synthetic_dataset_has_50_pairs(self) -> None:
        assert len(SYNTHETIC_QA_PAIRS) == 50

    def test_all_pairs_have_queries(self) -> None:
        for qa in SYNTHETIC_QA_PAIRS:
            assert qa.query
            assert len(qa.relevant_doc_ids) >= 1


# ---------------------------------------------------------------------------
# BenchmarkSuite — metric calculations
# ---------------------------------------------------------------------------


@pytest.fixture()
def suite() -> BenchmarkSuite:
    pairs = [
        QAPair("query-1", ["doc-a", "doc-b"]),
        QAPair("query-2", ["doc-c"]),
        QAPair("query-3", ["doc-d", "doc-e", "doc-f"]),
    ]
    return BenchmarkSuite(qa_pairs=pairs, k=3)


class TestBenchmarkMetrics:
    """Tests for individual IR metrics."""

    def test_reciprocal_rank_first(self, suite: BenchmarkSuite) -> None:
        assert suite._reciprocal_rank(["doc-a", "doc-x", "doc-y"], ["doc-a"]) == 1.0

    def test_reciprocal_rank_second(self, suite: BenchmarkSuite) -> None:
        assert suite._reciprocal_rank(["doc-x", "doc-a", "doc-y"], ["doc-a"]) == 0.5

    def test_reciprocal_rank_not_found(self, suite: BenchmarkSuite) -> None:
        assert suite._reciprocal_rank(["doc-x", "doc-y"], ["doc-a"]) == 0.0

    def test_precision_at_k(self, suite: BenchmarkSuite) -> None:
        p = suite._precision_at_k(["doc-a", "doc-b", "doc-x"], ["doc-a", "doc-b"], k=3)
        assert abs(p - 2 / 3) < 1e-6

    def test_precision_at_k_zero(self, suite: BenchmarkSuite) -> None:
        assert suite._precision_at_k(["doc-x"], ["doc-a"], k=0) == 0.0

    def test_recall_at_k(self, suite: BenchmarkSuite) -> None:
        r = suite._recall_at_k(["doc-a", "doc-x"], ["doc-a", "doc-b"], k=2)
        assert abs(r - 0.5) < 1e-6

    def test_recall_at_k_empty_relevant(self, suite: BenchmarkSuite) -> None:
        assert suite._recall_at_k(["doc-a"], [], k=1) == 0.0

    def test_ndcg_at_k_perfect(self, suite: BenchmarkSuite) -> None:
        ndcg = suite._ndcg_at_k(["doc-a", "doc-b"], ["doc-a", "doc-b"], k=2)
        assert abs(ndcg - 1.0) < 1e-6

    def test_ndcg_at_k_no_relevant(self, suite: BenchmarkSuite) -> None:
        assert suite._ndcg_at_k(["doc-x", "doc-y"], ["doc-a"], k=2) == 0.0

    def test_ndcg_at_k_empty_relevant(self, suite: BenchmarkSuite) -> None:
        assert suite._ndcg_at_k(["doc-x"], [], k=1) == 0.0


# ---------------------------------------------------------------------------
# BenchmarkSuite.run
# ---------------------------------------------------------------------------


def _perfect_retriever(query: str) -> list[str]:
    """Returns the exact relevant docs for our test QA pairs."""
    mapping = {
        "query-1": ["doc-a", "doc-b"],
        "query-2": ["doc-c"],
        "query-3": ["doc-d", "doc-e", "doc-f"],
    }
    return mapping.get(query, [])


def _empty_retriever(query: str) -> list[str]:
    return []


class TestBenchmarkRun:
    """Tests for running a full benchmark."""

    def test_perfect_retrieval(self, suite: BenchmarkSuite) -> None:
        result = suite.run(_perfect_retriever, method_name="perfect")
        assert result.method == "perfect"
        assert result.mrr == 1.0
        assert result.precision_at_k > 0.0
        assert result.recall_at_k == 1.0
        assert result.latency_ms >= 0.0

    def test_empty_retrieval(self, suite: BenchmarkSuite) -> None:
        result = suite.run(_empty_retriever, method_name="empty")
        assert result.mrr == 0.0
        assert result.precision_at_k == 0.0
        assert result.recall_at_k == 0.0

    def test_latency_measured(self, suite: BenchmarkSuite) -> None:
        result = suite.run(_perfect_retriever)
        assert result.latency_ms >= 0.0

    def test_result_is_dataclass(self, suite: BenchmarkSuite) -> None:
        result = suite.run(_perfect_retriever)
        assert isinstance(result, BenchmarkResult)


# ---------------------------------------------------------------------------
# BenchmarkRegistry
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> BenchmarkRegistry:
    return BenchmarkRegistry()


@pytest.fixture()
def sample_result() -> BenchmarkResult:
    return BenchmarkResult(
        method="bm25", mrr=0.8, ndcg_at_k=0.75, precision_at_k=0.6, recall_at_k=0.7, latency_ms=5.0
    )


class TestBenchmarkRegistry:
    """Tests for result tracking and comparison."""

    def test_register_and_list(self, registry: BenchmarkRegistry, sample_result: BenchmarkResult) -> None:
        registry.register(sample_result)
        assert len(registry.list_results()) == 1

    def test_best_by_metric(self, registry: BenchmarkRegistry) -> None:
        r1 = BenchmarkResult("a", mrr=0.5, ndcg_at_k=0.4, precision_at_k=0.3, recall_at_k=0.2, latency_ms=10)
        r2 = BenchmarkResult("b", mrr=0.9, ndcg_at_k=0.8, precision_at_k=0.7, recall_at_k=0.6, latency_ms=5)
        registry.register(r1)
        registry.register(r2)
        best = registry.best_by("mrr")
        assert best is not None
        assert best.method == "b"

    def test_best_by_empty(self, registry: BenchmarkRegistry) -> None:
        assert registry.best_by("mrr") is None

    def test_set_baseline_and_detect_regression(self, registry: BenchmarkRegistry) -> None:
        baseline = BenchmarkResult("bm25", mrr=0.9, ndcg_at_k=0.85, precision_at_k=0.8, recall_at_k=0.75, latency_ms=5)
        registry.set_baseline(baseline)

        worse = BenchmarkResult("bm25", mrr=0.7, ndcg_at_k=0.65, precision_at_k=0.6, recall_at_k=0.55, latency_ms=8)
        regressions = registry.detect_regression(worse)
        assert "mrr" in regressions
        assert regressions["mrr"] > 0.05

    def test_no_regression_within_threshold(self, registry: BenchmarkRegistry) -> None:
        baseline = BenchmarkResult("bm25", mrr=0.9, ndcg_at_k=0.85, precision_at_k=0.8, recall_at_k=0.75, latency_ms=5)
        registry.set_baseline(baseline)

        similar = BenchmarkResult("bm25", mrr=0.88, ndcg_at_k=0.83, precision_at_k=0.78, recall_at_k=0.73, latency_ms=5)
        regressions = registry.detect_regression(similar)
        assert len(regressions) == 0

    def test_no_baseline_returns_empty(self, registry: BenchmarkRegistry, sample_result: BenchmarkResult) -> None:
        regressions = registry.detect_regression(sample_result)
        assert regressions == {}

    def test_compare_results(self, registry: BenchmarkRegistry) -> None:
        r1 = BenchmarkResult("a", mrr=0.9, ndcg_at_k=0.8, precision_at_k=0.7, recall_at_k=0.6, latency_ms=5)
        r2 = BenchmarkResult("b", mrr=0.7, ndcg_at_k=0.6, precision_at_k=0.5, recall_at_k=0.4, latency_ms=10)
        deltas = registry.compare(r1, r2)
        assert deltas["mrr"] > 0  # A is better
        assert deltas["latency_ms"] > 0  # A is faster (lower latency = positive delta)

    def test_summary(self, registry: BenchmarkRegistry, sample_result: BenchmarkResult) -> None:
        registry.register(sample_result)
        s = registry.summary()
        assert len(s) == 1
        assert s[0]["method"] == "bm25"
        assert "mrr" in s[0]
