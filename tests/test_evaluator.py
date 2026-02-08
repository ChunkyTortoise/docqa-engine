"""Tests for retrieval evaluation metrics (MRR, NDCG, Precision, Recall, Hit Rate)."""

from __future__ import annotations

import pytest

from docqa_engine.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def evaluator() -> Evaluator:
    return Evaluator()


# ---------------------------------------------------------------------------
# Single-query evaluation
# ---------------------------------------------------------------------------


class TestEvaluateSingle:
    """Tests for evaluate_single on individual query results."""

    def test_perfect_ranking(self, evaluator: Evaluator) -> None:
        """All relevant docs ranked at the top in ideal order."""
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = {"d1", "d2", "d3"}
        result = evaluator.evaluate_single(retrieved, relevant, k=5)

        assert result["mrr"] == 1.0
        assert result["precision"] == pytest.approx(3 / 5)
        assert result["recall"] == 1.0
        assert result["hit_rate"] == 1.0
        assert result["ndcg"] == pytest.approx(1.0)

    def test_worst_ranking(self, evaluator: Evaluator) -> None:
        """No relevant docs in the retrieved set."""
        retrieved = ["x1", "x2", "x3", "x4", "x5"]
        relevant = {"d1", "d2"}
        result = evaluator.evaluate_single(retrieved, relevant, k=5)

        assert result["mrr"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["hit_rate"] == 0.0
        assert result["ndcg"] == 0.0

    def test_partial_match(self, evaluator: Evaluator) -> None:
        """Some relevant docs appear but not at the top."""
        retrieved = ["x1", "d1", "x2", "d2", "x3"]
        relevant = {"d1", "d2", "d3"}
        result = evaluator.evaluate_single(retrieved, relevant, k=5)

        # d1 is at rank 2 -> MRR = 1/2
        assert result["mrr"] == pytest.approx(0.5)
        assert result["precision"] == pytest.approx(2 / 5)
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["hit_rate"] == 1.0
        assert result["ndcg"] > 0.0

    def test_empty_retrieved(self, evaluator: Evaluator) -> None:
        """No documents were retrieved."""
        retrieved: list[str] = []
        relevant = {"d1", "d2"}
        result = evaluator.evaluate_single(retrieved, relevant, k=5)

        assert result["mrr"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["hit_rate"] == 0.0
        assert result["ndcg"] == 0.0

    def test_no_relevant_docs(self, evaluator: Evaluator) -> None:
        """The relevant set is empty (no ground truth)."""
        retrieved = ["d1", "d2", "d3"]
        relevant: set[str] = set()
        result = evaluator.evaluate_single(retrieved, relevant, k=5)

        assert result["mrr"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["hit_rate"] == 0.0
        assert result["ndcg"] == 0.0

    def test_single_relevant_at_top(self, evaluator: Evaluator) -> None:
        """Only one relevant document, and it is the first result."""
        retrieved = ["d1", "x1", "x2"]
        relevant = {"d1"}
        result = evaluator.evaluate_single(retrieved, relevant, k=3)

        assert result["mrr"] == 1.0
        assert result["precision"] == pytest.approx(1 / 3)
        assert result["recall"] == 1.0
        assert result["hit_rate"] == 1.0
        assert result["ndcg"] == pytest.approx(1.0)

    def test_single_relevant_at_end(self, evaluator: Evaluator) -> None:
        """Only one relevant document, and it is the last in top-K."""
        retrieved = ["x1", "x2", "d1"]
        relevant = {"d1"}
        result = evaluator.evaluate_single(retrieved, relevant, k=3)

        # d1 at rank 3 -> MRR = 1/3
        assert result["mrr"] == pytest.approx(1 / 3)
        assert result["precision"] == pytest.approx(1 / 3)
        assert result["recall"] == 1.0
        assert result["hit_rate"] == 1.0

    def test_k_smaller_than_retrieved(self, evaluator: Evaluator) -> None:
        """K truncates the retrieved list."""
        retrieved = ["x1", "x2", "d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        result = evaluator.evaluate_single(retrieved, relevant, k=2)

        # Only top-2 considered: x1, x2 — no relevant docs
        assert result["mrr"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["hit_rate"] == 0.0

    def test_k_larger_than_retrieved(self, evaluator: Evaluator) -> None:
        """K exceeds the length of retrieved docs — should not error."""
        retrieved = ["d1", "d2"]
        relevant = {"d1", "d2", "d3"}
        result = evaluator.evaluate_single(retrieved, relevant, k=10)

        assert result["mrr"] == 1.0
        assert result["precision"] == pytest.approx(2 / 2)  # 2 out of 2 retrieved
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["hit_rate"] == 1.0


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


class TestEvaluateBatch:
    """Tests for evaluate over multiple queries."""

    def test_average_across_queries(self, evaluator: Evaluator) -> None:
        """Metrics are averaged across queries."""
        queries = ["q1", "q2"]
        retrieved_docs = [
            ["d1", "d2", "d3"],  # q1: perfect — d1 relevant, at rank 1
            ["x1", "x2", "x3"],  # q2: worst — no relevant docs
        ]
        relevant_docs = [{"d1"}, {"d1"}]
        result = evaluator.evaluate(queries, retrieved_docs, relevant_docs, k=3)

        # q1: mrr=1.0, q2: mrr=0.0 -> average=0.5
        assert result["mrr"] == pytest.approx(0.5)
        # q1: hit_rate=1.0, q2: hit_rate=0.0 -> average=0.5
        assert result["hit_rate"] == pytest.approx(0.5)

    def test_empty_queries(self, evaluator: Evaluator) -> None:
        """No queries returns all-zero metrics."""
        result = evaluator.evaluate([], [], [], k=5)
        assert result["mrr"] == 0.0
        assert result["ndcg"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["hit_rate"] == 0.0

    def test_length_mismatch_raises(self, evaluator: Evaluator) -> None:
        """Mismatched input lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluator.evaluate(
                queries=["q1", "q2"],
                retrieved_docs=[["d1"]],
                relevant_docs=[{"d1"}, {"d2"}],
                k=5,
            )
