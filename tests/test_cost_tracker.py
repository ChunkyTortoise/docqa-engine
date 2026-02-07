"""Tests for the cost tracking module."""

import pytest

from docqa_engine.cost_tracker import CostTracker, QueryCost


@pytest.fixture
def tracker():
    t = CostTracker()
    t.record_query("q1", "What is AI?", "claude", "claude-3", 500, 200)
    t.record_query("q2", "What is ML?", "openai", "gpt-4o", 400, 150)
    t.record_query("q3", "What is RAG?", "claude", "claude-3", 600, 300)
    return t


class TestRecordQuery:
    def test_basic(self):
        t = CostTracker()
        result = t.record_query("q1", "Test?", "claude", "claude-3", 100, 50)
        assert isinstance(result, QueryCost)
        assert result.total_tokens == 150
        assert result.cost_estimate > 0
        assert result.timestamp

    def test_mock_is_free(self):
        t = CostTracker()
        result = t.record_query("q1", "Test?", "mock", "mock-v1", 100, 50)
        assert result.cost_estimate == 0.0


class TestAggregations:
    def test_total_cost(self, tracker):
        total = tracker.total_cost()
        assert total > 0

    def test_cost_by_provider(self, tracker):
        costs = tracker.cost_by_provider()
        assert "claude" in costs
        assert "openai" in costs
        assert costs["claude"] > costs["openai"]  # More claude queries

    def test_tokens_by_provider(self, tracker):
        tokens = tracker.tokens_by_provider()
        assert tokens["claude"] == (500 + 200) + (600 + 300)
        assert tokens["openai"] == 400 + 150

    def test_query_count(self, tracker):
        assert tracker.query_count() == 3


class TestSummary:
    def test_summary_structure(self, tracker):
        summary = tracker.summary()
        assert summary["total_queries"] == 3
        assert summary["total_cost"] > 0
        assert summary["total_tokens"] > 0
        assert "claude" in summary["by_provider"]
        assert "queries" in summary["by_provider"]["claude"]

    def test_empty_tracker(self):
        t = CostTracker()
        summary = t.summary()
        assert summary["total_queries"] == 0
        assert summary["total_cost"] == 0


class TestCustomPricing:
    def test_custom_rates(self):
        t = CostTracker(pricing={"custom": 0.05})
        result = t.record_query("q1", "Test?", "custom", "model", 1000, 500)
        assert result.cost_estimate == 0.075  # 0.05 * 1500 / 1000
