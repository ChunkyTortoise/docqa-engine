"""Tests for cross-encoder re-ranker: TF-IDF cross-similarity scoring."""

from __future__ import annotations

import pytest

from docqa_engine.ingest import DocumentChunk
from docqa_engine.reranker import CrossEncoderReranker, RerankReport, RerankResult
from docqa_engine.retriever import SearchResult


def _make_result(content: str, rank: int, score: float = 1.0) -> SearchResult:
    """Helper to create a SearchResult with minimal boilerplate."""
    chunk = DocumentChunk(
        chunk_id=f"chunk-{rank}",
        document_id="doc-1",
        content=content,
    )
    return SearchResult(chunk=chunk, score=score, rank=rank, source="bm25")


@pytest.fixture()
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@pytest.fixture()
def sample_results() -> list[SearchResult]:
    return [
        _make_result("Python is a popular programming language for data science", rank=1),
        _make_result("Basketball players practice free throws every day", rank=2),
        _make_result("Machine learning and Python are used in data analysis", rank=3),
        _make_result("Cooking recipes require fresh ingredients and spices", rank=4),
    ]


# ---------------------------------------------------------------------------
# Basic reranking
# ---------------------------------------------------------------------------


class TestRerank:
    """Tests for rerank method."""

    def test_reranking_changes_order(self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]) -> None:
        """Reranking with a specific query should change the order."""
        reranked = reranker.rerank("Python data science programming", sample_results)

        assert len(reranked) == 4
        # Python-related results should be ranked higher
        top_content = reranked[0].search_result.chunk.content
        assert "Python" in top_content or "data" in top_content

    def test_rerank_preserves_count(self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]) -> None:
        """All results are returned when no top_k specified."""
        reranked = reranker.rerank("any query", sample_results)
        assert len(reranked) == 4

    def test_rerank_with_top_k(self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]) -> None:
        """top_k limits the number of returned results."""
        reranked = reranker.rerank("Python", sample_results, top_k=2)
        assert len(reranked) == 2

    def test_rerank_scores_descending(self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]) -> None:
        """Results are sorted by score in descending order."""
        reranked = reranker.rerank("Python programming", sample_results)
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_new_ranks_sequential(
        self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]
    ) -> None:
        """New ranks are sequential starting from 1."""
        reranked = reranker.rerank("data analysis", sample_results)
        new_ranks = [r.new_rank for r in reranked]
        assert new_ranks == list(range(1, len(reranked) + 1))

    def test_rerank_original_ranks_preserved(
        self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]
    ) -> None:
        """Original ranks from SearchResult are preserved."""
        reranked = reranker.rerank("data analysis", sample_results)
        original_ranks = sorted([r.original_rank for r in reranked])
        assert original_ranks == [1, 2, 3, 4]

    def test_rerank_empty_results(self, reranker: CrossEncoderReranker) -> None:
        """Empty input returns empty list."""
        reranked = reranker.rerank("query", [])
        assert reranked == []

    def test_rerank_single_result(self, reranker: CrossEncoderReranker) -> None:
        """Single result is returned with rank 1."""
        results = [_make_result("Python programming language", rank=1)]
        reranked = reranker.rerank("Python", results)

        assert len(reranked) == 1
        assert reranked[0].new_rank == 1
        assert reranked[0].score >= 0.0

    def test_rerank_identical_scores(self, reranker: CrossEncoderReranker) -> None:
        """Identical content produces equal scores."""
        results = [
            _make_result("Python programming", rank=1),
            _make_result("Python programming", rank=2),
        ]
        reranked = reranker.rerank("Python", results)

        assert len(reranked) == 2
        assert reranked[0].score == reranked[1].score


# ---------------------------------------------------------------------------
# Cascade reranking
# ---------------------------------------------------------------------------


class TestCascadeRerank:
    """Tests for cascade_rerank with multiple stages."""

    def test_cascade_single_stage(self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]) -> None:
        """Single-stage cascade works like regular rerank."""
        stage_fn = reranker._score_pair
        reranked = reranker.cascade_rerank("Python", sample_results, stages=[stage_fn])

        assert len(reranked) == 4
        assert reranked[0].new_rank == 1

    def test_cascade_multiple_stages(self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]) -> None:
        """Multiple stages are applied sequentially."""

        def length_scorer(query: str, document: str) -> float:
            return 1.0 / max(len(document), 1)

        reranked = reranker.cascade_rerank(
            "Python",
            sample_results,
            stages=[reranker._score_pair, length_scorer],
        )

        assert len(reranked) == 4
        # After length scoring, shorter docs should rank higher
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_cascade_empty_results(self, reranker: CrossEncoderReranker) -> None:
        """Empty results return empty list."""
        reranked = reranker.cascade_rerank("query", [], stages=[reranker._score_pair])
        assert reranked == []

    def test_cascade_empty_stages(self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]) -> None:
        """Empty stages list returns results with original scores."""
        reranked = reranker.cascade_rerank("query", sample_results, stages=[])
        assert len(reranked) == 4


# ---------------------------------------------------------------------------
# Report with Kendall tau
# ---------------------------------------------------------------------------


class TestReport:
    """Tests for report method with Kendall tau."""

    def test_report_has_all_fields(self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]) -> None:
        """Report contains results, kendall_tau, and improvement_ratio."""
        report = reranker.report("Python programming", sample_results)

        assert isinstance(report, RerankReport)
        assert len(report.results) == 4
        assert -1.0 <= report.kendall_tau <= 1.0
        assert 0.0 <= report.improvement_ratio <= 1.0

    def test_kendall_tau_perfect_agreement(self, reranker: CrossEncoderReranker) -> None:
        """Kendall tau = 1.0 when original and new ranks match perfectly."""
        # Create results where TF-IDF scoring would maintain original order
        results = [_make_result("exact match query terms here", rank=1)]
        report = reranker.report("exact match query terms here", results)

        assert report.kendall_tau == 1.0

    def test_kendall_tau_bounded(self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]) -> None:
        """Kendall tau is always between -1 and 1."""
        report = reranker.report("random query string", sample_results)
        assert -1.0 <= report.kendall_tau <= 1.0

    def test_report_empty_results(self, reranker: CrossEncoderReranker) -> None:
        """Empty results produce empty report."""
        report = reranker.report("query", [])
        assert report.results == []
        assert report.kendall_tau == 1.0
        assert report.improvement_ratio == 0.0

    def test_improvement_ratio_range(self, reranker: CrossEncoderReranker, sample_results: list[SearchResult]) -> None:
        """Improvement ratio is between 0 and 1."""
        report = reranker.report("Python data science", sample_results)
        assert 0.0 <= report.improvement_ratio <= 1.0


# ---------------------------------------------------------------------------
# RerankResult dataclass
# ---------------------------------------------------------------------------


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_result_fields(self) -> None:
        """RerankResult holds all expected fields."""
        sr = _make_result("test content", rank=3)
        rr = RerankResult(original_rank=3, new_rank=1, score=0.85, search_result=sr)

        assert rr.original_rank == 3
        assert rr.new_rank == 1
        assert rr.score == 0.85
        assert rr.search_result is sr

    def test_score_pair_empty_query(self, reranker: CrossEncoderReranker) -> None:
        """Empty query produces score 0.0."""
        assert reranker._score_pair("", "some document") == 0.0

    def test_score_pair_empty_document(self, reranker: CrossEncoderReranker) -> None:
        """Empty document produces score 0.0."""
        assert reranker._score_pair("some query", "") == 0.0
