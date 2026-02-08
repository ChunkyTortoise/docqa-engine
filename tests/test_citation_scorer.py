"""Tests for citation accuracy scoring: faithfulness, coverage, redundancy."""

from __future__ import annotations

import pytest

from docqa_engine.citation_scorer import CitationScorer


@pytest.fixture()
def scorer() -> CitationScorer:
    return CitationScorer()


# ---------------------------------------------------------------------------
# Single citation scoring
# ---------------------------------------------------------------------------


class TestSingleCitation:
    """Tests for score_citation on individual citations."""

    def test_faithful_citation(self, scorer: CitationScorer) -> None:
        """High faithfulness for verbatim quote from source."""
        source = "Machine learning enables systems to learn from data and improve over time."
        citation = "Machine learning enables systems to learn from data."
        score = scorer.score_citation(citation, source)

        assert score.faithfulness > 0.8

    def test_unfaithful_citation(self, scorer: CitationScorer) -> None:
        """Low faithfulness for unrelated text."""
        source = "Machine learning enables systems to learn from data and improve over time."
        citation = "Basketball players practice free throws every afternoon."
        score = scorer.score_citation(citation, source)

        assert score.faithfulness < 0.3

    def test_relevance_with_query(self, scorer: CitationScorer) -> None:
        """Citation relevant to query scores high."""
        source = "Python supports multiple programming paradigms including functional programming."
        citation = "Python supports functional programming paradigms."
        query = "What programming paradigms does Python support?"
        score = scorer.score_citation(citation, source, query=query)

        assert score.relevance >= 0.5

    def test_empty_citation(self, scorer: CitationScorer) -> None:
        """Empty string scores 0."""
        source = "Some source text here."
        score = scorer.score_citation("", source)

        assert score.faithfulness == 0.0
        assert score.relevance == 0.0


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------


class TestCoverage:
    """Tests for coverage_analysis."""

    def test_full_coverage(self, scorer: CitationScorer) -> None:
        """All source words cited gives coverage close to 1.0."""
        source = "Machine learning enables data analysis."
        citations = ["Machine learning enables data analysis and processing."]
        coverage = scorer.coverage_analysis(citations, source)

        assert coverage > 0.8

    def test_partial_coverage(self, scorer: CitationScorer) -> None:
        """Some words cited gives moderate coverage."""
        source = "Machine learning enables data analysis and deep neural networks for vision tasks."
        citations = ["Machine learning for data analysis."]
        coverage = scorer.coverage_analysis(citations, source)

        assert 0.1 < coverage < 0.9

    def test_no_coverage(self, scorer: CitationScorer) -> None:
        """Unrelated citations give coverage close to 0.0."""
        source = "Machine learning enables data analysis."
        citations = ["Basketball players practice free throws."]
        coverage = scorer.coverage_analysis(citations, source)

        assert coverage < 0.2


# ---------------------------------------------------------------------------
# Redundancy detection
# ---------------------------------------------------------------------------


class TestRedundancy:
    """Tests for redundancy_detection."""

    def test_identical_citations(self, scorer: CitationScorer) -> None:
        """All same citations gives redundancy = 1.0."""
        citations = [
            "Machine learning enables data analysis.",
            "Machine learning enables data analysis.",
            "Machine learning enables data analysis.",
        ]
        redundancy = scorer.redundancy_detection(citations)

        assert redundancy == 1.0

    def test_unique_citations(self, scorer: CitationScorer) -> None:
        """All different citations gives redundancy close to 0.0."""
        citations = [
            "Machine learning enables data analysis.",
            "Basketball players practice free throws daily.",
            "Cooking requires fresh ingredients and patience.",
        ]
        redundancy = scorer.redundancy_detection(citations)

        assert redundancy < 0.2

    def test_single_citation(self, scorer: CitationScorer) -> None:
        """No pairs to compare gives redundancy = 0.0."""
        redundancy = scorer.redundancy_detection(["Only one citation here."])

        assert redundancy == 0.0


# ---------------------------------------------------------------------------
# Comprehensive scoring
# ---------------------------------------------------------------------------


class TestScoreAll:
    """Tests for score_all comprehensive report."""

    def test_report_completeness(self, scorer: CitationScorer) -> None:
        """Report has all required fields."""
        source = "Machine learning enables systems to learn from data."
        citations = [
            "Machine learning systems learn from data.",
            "Systems can learn automatically.",
        ]
        report = scorer.score_all(citations, source, query="What is machine learning?")

        assert len(report.scores) == 2
        assert report.avg_faithfulness >= 0.0
        assert report.avg_relevance >= 0.0
        assert 0.0 <= report.coverage <= 1.0
        assert 0.0 <= report.redundancy <= 1.0
        assert report.overall_score is not None

    def test_overall_score_range(self, scorer: CitationScorer) -> None:
        """overall_score is in [0, 1]."""
        source = "Python is a popular programming language."
        citations = ["Python is popular.", "Programming in Python."]
        report = scorer.score_all(citations, source, query="Tell me about Python")

        assert 0.0 <= report.overall_score <= 1.0
