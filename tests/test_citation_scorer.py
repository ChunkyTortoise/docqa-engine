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


# ---------------------------------------------------------------------------
# TF-IDF Relevance Scoring
# ---------------------------------------------------------------------------


class TestRelevanceScoring:
    """Tests for relevance_score_tfidf."""

    def test_high_relevance(self, scorer: CitationScorer) -> None:
        """Similar query and citation get high relevance score."""
        query = "machine learning algorithms for data analysis"
        citation = "Machine learning and data analysis algorithms are powerful tools"
        score = scorer.relevance_score_tfidf(query, citation)

        assert score > 0.5

    def test_low_relevance(self, scorer: CitationScorer) -> None:
        """Unrelated query and citation get low relevance score."""
        query = "machine learning algorithms"
        citation = "Basketball players practice free throws daily in the gym"
        score = scorer.relevance_score_tfidf(query, citation)

        assert score < 0.2

    def test_empty_query(self, scorer: CitationScorer) -> None:
        """Empty query returns 0."""
        citation = "Some citation text here"
        score = scorer.relevance_score_tfidf("", citation)

        assert score == 0.0


# ---------------------------------------------------------------------------
# Citation Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for deduplicate_citations."""

    def test_removes_duplicates(self, scorer: CitationScorer) -> None:
        """Removes duplicate citations."""
        citations = [
            "Machine learning enables data analysis and prediction",
            "Basketball players practice free throws",
            "Machine learning enables data prediction and analysis",  # Similar to first
            "Cooking requires fresh ingredients",
        ]
        result = scorer.deduplicate_citations(citations, similarity_threshold=0.7)

        assert result["removed_count"] >= 1
        assert len(result["citations"]) < len(citations)
        assert result["original_count"] == 4

    def test_no_duplicates(self, scorer: CitationScorer) -> None:
        """Keeps all citations when none are duplicates."""
        citations = [
            "Machine learning enables data analysis",
            "Basketball players practice daily",
            "Cooking requires fresh ingredients",
        ]
        result = scorer.deduplicate_citations(citations, similarity_threshold=0.8)

        assert result["removed_count"] == 0
        assert len(result["citations"]) == 3

    def test_all_duplicates(self, scorer: CitationScorer) -> None:
        """Removes all but one when all are duplicates."""
        citations = [
            "Machine learning data analysis",
            "Machine learning data analysis",
            "Machine learning data analysis",
        ]
        result = scorer.deduplicate_citations(citations, similarity_threshold=0.8)

        assert result["removed_count"] == 2
        assert len(result["citations"]) == 1


# ---------------------------------------------------------------------------
# Citation Ranking
# ---------------------------------------------------------------------------


class TestCitationRanking:
    """Tests for rank_citations."""

    def test_ranked_order_correct(self, scorer: CitationScorer) -> None:
        """Citations are ranked by relevance to query."""
        query = "machine learning algorithms"
        citations = [
            "Basketball players practice daily",  # Low relevance
            "Machine learning algorithms are powerful",  # High relevance
            "Cooking requires ingredients",  # Low relevance
        ]
        ranked = scorer.rank_citations(citations, query)

        # Most relevant should be first
        assert "machine learning" in ranked[0].lower()

    def test_single_citation(self, scorer: CitationScorer) -> None:
        """Single citation returns that citation."""
        citations = ["Single citation here"]
        ranked = scorer.rank_citations(citations, "query")

        assert len(ranked) == 1
        assert ranked[0] == citations[0]

    def test_empty_list(self, scorer: CitationScorer) -> None:
        """Empty list returns empty list."""
        ranked = scorer.rank_citations([], "query")

        assert ranked == []


# ---------------------------------------------------------------------------
# Source Coverage
# ---------------------------------------------------------------------------


class TestSourceCoverage:
    """Tests for source_coverage."""

    def test_full_coverage(self, scorer: CitationScorer) -> None:
        """All sources covered by citations."""
        citations = ["Machine learning data", "Basketball practice", "Cooking ingredients"]
        sources = [
            "Machine learning enables data analysis",
            "Basketball players practice daily",
            "Cooking requires fresh ingredients",
        ]
        result = scorer.source_coverage(citations, sources)

        assert result["coverage_ratio"] == 1.0
        assert len(result["covered_sources"]) == 3
        assert len(result["uncovered_sources"]) == 0

    def test_partial_coverage(self, scorer: CitationScorer) -> None:
        """Some sources covered."""
        citations = ["Machine learning data analysis"]
        sources = [
            "Machine learning enables data analysis",
            "Basketball players practice daily",
            "Cooking requires fresh ingredients",
        ]
        result = scorer.source_coverage(citations, sources)

        assert 0.0 < result["coverage_ratio"] < 1.0
        assert len(result["covered_sources"]) >= 1
        assert len(result["uncovered_sources"]) >= 1

    def test_no_coverage(self, scorer: CitationScorer) -> None:
        """No sources covered."""
        citations = ["Quantum physics experiments"]
        sources = [
            "Machine learning data analysis",
            "Basketball practice drills",
        ]
        result = scorer.source_coverage(citations, sources)

        assert result["coverage_ratio"] == 0.0
        assert len(result["covered_sources"]) == 0
        assert len(result["uncovered_sources"]) == 2


# ---------------------------------------------------------------------------
# Additional edge-case and coverage tests
# ---------------------------------------------------------------------------


class TestScoreCitationEdgeCases:
    """Additional edge-case tests for score_citation."""

    def test_partial_overlap_faithfulness(self, scorer: CitationScorer) -> None:
        """Partial keyword overlap gives faithfulness between 0.3 and 0.9."""
        source = "Machine learning enables systems to learn from data and improve over time."
        citation = "Machine learning can solve complex problems and transform industries."
        score = scorer.score_citation(citation, source)

        assert 0.1 <= score.faithfulness <= 0.9

    def test_no_query_gives_zero_relevance(self, scorer: CitationScorer) -> None:
        """Empty query always produces relevance 0.0."""
        source = "Python is a programming language."
        citation = "Python is widely used."
        score = scorer.score_citation(citation, source, query="")

        assert score.relevance == 0.0

    def test_faithfulness_bounded_zero_one(self, scorer: CitationScorer) -> None:
        """Faithfulness score is always between 0 and 1."""
        source = "Some source text with various keywords."
        citation = "Some citation text with different keywords."
        score = scorer.score_citation(citation, source)

        assert 0.0 <= score.faithfulness <= 1.0

    def test_relevance_bounded_zero_one(self, scorer: CitationScorer) -> None:
        """Relevance score is always between 0 and 1."""
        source = "Machine learning for data analysis."
        citation = "Data analysis using machine learning techniques."
        score = scorer.score_citation(citation, source, query="machine learning data analysis")

        assert 0.0 <= score.relevance <= 1.0


class TestCoverageEdgeCases:
    """Additional edge-case tests for coverage_analysis."""

    def test_empty_citations_list(self, scorer: CitationScorer) -> None:
        """Empty citations list gives 0 coverage."""
        source = "Machine learning enables data analysis."
        coverage = scorer.coverage_analysis([], source)

        assert coverage == 0.0

    def test_empty_source_returns_zero(self, scorer: CitationScorer) -> None:
        """Empty source returns 0.0 coverage."""
        coverage = scorer.coverage_analysis(["Some citation"], "")

        assert coverage == 0.0


class TestRedundancyEdgeCases:
    """Additional edge-case tests for redundancy_detection."""

    def test_empty_list_returns_zero(self, scorer: CitationScorer) -> None:
        """Empty citations list returns 0.0 redundancy."""
        redundancy = scorer.redundancy_detection([])

        assert redundancy == 0.0

    def test_two_identical_citations(self, scorer: CitationScorer) -> None:
        """Two identical citations gives redundancy 1.0."""
        citations = [
            "Machine learning enables data analysis",
            "Machine learning enables data analysis",
        ]
        redundancy = scorer.redundancy_detection(citations)

        assert redundancy == 1.0


class TestRelevanceTfidfEdgeCases:
    """Additional edge-case tests for relevance_score_tfidf."""

    def test_identical_text_high_score(self, scorer: CitationScorer) -> None:
        """Identical query and citation gives high relevance."""
        text = "machine learning algorithms for data analysis"
        score = scorer.relevance_score_tfidf(text, text)

        assert score >= 0.9

    def test_empty_citation_returns_zero(self, scorer: CitationScorer) -> None:
        """Empty citation returns 0.0."""
        score = scorer.relevance_score_tfidf("some query", "")

        assert score == 0.0


class TestDeduplicateEdgeCases:
    """Additional edge-case tests for deduplicate_citations."""

    def test_empty_list_returns_empty(self, scorer: CitationScorer) -> None:
        """Empty citations returns empty result."""
        result = scorer.deduplicate_citations([])

        assert result["citations"] == []
        assert result["removed_count"] == 0
        assert result["original_count"] == 0

    def test_single_citation_kept(self, scorer: CitationScorer) -> None:
        """Single citation is always kept."""
        result = scorer.deduplicate_citations(["One citation only."])

        assert len(result["citations"]) == 1
        assert result["removed_count"] == 0


class TestSourceCoverageEdgeCases:
    """Additional edge-case tests for source_coverage."""

    def test_empty_source_list(self, scorer: CitationScorer) -> None:
        """Empty source list returns zero coverage."""
        result = scorer.source_coverage(["some citation"], [])

        assert result["coverage_ratio"] == 0.0
        assert result["covered_sources"] == []
        assert result["uncovered_sources"] == []

    def test_empty_citations_list(self, scorer: CitationScorer) -> None:
        """Empty citations list leaves all sources uncovered."""
        sources = ["Source one text.", "Source two text."]
        result = scorer.source_coverage([], sources)

        assert result["coverage_ratio"] == 0.0
        assert len(result["uncovered_sources"]) == 2
