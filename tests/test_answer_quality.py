"""Tests for answer quality metrics: coherence, conciseness, completeness, groundedness."""

from __future__ import annotations

import pytest

from docqa_engine.answer_quality import AnswerComparison, AnswerQualityScorer, QualityReport


@pytest.fixture()
def scorer() -> AnswerQualityScorer:
    return AnswerQualityScorer()


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------


class TestCoherence:
    """Tests for coherence scoring."""

    def test_coherent_text_scores_high(self, scorer: AnswerQualityScorer) -> None:
        """Sentences about the same topic have high coherence."""
        answer = (
            "Python is a programming language. Python supports multiple paradigms. "
            "Python is used for data science and machine learning."
        )
        score = scorer.coherence(answer)
        assert score > 0.1

    def test_incoherent_text_scores_low(self, scorer: AnswerQualityScorer) -> None:
        """Unrelated sentences have low coherence."""
        answer = "Python is a programming language. Basketball players dribble the ball. Cooking requires fresh spices."
        score = scorer.coherence(answer)
        assert score < 0.5

    def test_single_sentence_returns_one(self, scorer: AnswerQualityScorer) -> None:
        """Single sentence gets coherence 1.0."""
        score = scorer.coherence("Python is great")
        assert score == 1.0

    def test_empty_answer_returns_zero(self, scorer: AnswerQualityScorer) -> None:
        """Empty answer gets coherence 0.0."""
        assert scorer.coherence("") == 0.0


# ---------------------------------------------------------------------------
# Conciseness
# ---------------------------------------------------------------------------


class TestConciseness:
    """Tests for conciseness scoring."""

    def test_short_answer_gets_one(self, scorer: AnswerQualityScorer) -> None:
        """Answer under max_words gets 1.0."""
        answer = "Python is a great language"
        score = scorer.conciseness(answer, max_words=200)
        assert score == 1.0

    def test_long_answer_decreases(self, scorer: AnswerQualityScorer) -> None:
        """Answer over max_words decreases linearly."""
        answer = " ".join(["word"] * 300)
        score = scorer.conciseness(answer, max_words=200)
        assert 0.0 < score < 1.0

    def test_very_long_answer_gets_zero(self, scorer: AnswerQualityScorer) -> None:
        """Answer at 3x max_words gets 0.0."""
        answer = " ".join(["word"] * 600)
        score = scorer.conciseness(answer, max_words=200)
        assert score == 0.0

    def test_empty_answer_gets_one(self, scorer: AnswerQualityScorer) -> None:
        """Empty answer gets 1.0 (maximally concise)."""
        assert scorer.conciseness("") == 1.0

    def test_exact_max_words(self, scorer: AnswerQualityScorer) -> None:
        """Exactly max_words gets 1.0."""
        answer = " ".join(["word"] * 200)
        score = scorer.conciseness(answer, max_words=200)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


class TestCompleteness:
    """Tests for completeness scoring."""

    def test_full_coverage(self, scorer: AnswerQualityScorer) -> None:
        """Answer covering all query keywords scores 1.0."""
        query = "Python machine learning"
        answer = "Python is used for machine learning and data science."
        score = scorer.completeness(answer, query)
        assert score == 1.0

    def test_partial_coverage(self, scorer: AnswerQualityScorer) -> None:
        """Answer covering some query keywords gives partial score."""
        query = "Python machine learning deployment"
        answer = "Python is a great programming language."
        score = scorer.completeness(answer, query)
        assert 0.0 < score < 1.0

    def test_no_coverage(self, scorer: AnswerQualityScorer) -> None:
        """Answer with no query keywords scores 0.0."""
        query = "quantum physics"
        answer = "Basketball players practice daily."
        score = scorer.completeness(answer, query)
        assert score == 0.0

    def test_empty_query(self, scorer: AnswerQualityScorer) -> None:
        """Empty query returns 0.0."""
        assert scorer.completeness("some answer", "") == 0.0

    def test_empty_answer(self, scorer: AnswerQualityScorer) -> None:
        """Empty answer returns 0.0."""
        assert scorer.completeness("", "some query") == 0.0

    def test_stopwords_excluded(self, scorer: AnswerQualityScorer) -> None:
        """Stopwords are not counted as query keywords."""
        query = "what is the"
        answer = "Python is a language."
        score = scorer.completeness(answer, query)
        # "what", "is", "the" are all stopwords, so no meaningful keywords
        assert score == 0.0


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------


class TestGroundedness:
    """Tests for groundedness scoring."""

    def test_grounded_answer(self, scorer: AnswerQualityScorer) -> None:
        """Answer matching context scores high."""
        context = "Python is a popular programming language used in data science."
        answer = "Python is used for programming and data science."
        score = scorer.groundedness(answer, context)
        assert score > 0.5

    def test_ungrounded_answer(self, scorer: AnswerQualityScorer) -> None:
        """Answer not matching context scores low."""
        context = "Python is a popular programming language."
        answer = "Basketball requires physical fitness and teamwork."
        score = scorer.groundedness(answer, context)
        assert score < 0.5

    def test_empty_context(self, scorer: AnswerQualityScorer) -> None:
        """Empty context returns 0.0."""
        assert scorer.groundedness("some answer", "") == 0.0

    def test_empty_answer(self, scorer: AnswerQualityScorer) -> None:
        """Empty answer returns 0.0."""
        assert scorer.groundedness("", "some context") == 0.0


# ---------------------------------------------------------------------------
# Overall score
# ---------------------------------------------------------------------------


class TestScore:
    """Tests for the combined score method."""

    def test_score_returns_report(self, scorer: AnswerQualityScorer) -> None:
        """Score returns a QualityReport with all dimensions."""
        report = scorer.score(
            answer="Python is great for data science.",
            query="Python data science",
            context="Python is widely used for data science projects.",
        )

        assert isinstance(report, QualityReport)
        assert 0.0 <= report.coherence <= 1.0
        assert 0.0 <= report.conciseness <= 1.0
        assert 0.0 <= report.completeness <= 1.0
        assert 0.0 <= report.groundedness <= 1.0
        assert 0.0 <= report.overall <= 1.0

    def test_overall_is_weighted_average(self, scorer: AnswerQualityScorer) -> None:
        """Overall score uses weights: coherence 0.25, conciseness 0.15, completeness 0.30, groundedness 0.30."""
        report = scorer.score(
            answer="Python is great for data science.",
            query="Python data science",
            context="Python is widely used for data science.",
        )

        expected = (
            0.25 * report.coherence
            + 0.15 * report.conciseness
            + 0.30 * report.completeness
            + 0.30 * report.groundedness
        )
        assert abs(report.overall - expected) < 1e-10


# ---------------------------------------------------------------------------
# Compare answers
# ---------------------------------------------------------------------------


class TestCompareAnswers:
    """Tests for compare_answers method."""

    def test_comparison_returns_winner(self, scorer: AnswerQualityScorer) -> None:
        """Comparison identifies the better answer."""
        query = "Python data science"
        context = "Python is widely used for data science and machine learning."
        answer_a = "Python is used for data science and machine learning applications."
        answer_b = "Basketball players practice free throws."

        comparison = scorer.compare_answers(answer_a, answer_b, query, context)

        assert isinstance(comparison, AnswerComparison)
        assert comparison.winner == "a"
        assert comparison.deltas["overall"] > 0

    def test_comparison_tie(self, scorer: AnswerQualityScorer) -> None:
        """Identical answers produce a tie."""
        query = "Python"
        context = "Python is a language."
        answer = "Python is used."

        comparison = scorer.compare_answers(answer, answer, query, context)
        assert comparison.winner == "tie"

    def test_comparison_has_deltas(self, scorer: AnswerQualityScorer) -> None:
        """Comparison includes deltas for all dimensions."""
        comparison = scorer.compare_answers("answer a", "answer b", "query", "context")
        assert "coherence" in comparison.deltas
        assert "conciseness" in comparison.deltas
        assert "completeness" in comparison.deltas
        assert "groundedness" in comparison.deltas
        assert "overall" in comparison.deltas


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


class TestScoreBatch:
    """Tests for score_batch method."""

    def test_batch_returns_list(self, scorer: AnswerQualityScorer) -> None:
        """Batch scoring returns a list of QualityReport."""
        items = [
            ("Python is great.", "Python", "Python is a language."),
            ("Java is popular.", "Java", "Java is used in enterprise."),
        ]
        reports = scorer.score_batch(items)

        assert len(reports) == 2
        assert all(isinstance(r, QualityReport) for r in reports)

    def test_batch_empty(self, scorer: AnswerQualityScorer) -> None:
        """Empty batch returns empty list."""
        assert scorer.score_batch([]) == []

    def test_batch_single_item(self, scorer: AnswerQualityScorer) -> None:
        """Single-item batch works correctly."""
        items = [("Python.", "Python", "Python is a language.")]
        reports = scorer.score_batch(items)
        assert len(reports) == 1
