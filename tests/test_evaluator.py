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


# ---------------------------------------------------------------------------
# RAGAS-style generation quality metrics
# ---------------------------------------------------------------------------


class TestContextRelevancy:
    """Tests for context_relevancy scoring."""

    def test_relevant_context(self, evaluator: Evaluator) -> None:
        """Related text scores high."""
        context = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
        query = "What is machine learning?"
        score = evaluator.context_relevancy(context, query)

        assert score > 0.2

    def test_irrelevant_context(self, evaluator: Evaluator) -> None:
        """Unrelated text scores low."""
        context = "Basketball players practice free throws every afternoon at the gymnasium."
        query = "What is machine learning?"
        score = evaluator.context_relevancy(context, query)

        assert score < 0.2

    def test_empty_strings(self, evaluator: Evaluator) -> None:
        """Handles empty inputs gracefully."""
        assert evaluator.context_relevancy("", "query") == 0.0
        assert evaluator.context_relevancy("context", "") == 0.0
        assert evaluator.context_relevancy("", "") == 0.0


class TestAnswerRelevancy:
    """Tests for answer_relevancy scoring."""

    def test_relevant_answer(self, evaluator: Evaluator) -> None:
        """On-topic answer scores high."""
        answer = "Machine learning uses algorithms to learn patterns from data and make predictions."
        query = "How does machine learning work?"
        score = evaluator.answer_relevancy(answer, query)

        assert score > 0.1

    def test_irrelevant_answer(self, evaluator: Evaluator) -> None:
        """Off-topic answer scores low."""
        answer = "Chocolate cake requires flour, sugar, eggs, and cocoa powder to bake properly."
        query = "How does machine learning work?"
        score = evaluator.answer_relevancy(answer, query)

        assert score < 0.15


class TestFaithfulness:
    """Tests for faithfulness scoring."""

    def test_faithful_answer(self, evaluator: Evaluator) -> None:
        """Answer derived from context scores high."""
        context = "Python is a high-level programming language known for its readability and versatility."
        answer = "Python is a high-level language known for readability."
        score = evaluator.faithfulness(answer, context)

        assert score > 0.7

    def test_hallucinated_answer(self, evaluator: Evaluator) -> None:
        """Unrelated answer scores low."""
        context = "Python is a high-level programming language known for its readability and versatility."
        answer = "Basketball requires dribbling, passing, and shooting skills to compete effectively."
        score = evaluator.faithfulness(answer, context)

        assert score < 0.2


class TestEvaluateRAG:
    """Tests for evaluate_rag comprehensive evaluation."""

    def test_full_evaluation(self, evaluator: Evaluator) -> None:
        """Returns all expected keys when retrieval data is provided."""
        result = evaluator.evaluate_rag(
            query="What is Python?",
            context="Python is a programming language.",
            answer="Python is a popular programming language.",
            retrieved=["d1", "d2", "d3"],
            relevant={"d1", "d3"},
            k=5,
        )

        # Generation metrics
        assert "context_relevancy" in result
        assert "answer_relevancy" in result
        assert "faithfulness" in result

        # Retrieval metrics
        assert "mrr" in result
        assert "ndcg" in result
        assert "precision" in result
        assert "recall" in result
        assert "hit_rate" in result

    def test_without_retrieval(self, evaluator: Evaluator) -> None:
        """Works without retrieved/relevant data, returning only generation metrics."""
        result = evaluator.evaluate_rag(
            query="What is Python?",
            context="Python is a programming language.",
            answer="Python is a popular programming language.",
        )

        assert "context_relevancy" in result
        assert "answer_relevancy" in result
        assert "faithfulness" in result

        # Retrieval metrics should NOT be present
        assert "mrr" not in result
        assert "ndcg" not in result


# ---------------------------------------------------------------------------
# Enhanced Faithfulness Check
# ---------------------------------------------------------------------------


class TestFaithfulnessEnhanced:
    """Tests for check_faithfulness sentence-level support."""

    def test_faithful_answer(self, evaluator: Evaluator) -> None:
        """Fully faithful answer scores 1.0."""
        context = "Machine learning enables systems to learn from data. It uses algorithms and neural networks."
        answer = "Machine learning enables systems to learn from data using algorithms."
        result = evaluator.check_faithfulness(answer, context)

        assert result["score"] > 0.8
        assert len(result["supported_sentences"]) >= 1

    def test_unfaithful_answer(self, evaluator: Evaluator) -> None:
        """Unfaithful answer scores low."""
        context = "Machine learning enables systems to learn from data."
        answer = "Basketball players practice daily. Free throws require concentration."
        result = evaluator.check_faithfulness(answer, context)

        assert result["score"] < 0.3
        assert len(result["unsupported_sentences"]) >= 1

    def test_partially_faithful(self, evaluator: Evaluator) -> None:
        """Mix of faithful and unfaithful sentences."""
        context = "Machine learning uses algorithms to learn from data."
        answer = "Machine learning uses algorithms. Basketball is a popular sport."
        result = evaluator.check_faithfulness(answer, context)

        assert 0.3 < result["score"] < 0.8
        assert len(result["supported_sentences"]) >= 1
        assert len(result["unsupported_sentences"]) >= 1


# ---------------------------------------------------------------------------
# Completeness Check
# ---------------------------------------------------------------------------


class TestCompleteness:
    """Tests for check_completeness."""

    def test_complete_answer(self, evaluator: Evaluator) -> None:
        """Answer addresses all question terms."""
        question = "What programming language is used for data science?"
        answer = "Python is the programming language commonly used for data science applications."
        context = "Python is popular in data science."
        result = evaluator.check_completeness(answer, question, context)

        assert result["score"] > 0.7
        assert len(result["addressed_terms"]) >= 2

    def test_incomplete_answer(self, evaluator: Evaluator) -> None:
        """Answer misses key question terms."""
        question = "What programming language is used for data science?"
        answer = "It is commonly used."
        context = "Python is used for data science."
        result = evaluator.check_completeness(answer, question, context)

        assert result["score"] < 0.5
        assert len(result["missing_terms"]) >= 1

    def test_no_overlap(self, evaluator: Evaluator) -> None:
        """Answer completely unrelated to question."""
        question = "What is machine learning?"
        answer = "Basketball requires teamwork and practice."
        context = "Machine learning uses algorithms."
        result = evaluator.check_completeness(answer, question, context)

        assert result["score"] == 0.0


# ---------------------------------------------------------------------------
# Hallucination Detection
# ---------------------------------------------------------------------------


class TestHallucinationDetection:
    """Tests for detect_hallucinations."""

    def test_no_hallucination(self, evaluator: Evaluator) -> None:
        """Answer fully grounded in context."""
        context = "Machine learning uses algorithms to learn from data. Neural networks are a key component."
        answer = "Machine learning uses algorithms and neural networks to learn from data."
        result = evaluator.detect_hallucinations(answer, context)

        assert result["hallucination_count"] == 0
        assert result["hallucination_rate"] == 0.0

    def test_with_hallucination(self, evaluator: Evaluator) -> None:
        """Answer contains unsupported claims."""
        context = "Machine learning uses algorithms to learn from data."
        answer = "Machine learning uses algorithms. It was invented in ancient Rome by Julius Caesar."
        result = evaluator.detect_hallucinations(answer, context)

        assert result["hallucination_count"] >= 1
        assert result["hallucination_rate"] > 0.0
        assert len(result["hallucinated_sentences"]) >= 1

    def test_all_hallucinated(self, evaluator: Evaluator) -> None:
        """Completely ungrounded answer."""
        context = "Machine learning uses algorithms."
        answer = "Basketball players practice free throws. Cooking requires ingredients."
        result = evaluator.detect_hallucinations(answer, context)

        assert result["hallucination_rate"] > 0.8
        assert result["total_sentences"] == 2


# ---------------------------------------------------------------------------
# Context Quality (Precision/Recall)
# ---------------------------------------------------------------------------


class TestContextQuality:
    """Tests for context_quality precision/recall metrics."""

    def test_perfect_precision_recall(self, evaluator: Evaluator) -> None:
        """All retrieved are relevant, all relevant retrieved."""
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = ["chunk1", "chunk2", "chunk3"]
        result = evaluator.context_quality(retrieved, relevant)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_precision_recall(self, evaluator: Evaluator) -> None:
        """Some retrieved relevant, some relevant not retrieved."""
        retrieved = ["chunk1", "chunk2", "chunk4"]
        relevant = ["chunk1", "chunk2", "chunk3"]
        result = evaluator.context_quality(retrieved, relevant)

        # 2/3 retrieved are relevant -> precision = 0.67
        assert 0.6 < result["precision"] < 0.7
        # 2/3 relevant were retrieved -> recall = 0.67
        assert 0.6 < result["recall"] < 0.7
        assert result["f1"] > 0.0

    def test_zero_precision_recall(self, evaluator: Evaluator) -> None:
        """No overlap between retrieved and relevant."""
        retrieved = ["chunk1", "chunk2"]
        relevant = ["chunk3", "chunk4"]
        result = evaluator.context_quality(retrieved, relevant)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0


# ---------------------------------------------------------------------------
# Additional edge-case and coverage tests
# ---------------------------------------------------------------------------


class TestContextRelevancyEdgeCases:
    """Additional edge-case tests for context_relevancy."""

    def test_identical_text_high_score(self, evaluator: Evaluator) -> None:
        """Identical context and query gives high relevance score."""
        text = "Machine learning algorithms enable data analysis"
        score = evaluator.context_relevancy(text, text)

        assert score >= 0.9

    def test_score_bounded_zero_one(self, evaluator: Evaluator) -> None:
        """Score is always between 0.0 and 1.0."""
        context = "Python is used for web development and data science."
        query = "What is Python used for?"
        score = evaluator.context_relevancy(context, query)

        assert 0.0 <= score <= 1.0


class TestAnswerRelevancyEdgeCases:
    """Additional edge-case tests for answer_relevancy."""

    def test_empty_answer_returns_zero(self, evaluator: Evaluator) -> None:
        """Empty answer returns 0.0."""
        score = evaluator.answer_relevancy("", "What is Python?")

        assert score == 0.0

    def test_empty_query_returns_zero(self, evaluator: Evaluator) -> None:
        """Empty query returns 0.0."""
        score = evaluator.answer_relevancy("Python is a language.", "")

        assert score == 0.0


class TestFaithfulnessEdgeCases:
    """Additional edge-case tests for faithfulness scoring."""

    def test_identical_text_scores_one(self, evaluator: Evaluator) -> None:
        """Identical answer and context gives faithfulness 1.0."""
        text = "Machine learning uses algorithms to learn from data."
        score = evaluator.faithfulness(text, text)

        assert score == pytest.approx(1.0)

    def test_empty_answer_returns_zero(self, evaluator: Evaluator) -> None:
        """Empty answer returns 0.0."""
        score = evaluator.faithfulness("", "Some context here.")

        assert score == 0.0


class TestEvaluateRAGEdgeCases:
    """Additional edge-case tests for evaluate_rag."""

    def test_all_metrics_bounded(self, evaluator: Evaluator) -> None:
        """All generation metrics are in [0, 1]."""
        result = evaluator.evaluate_rag(
            query="What is machine learning?",
            context="Machine learning uses algorithms to learn from data.",
            answer="Machine learning learns from data using algorithms.",
        )

        assert 0.0 <= result["context_relevancy"] <= 1.0
        assert 0.0 <= result["answer_relevancy"] <= 1.0
        assert 0.0 <= result["faithfulness"] <= 1.0

    def test_empty_inputs_return_zeros(self, evaluator: Evaluator) -> None:
        """Empty query/context/answer returns all-zero generation metrics."""
        result = evaluator.evaluate_rag(query="", context="", answer="")

        assert result["context_relevancy"] == 0.0
        assert result["answer_relevancy"] == 0.0
        assert result["faithfulness"] == 0.0


class TestCheckFaithfulnessEdgeCases:
    """Additional edge-case tests for check_faithfulness."""

    def test_empty_answer_returns_zero_score(self, evaluator: Evaluator) -> None:
        """Empty answer returns score 0.0."""
        result = evaluator.check_faithfulness("", "Some context here.")

        assert result["score"] == 0.0
        assert result["supported_sentences"] == []
        assert result["unsupported_sentences"] == []

    def test_whitespace_answer_returns_zero_score(self, evaluator: Evaluator) -> None:
        """Whitespace-only answer returns score 0.0."""
        result = evaluator.check_faithfulness("   ", "Some context here.")

        assert result["score"] == 0.0


class TestCheckCompletenessEdgeCases:
    """Additional edge-case tests for check_completeness."""

    def test_empty_question_returns_zero(self, evaluator: Evaluator) -> None:
        """Empty question returns score 0.0."""
        result = evaluator.check_completeness("Some answer.", "", "Some context.")

        assert result["score"] == 0.0
        assert result["addressed_terms"] == []
        assert result["missing_terms"] == []

    def test_all_terms_addressed(self, evaluator: Evaluator) -> None:
        """Answer containing all question key terms scores high."""
        question = "What are Python programming applications?"
        answer = "Python programming has many applications including web development."
        context = "Python is versatile."
        result = evaluator.check_completeness(answer, question, context)

        assert result["score"] >= 0.5
        assert len(result["addressed_terms"]) >= 2


class TestDetectHallucinationsEdgeCases:
    """Additional edge-case tests for detect_hallucinations."""

    def test_empty_answer_returns_zero_count(self, evaluator: Evaluator) -> None:
        """Empty answer returns 0 hallucination count."""
        result = evaluator.detect_hallucinations("", "Some context.")

        assert result["hallucination_count"] == 0
        assert result["total_sentences"] == 0
        assert result["hallucination_rate"] == 0.0

    def test_grounded_multi_sentence(self, evaluator: Evaluator) -> None:
        """Multiple grounded sentences produce zero hallucinations."""
        context = "Python is popular. It supports data science. Machine learning uses Python."
        answer = "Python is popular for data science. Machine learning often uses Python."
        result = evaluator.detect_hallucinations(answer, context)

        assert result["hallucination_rate"] == 0.0


class TestContextQualityEdgeCases:
    """Additional edge-case tests for context_quality."""

    def test_both_empty_returns_zeros(self, evaluator: Evaluator) -> None:
        """Both empty lists return all-zero scores."""
        result = evaluator.context_quality([], [])

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_single_matching_chunk(self, evaluator: Evaluator) -> None:
        """Single chunk in both retrieved and relevant gives perfect scores."""
        result = evaluator.context_quality(["chunk1"], ["chunk1"])

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
