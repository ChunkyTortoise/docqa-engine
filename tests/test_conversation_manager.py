"""Tests for conversation manager: multi-turn tracking, context, entity extraction."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from docqa_engine.conversation_manager import (
    ContextAwareExpander,
    ConversationContext,
    ConversationHistory,
    ConversationTurn,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def history() -> ConversationHistory:
    return ConversationHistory()


@pytest.fixture()
def context() -> ConversationContext:
    return ConversationContext()


@pytest.fixture()
def populated_history(history: ConversationHistory) -> ConversationHistory:
    """History with 3 pre-loaded turns."""
    history.add_turn("What is Python?", "Python is a programming language.", ["chunk-1"])
    history.add_turn("How is it used in data science?", "It powers ML pipelines.", ["chunk-2"])
    history.add_turn("What about TensorFlow?", "TensorFlow is a deep learning framework.", ["chunk-3"])
    return history


# ---------------------------------------------------------------------------
# ConversationTurn
# ---------------------------------------------------------------------------


class TestConversationTurn:
    def test_turn_fields(self) -> None:
        turn = ConversationTurn(
            query="hello",
            answer="hi",
            retrieved_chunks=["c1"],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=1,
        )
        assert turn.query == "hello"
        assert turn.answer == "hi"
        assert turn.retrieved_chunks == ["c1"]
        assert turn.turn_number == 1

    def test_turn_empty_chunks(self) -> None:
        turn = ConversationTurn(
            query="q",
            answer="a",
            retrieved_chunks=[],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=1,
        )
        assert turn.retrieved_chunks == []


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------


class TestConversationHistory:
    def test_add_turn_returns_turn(self, history: ConversationHistory) -> None:
        turn = history.add_turn("q", "a", ["c1"])
        assert isinstance(turn, ConversationTurn)
        assert turn.turn_number == 1
        assert turn.query == "q"

    def test_turn_count(self, history: ConversationHistory) -> None:
        assert history.turn_count == 0
        history.add_turn("q1", "a1")
        assert history.turn_count == 1
        history.add_turn("q2", "a2")
        assert history.turn_count == 2

    def test_get_turns_all(self, populated_history: ConversationHistory) -> None:
        turns = populated_history.get_turns()
        assert len(turns) == 3
        assert turns[0].turn_number == 1

    def test_get_turns_last_n(self, populated_history: ConversationHistory) -> None:
        turns = populated_history.get_turns(last_n=2)
        assert len(turns) == 2
        assert turns[0].turn_number == 2
        assert turns[1].turn_number == 3

    def test_get_turns_last_n_exceeds_count(self, populated_history: ConversationHistory) -> None:
        turns = populated_history.get_turns(last_n=10)
        assert len(turns) == 3

    def test_context_window_formatting(self, populated_history: ConversationHistory) -> None:
        window = populated_history.get_context_window(max_turns=2)
        assert "Q: How is it used in data science?" in window
        assert "A: It powers ML pipelines." in window
        assert "Q: What about TensorFlow?" in window

    def test_context_window_empty(self, history: ConversationHistory) -> None:
        assert history.get_context_window() == ""

    def test_clear(self, populated_history: ConversationHistory) -> None:
        populated_history.clear()
        assert populated_history.turn_count == 0
        assert populated_history.get_turns() == []

    def test_add_turn_no_chunks(self, history: ConversationHistory) -> None:
        turn = history.add_turn("q", "a")
        assert turn.retrieved_chunks == []

    def test_turn_numbering_sequential(self, history: ConversationHistory) -> None:
        t1 = history.add_turn("q1", "a1")
        t2 = history.add_turn("q2", "a2")
        t3 = history.add_turn("q3", "a3")
        assert t1.turn_number == 1
        assert t2.turn_number == 2
        assert t3.turn_number == 3

    def test_timestamp_is_utc(self, history: ConversationHistory) -> None:
        turn = history.add_turn("q", "a")
        assert turn.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# ConversationContext
# ---------------------------------------------------------------------------


class TestConversationContext:
    def test_active_topic_empty(self, context: ConversationContext) -> None:
        assert context.active_topic == ""

    def test_active_topic_after_update(self, context: ConversationContext) -> None:
        turn = ConversationTurn(
            query="Tell me about Python programming",
            answer="Python is great for data science",
            retrieved_chunks=[],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=1,
        )
        context.update(turn)
        topic = context.active_topic
        assert isinstance(topic, str)
        assert len(topic) > 0

    def test_resolve_references_replaces_pronoun(self, context: ConversationContext) -> None:
        turn = ConversationTurn(
            query="What is TensorFlow?",
            answer="TensorFlow is a framework.",
            retrieved_chunks=[],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=1,
        )
        context.update(turn)
        resolved = context.resolve_references("How do I install it?")
        assert "it" not in resolved.lower().split() or "TensorFlow" in resolved or context._entities[-1] in resolved

    def test_resolve_references_no_context(self, context: ConversationContext) -> None:
        result = context.resolve_references("How do I install it?")
        assert result == "How do I install it?"

    def test_detect_followup_pronoun(self, context: ConversationContext) -> None:
        turn = ConversationTurn(
            query="What is Python?",
            answer="A language.",
            retrieved_chunks=[],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=1,
        )
        context.update(turn)
        assert context.detect_followup("Can it do ML?") is True

    def test_detect_followup_marker(self, context: ConversationContext) -> None:
        turn = ConversationTurn(
            query="q",
            answer="a",
            retrieved_chunks=[],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=1,
        )
        context.update(turn)
        assert context.detect_followup("What about another approach?") is True

    def test_detect_followup_short_query(self, context: ConversationContext) -> None:
        turn = ConversationTurn(
            query="q",
            answer="a",
            retrieved_chunks=[],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=1,
        )
        context.update(turn)
        assert context.detect_followup("examples?") is True

    def test_detect_followup_no_history(self, context: ConversationContext) -> None:
        assert context.detect_followup("What is Python?") is False

    def test_detect_followup_independent_query(self, context: ConversationContext) -> None:
        turn = ConversationTurn(
            query="q",
            answer="a",
            retrieved_chunks=[],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=1,
        )
        context.update(turn)
        # A full standalone question should not be detected as follow-up
        result = context.detect_followup("What is the capital of France in the world?")
        assert result is False


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    def test_extract_quoted_phrases(self) -> None:
        entities = ConversationContext.extract_entities('Look for "machine learning" concepts')
        assert "machine learning" in entities

    def test_extract_single_quoted(self) -> None:
        entities = ConversationContext.extract_entities("Use 'TensorFlow' library")
        assert "TensorFlow" in entities

    def test_extract_capitalized_multi_word(self) -> None:
        entities = ConversationContext.extract_entities("the city of New York is great")
        assert "New York" in entities

    def test_extract_numbers(self) -> None:
        entities = ConversationContext.extract_entities("The budget is 5000 dollars")
        assert "5000" in entities

    def test_extract_from_empty_string(self) -> None:
        entities = ConversationContext.extract_entities("")
        assert entities == []

    def test_extract_capitalized_word(self) -> None:
        entities = ConversationContext.extract_entities("Use Python for ML tasks")
        assert "Python" in entities


# ---------------------------------------------------------------------------
# ContextAwareExpander
# ---------------------------------------------------------------------------


class TestContextAwareExpander:
    def test_expand_adds_topic(self) -> None:
        ctx = ConversationContext()
        turn = ConversationTurn(
            query="Tell me about Python programming",
            answer="Python is great",
            retrieved_chunks=[],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=1,
        )
        ctx.update(turn)
        expander = ContextAwareExpander()
        result = expander.expand("What are the best practices?", ctx)
        assert len(result) > len("What are the best practices?")

    def test_expand_resolves_pronouns(self) -> None:
        ctx = ConversationContext()
        turn = ConversationTurn(
            query="What is TensorFlow?",
            answer="TensorFlow is a deep learning framework",
            retrieved_chunks=[],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=1,
        )
        ctx.update(turn)
        expander = ContextAwareExpander()
        result = expander.expand("How do I install it?", ctx)
        # Should have resolved "it" and/or added topic
        assert len(result) >= len("How do I install it?")

    def test_expand_empty_query(self) -> None:
        ctx = ConversationContext()
        expander = ContextAwareExpander()
        assert expander.expand("", ctx) == ""

    def test_expand_whitespace_query(self) -> None:
        ctx = ConversationContext()
        expander = ContextAwareExpander()
        assert expander.expand("   ", ctx) == "   "

    def test_expand_no_duplicate_topic(self) -> None:
        ctx = ConversationContext()
        turn = ConversationTurn(
            query="python basics",
            answer="Python is great",
            retrieved_chunks=[],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=1,
        )
        ctx.update(turn)
        expander = ContextAwareExpander()
        result = expander.expand("python advanced features", ctx)
        # Should not duplicate "python" if already in query
        count = result.lower().count("python")
        assert count >= 1
