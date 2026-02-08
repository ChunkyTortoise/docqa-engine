"""Tests for pluggable document chunking strategies."""

from __future__ import annotations

import pytest

from docqa_engine.chunking import Chunker


@pytest.fixture()
def chunker() -> Chunker:
    return Chunker()


@pytest.fixture()
def long_text() -> str:
    """A text long enough to produce multiple chunks at default sizes."""
    return (
        "Machine learning is a subset of artificial intelligence. "
        "It enables systems to learn from data. "
        "Deep learning uses neural networks with many layers. "
        "Natural language processing deals with text understanding. "
        "Computer vision focuses on image recognition tasks. "
        "Reinforcement learning trains agents through rewards. "
        "Supervised learning uses labeled training examples. "
        "Unsupervised learning discovers hidden patterns. "
        "Transfer learning reuses pretrained model weights. "
        "Feature engineering creates useful input representations. "
    ) * 5


# ---------------------------------------------------------------------------
# Fixed-size chunking
# ---------------------------------------------------------------------------


class TestFixedSize:
    """Tests for fixed_size chunking strategy."""

    def test_basic_chunking(self, chunker: Chunker, long_text: str) -> None:
        """Splits long text into multiple chunks."""
        result = chunker.fixed_size(long_text, chunk_size=200, overlap=0)

        assert result.total_chunks > 1
        assert result.strategy == "fixed_size"
        assert result.avg_chunk_size > 0
        for chunk in result.chunks:
            assert chunk.text
            assert chunk.end_char > chunk.start_char

    def test_overlap(self, chunker: Chunker, long_text: str) -> None:
        """Chunks overlap by the specified amount."""
        result = chunker.fixed_size(long_text, chunk_size=200, overlap=50)

        # With overlap, we should get more chunks than without
        result_no_overlap = chunker.fixed_size(long_text, chunk_size=200, overlap=0)
        assert result.total_chunks >= result_no_overlap.total_chunks

    def test_word_boundary(self, chunker: Chunker) -> None:
        """Doesn't split mid-word."""
        text = "hello world this is a test of word boundary splitting behavior"
        result = chunker.fixed_size(text, chunk_size=15, overlap=0)

        for chunk in result.chunks:
            # No chunk should start or end in the middle of a word
            # (except at the very start/end of text)
            stripped = chunk.text.strip()
            if stripped:
                # First char should not be a continuation of a split word
                assert not stripped[0].isalpha() or chunk.start_char == 0 or text[chunk.start_char - 1] == " "

    def test_short_text(self, chunker: Chunker) -> None:
        """Text shorter than chunk_size produces 1 chunk."""
        text = "Short text here."
        result = chunker.fixed_size(text, chunk_size=500)

        assert result.total_chunks == 1
        assert result.chunks[0].text.strip() == text.strip()

    def test_empty_text(self, chunker: Chunker) -> None:
        """Empty text returns empty result."""
        result = chunker.fixed_size("", chunk_size=500)

        assert result.total_chunks == 0
        assert result.chunks == []
        assert result.avg_chunk_size == 0.0


# ---------------------------------------------------------------------------
# Sentence chunking
# ---------------------------------------------------------------------------


class TestSentence:
    """Tests for sentence-based chunking strategy."""

    def test_sentence_grouping(self, chunker: Chunker) -> None:
        """Groups sentences correctly."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence."
        result = chunker.sentence(text, sentences_per_chunk=3)

        assert result.total_chunks == 2
        assert result.strategy == "sentence"
        # First chunk should contain 3 sentences
        assert "First" in result.chunks[0].text
        assert "Third" in result.chunks[0].text

    def test_single_sentence(self, chunker: Chunker) -> None:
        """Single sentence produces 1 chunk."""
        text = "Just one sentence here."
        result = chunker.sentence(text, sentences_per_chunk=5)

        assert result.total_chunks == 1
        assert result.chunks[0].text.strip() == text.strip()

    def test_various_delimiters(self, chunker: Chunker) -> None:
        """Handles ., !, ? delimiters."""
        text = "Is this a question? Yes it is! And this is a statement. Another one."
        result = chunker.sentence(text, sentences_per_chunk=2)

        assert result.total_chunks >= 2
        assert result.strategy == "sentence"


# ---------------------------------------------------------------------------
# Sliding window chunking
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    """Tests for sliding window chunking strategy."""

    def test_window_step(self, chunker: Chunker, long_text: str) -> None:
        """Correct step between consecutive chunks."""
        result = chunker.sliding_window(long_text, window_size=200, step=100)

        assert result.total_chunks > 1
        assert result.strategy == "sliding_window"

        # Check that chunks start at increasing positions with the right step
        for i in range(1, len(result.chunks)):
            expected_start = result.chunks[i - 1].start_char + 100
            assert result.chunks[i].start_char == expected_start

    def test_full_overlap(self, chunker: Chunker, long_text: str) -> None:
        """Step < window_size creates overlapping chunks."""
        result = chunker.sliding_window(long_text, window_size=200, step=50)

        # With heavy overlap, we get many more chunks
        result_no_overlap = chunker.sliding_window(long_text, window_size=200, step=200)
        assert result.total_chunks > result_no_overlap.total_chunks

        # Verify actual overlap: consecutive chunks share a substring
        if len(result.chunks) >= 2:
            c0 = result.chunks[0].text
            c1 = result.chunks[1].text
            # The step is 50, window is 200, so last 150 chars of c0
            # should overlap with the first 150 chars of c1
            overlap_region = c0[50:]  # chars after the step
            assert len(overlap_region) > 0
            # The start of c1 should match part of c0
            assert c1[:20] in c0


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------


class TestSemantic:
    """Tests for paragraph-based semantic chunking."""

    def test_paragraph_split(self, chunker: Chunker) -> None:
        """Splits on double newlines."""
        text = "First paragraph with enough text to exceed minimum.\n\nSecond paragraph also with enough text to exceed minimum.\n\nThird paragraph with sufficient content as well."
        result = chunker.semantic(text, min_chunk_size=10)

        assert result.total_chunks == 3
        assert result.strategy == "semantic"

    def test_merge_short(self, chunker: Chunker) -> None:
        """Short paragraphs get merged."""
        text = "Short.\n\nAlso short.\n\nThis is a much longer paragraph that exceeds the minimum chunk size threshold for splitting."
        result = chunker.semantic(text, min_chunk_size=100)

        # "Short." and "Also short." should be merged since they're under min_chunk_size
        assert result.total_chunks < 3

    def test_single_paragraph(self, chunker: Chunker) -> None:
        """One paragraph produces 1 chunk."""
        text = "This is a single paragraph with no double newlines at all, just one block of text."
        result = chunker.semantic(text, min_chunk_size=10)

        assert result.total_chunks == 1
        assert result.chunks[0].text == text


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------


class TestCompare:
    """Tests for compare_strategies."""

    def test_all_strategies_present(self, chunker: Chunker, long_text: str) -> None:
        """All 4 strategies appear in results."""
        comparison = chunker.compare_strategies(long_text)

        assert "fixed_size" in comparison.results
        assert "sentence" in comparison.results
        assert "sliding_window" in comparison.results
        assert "semantic" in comparison.results

    def test_best_selected(self, chunker: Chunker, long_text: str) -> None:
        """best_strategy has the avg closest to 500."""
        comparison = chunker.compare_strategies(long_text)

        best = comparison.best_strategy
        best_distance = abs(comparison.results[best].avg_chunk_size - 500)

        for name, result in comparison.results.items():
            distance = abs(result.avg_chunk_size - 500)
            assert distance >= best_distance - 1e-9  # best should be closest

    def test_comparison_consistency(self, chunker: Chunker, long_text: str) -> None:
        """Same text gives same results on repeated calls."""
        comp1 = chunker.compare_strategies(long_text)
        comp2 = chunker.compare_strategies(long_text)

        assert comp1.best_strategy == comp2.best_strategy
        for name in comp1.results:
            assert comp1.results[name].total_chunks == comp2.results[name].total_chunks
            assert comp1.results[name].avg_chunk_size == pytest.approx(comp2.results[name].avg_chunk_size)
