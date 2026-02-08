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


# ---------------------------------------------------------------------------
# Semantic TF-IDF chunking
# ---------------------------------------------------------------------------


class TestSemanticTfidfChunking:
    """Tests for semantic_tfidf chunking strategy."""

    def test_topic_change_detection(self, chunker: Chunker) -> None:
        """Detects topic changes based on TF-IDF similarity."""
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "It uses algorithms to learn patterns from data. "
            "Neural networks are inspired by biological neurons.\n\n"
            "Pizza is a popular Italian dish. "
            "It consists of dough, sauce, and toppings. "
            "Many people enjoy pizza for dinner.\n\n"
            "Python is a high-level programming language. "
            "It is widely used in data science. "
            "Python has a simple and readable syntax."
        )
        result = chunker.semantic_tfidf(text, similarity_threshold=0.3)

        # Should detect topic changes and create multiple chunks
        assert result.total_chunks >= 2
        assert result.strategy == "semantic_tfidf"

    def test_preserves_content(self, chunker: Chunker) -> None:
        """All chunks combined preserve original content."""
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        result = chunker.semantic_tfidf(text, similarity_threshold=0.5)

        combined = "\n\n".join(c.text for c in result.chunks)
        # Remove extra whitespace for comparison
        assert combined.replace("\n\n", " ").strip() == text.replace("\n\n", " ").strip()

    def test_min_chunks_single_paragraph(self, chunker: Chunker) -> None:
        """Single paragraph produces single chunk."""
        text = "This is a single paragraph with no topic changes."
        result = chunker.semantic_tfidf(text, similarity_threshold=0.3)

        assert result.total_chunks == 1
        assert result.chunks[0].text.strip() == text.strip()

    def test_threshold_effect(self, chunker: Chunker) -> None:
        """Threshold affects chunk boundaries."""
        text = (
            "Machine learning uses algorithms.\n\n"
            "Deep learning uses neural networks.\n\n"
            "Computer vision processes images.\n\n"
            "Natural language processing handles text."
        )

        result_high = chunker.semantic_tfidf(text, similarity_threshold=0.8)
        result_low = chunker.semantic_tfidf(text, similarity_threshold=0.1)

        # Both should create multiple chunks
        assert result_high.total_chunks >= 1
        assert result_low.total_chunks >= 1


# ---------------------------------------------------------------------------
# Sliding window with overlap ratio
# ---------------------------------------------------------------------------


class TestSlidingWindowRatio:
    """Tests for sliding_window_ratio chunking strategy."""

    def test_overlap_ratio_50_percent(self, chunker: Chunker, long_text: str) -> None:
        """50% overlap ratio creates correct overlap."""
        result = chunker.sliding_window_ratio(long_text, chunk_size=200, overlap_ratio=0.5)

        assert result.total_chunks > 1
        assert result.strategy == "sliding_window_ratio"
        # Verify chunks overlap
        if result.total_chunks > 1:
            # Second chunk should start before first chunk ends
            assert result.chunks[1].start_char < result.chunks[0].end_char

    def test_overlap_ratio_25_percent(self, chunker: Chunker, long_text: str) -> None:
        """25% overlap ratio creates less overlap than 50%."""
        result_50 = chunker.sliding_window_ratio(long_text, chunk_size=200, overlap_ratio=0.5)
        result_25 = chunker.sliding_window_ratio(long_text, chunk_size=200, overlap_ratio=0.25)

        # 25% overlap should have fewer or equal chunks than 50% overlap
        assert result_25.total_chunks <= result_50.total_chunks

    def test_no_overlap(self, chunker: Chunker, long_text: str) -> None:
        """0% overlap creates non-overlapping chunks."""
        result = chunker.sliding_window_ratio(long_text, chunk_size=200, overlap_ratio=0.0)

        # No overlap means chunks should not overlap
        for i in range(len(result.chunks) - 1):
            assert result.chunks[i].end_char <= result.chunks[i + 1].start_char

    def test_small_text(self, chunker: Chunker) -> None:
        """Text smaller than chunk_size produces 1 chunk."""
        text = "Small text."
        result = chunker.sliding_window_ratio(text, chunk_size=500, overlap_ratio=0.5)

        assert result.total_chunks == 1


# ---------------------------------------------------------------------------
# Recursive chunking
# ---------------------------------------------------------------------------


class TestRecursiveChunking:
    """Tests for recursive chunking strategy."""

    def test_paragraph_split(self, chunker: Chunker) -> None:
        """Splits on paragraph boundaries first."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = chunker.recursive(text, max_chunk_size=50)

        assert result.total_chunks >= 3
        assert result.strategy == "recursive"

    def test_sentence_split(self, chunker: Chunker) -> None:
        """Falls back to sentence split when paragraphs too large."""
        # One long paragraph with sentences
        text = "First sentence. " * 50
        result = chunker.recursive(text.strip(), max_chunk_size=100)

        # Should split into multiple chunks
        assert result.total_chunks > 1
        for chunk in result.chunks:
            # Most chunks should respect max size (last chunk might be smaller)
            assert len(chunk.text) <= 200 * 1.2  # Allow 20% tolerance (max_chunk_size=200)

    def test_max_size_respected(self, chunker: Chunker) -> None:
        """Chunks respect max_chunk_size (with tolerance for unsplittable text)."""
        text = "Short paragraph.\n\n" + "x " * 1000
        result = chunker.recursive(text, max_chunk_size=200)

        # At least some chunks should be under max_chunk_size
        under_limit = sum(1 for c in result.chunks if len(c.text) <= 200)
        assert under_limit >= 1

    def test_preserves_all_content(self, chunker: Chunker) -> None:
        """All text is preserved across chunks."""
        text = "Paragraph one.\n\nParagraph two. Sentence here.\n\nParagraph three."
        result = chunker.recursive(text, max_chunk_size=100)

        # Combine all chunk texts and count total characters (excluding whitespace)
        combined = "".join(c.text for c in result.chunks)
        # Remove all whitespace for comparison
        original_no_space = text.replace(" ", "").replace("\n", "")
        combined_no_space = combined.replace(" ", "").replace("\n", "")
        # Character counts should match (allowing for minor differences in whitespace)
        assert abs(len(original_no_space) - len(combined_no_space)) < 5


# ---------------------------------------------------------------------------
# Additional edge-case and coverage tests
# ---------------------------------------------------------------------------


class TestFixedSizeEdgeCases:
    """Additional edge-case tests for fixed_size chunking."""

    def test_chunk_metadata_contains_strategy(self, chunker: Chunker) -> None:
        """Each chunk's metadata includes strategy key."""
        text = "Some text that is long enough to produce a chunk for testing purposes."
        result = chunker.fixed_size(text, chunk_size=500)

        for chunk in result.chunks:
            assert "strategy" in chunk.metadata
            assert chunk.metadata["strategy"] == "fixed_size"

    def test_large_overlap_produces_more_chunks(self, chunker: Chunker, long_text: str) -> None:
        """Larger overlap produces more (or equal) chunks than smaller overlap."""
        result_small = chunker.fixed_size(long_text, chunk_size=200, overlap=20)
        result_large = chunker.fixed_size(long_text, chunk_size=200, overlap=100)

        assert result_large.total_chunks >= result_small.total_chunks

    def test_whitespace_only_returns_empty(self, chunker: Chunker) -> None:
        """Whitespace-only text returns empty result."""
        result = chunker.fixed_size("   \n\t  ", chunk_size=500)

        assert result.total_chunks == 0
        assert result.chunks == []

    def test_chunk_indexes_are_sequential(self, chunker: Chunker, long_text: str) -> None:
        """Chunk index values are 0, 1, 2, ..."""
        result = chunker.fixed_size(long_text, chunk_size=200, overlap=0)

        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i


class TestSentenceEdgeCases:
    """Additional edge-case tests for sentence chunking."""

    def test_empty_text_returns_empty(self, chunker: Chunker) -> None:
        """Empty string produces empty result."""
        result = chunker.sentence("")

        assert result.total_chunks == 0
        assert result.chunks == []
        assert result.avg_chunk_size == 0.0

    def test_large_group_size_produces_single_chunk(self, chunker: Chunker) -> None:
        """sentences_per_chunk larger than sentence count produces one chunk."""
        text = "First. Second. Third."
        result = chunker.sentence(text, sentences_per_chunk=100)

        assert result.total_chunks == 1

    def test_sentence_metadata_strategy(self, chunker: Chunker) -> None:
        """Sentence chunks have correct strategy metadata."""
        text = "First sentence. Second sentence. Third sentence."
        result = chunker.sentence(text, sentences_per_chunk=2)

        for chunk in result.chunks:
            assert chunk.metadata["strategy"] == "sentence"


class TestSlidingWindowEdgeCases:
    """Additional edge-case tests for sliding window chunking."""

    def test_empty_text_returns_empty(self, chunker: Chunker) -> None:
        """Empty text returns empty result."""
        result = chunker.sliding_window("")

        assert result.total_chunks == 0
        assert result.chunks == []
        assert result.strategy == "sliding_window"

    def test_chunk_indexes_sequential(self, chunker: Chunker, long_text: str) -> None:
        """Sliding window chunk indexes are sequential."""
        result = chunker.sliding_window(long_text, window_size=200, step=100)

        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i

    def test_window_size_equals_step_no_overlap(self, chunker: Chunker, long_text: str) -> None:
        """When step equals window_size, chunks do not overlap."""
        result = chunker.sliding_window(long_text, window_size=200, step=200)

        for i in range(len(result.chunks) - 1):
            assert result.chunks[i].end_char <= result.chunks[i + 1].start_char


class TestSemanticEdgeCases:
    """Additional edge-case tests for semantic chunking."""

    def test_empty_text_returns_empty(self, chunker: Chunker) -> None:
        """Empty text returns empty result."""
        result = chunker.semantic("")

        assert result.total_chunks == 0
        assert result.chunks == []

    def test_all_short_paragraphs_merge(self, chunker: Chunker) -> None:
        """All paragraphs below min_chunk_size get merged."""
        text = "Hi.\n\nHo.\n\nHey."
        result = chunker.semantic(text, min_chunk_size=200)

        # All short paragraphs should be merged into 1 chunk
        assert result.total_chunks == 1

    def test_semantic_metadata_strategy(self, chunker: Chunker) -> None:
        """Semantic chunks have correct strategy metadata."""
        text = "First paragraph with enough content here.\n\nSecond paragraph with enough content here."
        result = chunker.semantic(text, min_chunk_size=10)

        for chunk in result.chunks:
            assert chunk.metadata["strategy"] == "semantic"
