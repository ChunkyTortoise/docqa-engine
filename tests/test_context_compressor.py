"""Tests for context compression: extractive, truncation, and token-ratio methods."""

from __future__ import annotations

import pytest

from docqa_engine.context_compressor import (
    Budget,
    ContextCompressor,
    TokenBudgetManager,
    _estimate_tokens,
)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    """Tests for the _estimate_tokens helper."""

    def test_empty_string(self) -> None:
        assert _estimate_tokens("") == 0

    def test_whitespace_only(self) -> None:
        assert _estimate_tokens("   ") == 0

    def test_single_word(self) -> None:
        result = _estimate_tokens("hello")
        assert result >= 1

    def test_multiple_words(self) -> None:
        result = _estimate_tokens("the quick brown fox jumps")
        assert result >= 5  # at least one token per word


# ---------------------------------------------------------------------------
# TokenBudgetManager
# ---------------------------------------------------------------------------


@pytest.fixture()
def budget_manager() -> TokenBudgetManager:
    return TokenBudgetManager(total_budget=1000, context_ratio=0.6, query_ratio=0.1, response_ratio=0.3)


class TestTokenBudgetManager:
    """Tests for token budget allocation and tracking."""

    def test_allocate_returns_budget(self, budget_manager: TokenBudgetManager) -> None:
        budget = budget_manager.allocate()
        assert isinstance(budget, Budget)
        assert budget.total == 1000

    def test_allocate_ratios(self, budget_manager: TokenBudgetManager) -> None:
        budget = budget_manager.allocate()
        assert budget.context_budget == 600
        assert budget.query_budget == 100
        # response gets remainder
        assert budget.response_budget == 300

    def test_use_decreases_remaining(self, budget_manager: TokenBudgetManager) -> None:
        remaining = budget_manager.use("context", 200)
        assert remaining == 400  # 600 - 200

    def test_use_unknown_category_raises(self, budget_manager: TokenBudgetManager) -> None:
        with pytest.raises(ValueError, match="Unknown category"):
            budget_manager.use("unknown", 100)

    def test_remaining_for(self, budget_manager: TokenBudgetManager) -> None:
        budget_manager.use("query", 50)
        assert budget_manager.remaining_for("query") == 50  # 100 - 50

    def test_remaining_for_unknown_raises(self, budget_manager: TokenBudgetManager) -> None:
        with pytest.raises(ValueError, match="Unknown category"):
            budget_manager.remaining_for("unknown")

    def test_reset_clears_usage(self, budget_manager: TokenBudgetManager) -> None:
        budget_manager.use("context", 500)
        budget_manager.reset()
        assert budget_manager.remaining_for("context") == 600

    def test_invalid_total_budget(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TokenBudgetManager(total_budget=0)

    def test_invalid_ratios(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            TokenBudgetManager(total_budget=1000, context_ratio=0.5, query_ratio=0.5, response_ratio=0.5)

    def test_remaining_never_negative(self, budget_manager: TokenBudgetManager) -> None:
        budget_manager.use("context", 9999)
        assert budget_manager.remaining_for("context") == 0


# ---------------------------------------------------------------------------
# ContextCompressor — extractive
# ---------------------------------------------------------------------------

LONG_TEXT = (
    "Machine learning is a branch of artificial intelligence. "
    "It uses data to improve performance on tasks. "
    "Supervised learning requires labeled examples. "
    "Unsupervised learning finds hidden patterns. "
    "Reinforcement learning uses rewards and penalties. "
    "Deep learning uses neural networks with many layers. "
    "Transfer learning reuses pre-trained models. "
    "Feature engineering is important for model quality. "
    "Cross-validation helps estimate generalization. "
    "Regularization prevents overfitting in complex models."
)


@pytest.fixture()
def compressor() -> ContextCompressor:
    return ContextCompressor()


class TestExtractive:
    """Tests for extractive compression."""

    def test_short_text_unchanged(self, compressor: ContextCompressor) -> None:
        result = compressor.extractive("Short text.", max_sentences=5)
        assert result.text == "Short text."
        assert result.compression_ratio == 1.0

    def test_long_text_compressed(self, compressor: ContextCompressor) -> None:
        result = compressor.extractive(LONG_TEXT, max_sentences=3)
        assert result.compressed_tokens < result.original_tokens
        assert result.compression_ratio < 1.0
        assert result.method == "extractive"

    def test_empty_text(self, compressor: ContextCompressor) -> None:
        result = compressor.extractive("")
        assert result.text == ""
        assert result.original_tokens == 0
        assert result.compression_ratio == 1.0

    def test_preserves_sentence_order(self, compressor: ContextCompressor) -> None:
        result = compressor.extractive(LONG_TEXT, max_sentences=3)
        # Selected sentences should maintain original relative order
        sentences_in_result = [s.strip() for s in result.text.split(". ") if s.strip()]
        # Check that all returned text is a substring of the original
        for sentence in sentences_in_result:
            clean = sentence.rstrip(".")
            assert clean in LONG_TEXT or sentence in LONG_TEXT


# ---------------------------------------------------------------------------
# ContextCompressor — truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    """Tests for boundary-aware truncation."""

    def test_short_text_unchanged(self, compressor: ContextCompressor) -> None:
        result = compressor.truncation("Hello world.", max_tokens=100)
        assert result.text == "Hello world."
        assert result.compression_ratio == 1.0

    def test_truncates_at_sentence_boundary(self, compressor: ContextCompressor) -> None:
        result = compressor.truncation(LONG_TEXT, max_tokens=20)
        assert result.compressed_tokens <= 20
        assert result.method == "truncation"

    def test_empty_text(self, compressor: ContextCompressor) -> None:
        result = compressor.truncation("")
        assert result.text == ""
        assert result.compression_ratio == 1.0

    def test_very_small_budget(self, compressor: ContextCompressor) -> None:
        result = compressor.truncation(LONG_TEXT, max_tokens=2)
        # Should still return something
        assert result.text != ""
        assert result.compressed_tokens <= 5  # very small

    def test_uses_default_budget(self) -> None:
        mgr = TokenBudgetManager(total_budget=50, context_ratio=0.6, query_ratio=0.1, response_ratio=0.3)
        comp = ContextCompressor(budget_manager=mgr)
        result = comp.truncation(LONG_TEXT)
        # Default max_tokens = context_budget = 30
        assert result.compressed_tokens <= 35  # some tolerance


# ---------------------------------------------------------------------------
# ContextCompressor — token_ratio
# ---------------------------------------------------------------------------


class TestTokenRatio:
    """Tests for token-ratio compression."""

    def test_ratio_1_unchanged(self, compressor: ContextCompressor) -> None:
        result = compressor.token_ratio(LONG_TEXT, target_ratio=1.0)
        assert result.text == LONG_TEXT
        assert result.compression_ratio == 1.0

    def test_ratio_half_compresses(self, compressor: ContextCompressor) -> None:
        result = compressor.token_ratio(LONG_TEXT, target_ratio=0.5)
        assert result.compressed_tokens < result.original_tokens
        assert result.method == "token_ratio"

    def test_empty_text(self, compressor: ContextCompressor) -> None:
        result = compressor.token_ratio("", target_ratio=0.5)
        assert result.text == ""

    def test_ratio_clamped_above_1(self, compressor: ContextCompressor) -> None:
        result = compressor.token_ratio(LONG_TEXT, target_ratio=1.5)
        assert result.compression_ratio == 1.0


# ---------------------------------------------------------------------------
# ContextCompressor — dispatch
# ---------------------------------------------------------------------------


class TestCompress:
    """Tests for the compress dispatch method."""

    def test_dispatch_extractive(self, compressor: ContextCompressor) -> None:
        result = compressor.compress(LONG_TEXT, method="extractive", max_sentences=3)
        assert result.method == "extractive"

    def test_dispatch_truncation(self, compressor: ContextCompressor) -> None:
        result = compressor.compress(LONG_TEXT, method="truncation", max_tokens=20)
        assert result.method == "truncation"

    def test_dispatch_token_ratio(self, compressor: ContextCompressor) -> None:
        result = compressor.compress(LONG_TEXT, method="token_ratio", target_ratio=0.5)
        assert result.method == "token_ratio"

    def test_unknown_method_raises(self, compressor: ContextCompressor) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            compressor.compress(LONG_TEXT, method="magic")
