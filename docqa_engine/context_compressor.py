"""Context Compression: fit long conversations into token budgets while preserving semantics."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class CompressedContext:
    text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method: str


@dataclass
class Budget:
    total: int
    context_budget: int
    query_budget: int
    response_budget: int
    remaining: int


class TokenBudgetManager:
    """Allocate token budget across context, query, and response."""

    def __init__(
        self,
        total_budget: int = 4096,
        context_ratio: float = 0.6,
        query_ratio: float = 0.1,
        response_ratio: float = 0.3,
    ):
        if total_budget <= 0:
            raise ValueError("total_budget must be positive")
        ratios = context_ratio + query_ratio + response_ratio
        if abs(ratios - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")

        self.total_budget = total_budget
        self.context_ratio = context_ratio
        self.query_ratio = query_ratio
        self.response_ratio = response_ratio
        self._used: dict[str, int] = {"context": 0, "query": 0, "response": 0}

    def allocate(self) -> Budget:
        """Return a fresh budget allocation."""
        context_budget = int(self.total_budget * self.context_ratio)
        query_budget = int(self.total_budget * self.query_ratio)
        response_budget = self.total_budget - context_budget - query_budget
        return Budget(
            total=self.total_budget,
            context_budget=context_budget,
            query_budget=query_budget,
            response_budget=response_budget,
            remaining=self.total_budget - sum(self._used.values()),
        )

    def use(self, category: str, tokens: int) -> int:
        """Record token usage in a category. Returns remaining in that category."""
        if category not in self._used:
            raise ValueError(f"Unknown category: {category}")
        self._used[category] += tokens
        budget = self.allocate()
        limit = getattr(budget, f"{category}_budget")
        return max(0, limit - self._used[category])

    def remaining_for(self, category: str) -> int:
        """Return remaining tokens for a category."""
        if category not in self._used:
            raise ValueError(f"Unknown category: {category}")
        budget = self.allocate()
        limit = getattr(budget, f"{category}_budget")
        return max(0, limit - self._used[category])

    def reset(self) -> None:
        """Reset all usage counters."""
        self._used = {"context": 0, "query": 0, "response": 0}


def _estimate_tokens(text: str) -> int:
    """Estimate token count as word count * 1.3 (rough approximation)."""
    if not text or not text.strip():
        return 0
    return max(1, int(len(text.split()) * 1.3))


class ContextCompressor:
    """Compress context using extractive, truncation, or token-ratio methods."""

    def __init__(self, budget_manager: TokenBudgetManager | None = None):
        self.budget_manager = budget_manager or TokenBudgetManager()

    def extractive(self, text: str, max_sentences: int = 5) -> CompressedContext:
        """Select key sentences using TF-IDF importance scoring.

        Ranks sentences by their average TF-IDF weight and selects the top-k.
        """
        original_tokens = _estimate_tokens(text)

        if not text or not text.strip():
            return CompressedContext(
                text="",
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                method="extractive",
            )

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= max_sentences:
            return CompressedContext(
                text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                method="extractive",
            )

        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            truncated = " ".join(sentences[:max_sentences])
            compressed_tokens = _estimate_tokens(truncated)
            ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            return CompressedContext(
                text=truncated,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=ratio,
                method="extractive",
            )

        # Score each sentence by average TF-IDF weight
        scores = tfidf_matrix.mean(axis=1).A1

        # Get indices of top sentences, preserving original order
        ranked_indices = scores.argsort()[::-1][:max_sentences]
        selected_indices = sorted(ranked_indices)

        selected = [sentences[i] for i in selected_indices]
        compressed_text = " ".join(selected)
        compressed_tokens = _estimate_tokens(compressed_text)
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        return CompressedContext(
            text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            method="extractive",
        )

    def truncation(self, text: str, max_tokens: int | None = None) -> CompressedContext:
        """Smart boundary-aware truncation at sentence boundaries.

        Keeps full sentences up to the token budget.
        """
        original_tokens = _estimate_tokens(text)

        if not text or not text.strip():
            return CompressedContext(
                text="",
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                method="truncation",
            )

        if max_tokens is None:
            budget = self.budget_manager.allocate()
            max_tokens = budget.context_budget

        if original_tokens <= max_tokens:
            return CompressedContext(
                text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                method="truncation",
            )

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        kept: list[str] = []
        running_tokens = 0

        for sentence in sentences:
            sentence_tokens = _estimate_tokens(sentence)
            if running_tokens + sentence_tokens > max_tokens:
                break
            kept.append(sentence)
            running_tokens += sentence_tokens

        if not kept:
            # At least include a truncated first sentence
            words = text.split()
            # Approximate: tokens / 1.3 = words
            max_words = max(1, int(max_tokens / 1.3))
            truncated = " ".join(words[:max_words])
            compressed_tokens = _estimate_tokens(truncated)
            ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            return CompressedContext(
                text=truncated,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=ratio,
                method="truncation",
            )

        compressed_text = " ".join(kept)
        compressed_tokens = _estimate_tokens(compressed_text)
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        return CompressedContext(
            text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            method="truncation",
        )

    def token_ratio(self, text: str, target_ratio: float = 0.5) -> CompressedContext:
        """Compress to a target ratio of original token count.

        Uses extractive compression with a calculated sentence count.
        """
        original_tokens = _estimate_tokens(text)

        if not text or not text.strip():
            return CompressedContext(
                text="",
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                method="token_ratio",
            )

        target_ratio = max(0.0, min(1.0, target_ratio))

        if target_ratio >= 1.0:
            return CompressedContext(
                text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                method="token_ratio",
            )

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        target_sentences = max(1, int(len(sentences) * target_ratio))

        result = self.extractive(text, max_sentences=target_sentences)
        return CompressedContext(
            text=result.text,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            compression_ratio=result.compression_ratio,
            method="token_ratio",
        )

    def compress(
        self,
        text: str,
        method: str = "extractive",
        **kwargs: object,
    ) -> CompressedContext:
        """Dispatch to the appropriate compression method."""
        methods = {
            "extractive": self.extractive,
            "truncation": self.truncation,
            "token_ratio": self.token_ratio,
        }
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(methods)}")
        return methods[method](text, **kwargs)
