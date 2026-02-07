"""Tests for TF-IDF embedding module."""

from __future__ import annotations

import numpy as np
import pytest

from docqa_engine.embedder import TfidfEmbedder, embed_fn_factory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_texts() -> list[str]:
    return [
        "The real estate market is booming in California.",
        "Machine learning models need training data.",
        "Python programming language is versatile and popular.",
        "Home prices increased twelve percent year over year.",
        "Inventory supply is below the balanced market threshold.",
    ]


@pytest.fixture()
def fitted_embedder(sample_texts: list[str]) -> TfidfEmbedder:
    embedder = TfidfEmbedder(max_features=100)
    embedder.fit(sample_texts)
    return embedder


# ---------------------------------------------------------------------------
# TfidfEmbedder
# ---------------------------------------------------------------------------


class TestTfidfEmbedder:
    """Tests for the TF-IDF vectorizer wrapper."""

    def test_is_fitted_before_fit(self) -> None:
        embedder = TfidfEmbedder()
        assert embedder.is_fitted is False

    def test_is_fitted_after_fit(self, fitted_embedder: TfidfEmbedder) -> None:
        assert fitted_embedder.is_fitted is True

    def test_fit_and_embed(self, fitted_embedder: TfidfEmbedder, sample_texts: list[str]) -> None:
        result = fitted_embedder.embed(sample_texts)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(sample_texts)
        assert result.shape[1] > 0

    def test_embed_shape(self, fitted_embedder: TfidfEmbedder, sample_texts: list[str]) -> None:
        result = fitted_embedder.embed(sample_texts)
        n_docs = len(sample_texts)
        n_features = result.shape[1]
        assert result.shape == (n_docs, n_features)
        assert n_features <= 100  # max_features=100

    def test_embed_query_returns_1d(self, fitted_embedder: TfidfEmbedder) -> None:
        result = fitted_embedder.embed_query("real estate prices")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) > 0

    def test_embed_before_fit_raises(self) -> None:
        embedder = TfidfEmbedder()
        with pytest.raises(Exception):  # RuntimeError or ValueError depending on impl
            embedder.embed(["some text"])

    def test_embed_query_before_fit_raises(self) -> None:
        embedder = TfidfEmbedder()
        with pytest.raises(Exception):
            embedder.embed_query("query")

    def test_max_features_respected(self, sample_texts: list[str]) -> None:
        embedder = TfidfEmbedder(max_features=10)
        embedder.fit(sample_texts)
        result = embedder.embed(sample_texts)
        assert result.shape[1] <= 10


# ---------------------------------------------------------------------------
# embed_fn_factory
# ---------------------------------------------------------------------------


class TestEmbedFnFactory:
    """Tests for the async embedding function factory."""

    def test_returns_callable(self, fitted_embedder: TfidfEmbedder) -> None:
        fn = embed_fn_factory(fitted_embedder)
        assert callable(fn)

    @pytest.mark.asyncio
    async def test_factory_produces_correct_shape(
        self,
        fitted_embedder: TfidfEmbedder,
    ) -> None:
        fn = embed_fn_factory(fitted_embedder)
        texts = ["test query one", "test query two"]
        result = await fn(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2
        assert result.ndim == 2
