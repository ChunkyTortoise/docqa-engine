"""Lightweight Embedder: TF-IDF vectors for zero-config document retrieval."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    """TF-IDF based document embedder.

    Wraps scikit-learn's ``TfidfVectorizer`` to produce dense numpy arrays
    consumable by :class:`docqa_engine.retriever.DenseIndex`.
    """

    def __init__(self, max_features: int = 5000) -> None:
        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, texts: list[str]) -> None:
        """Fit the TF-IDF vocabulary on a corpus of texts."""
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        """Transform texts into dense TF-IDF vectors.

        Returns:
            2-D ``np.ndarray`` of shape ``(len(texts), max_features)``.
        """
        if not self._fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")
        return self._vectorizer.transform(texts).toarray()

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Returns:
            1-D ``np.ndarray`` of shape ``(max_features,)``.
        """
        if not self._fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")
        return self._vectorizer.transform([query]).toarray()[0]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the vectorizer has been fitted on a corpus."""
        return self._fitted


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def embed_fn_factory(
    embedder: TfidfEmbedder,
) -> Callable[[list[str]], np.ndarray]:
    """Return an async callable compatible with ``HybridRetriever.embed_fn``.

    Usage::

        fn = embed_fn_factory(embedder)
        retriever = HybridRetriever(embed_fn=fn)
    """

    async def _embed(texts: list[str]) -> np.ndarray:
        return embedder.embed(texts)

    return _embed
