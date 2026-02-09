"""Tests for hybrid retrieval module (BM25 + Dense + RRF)."""

from __future__ import annotations

import numpy as np
import pytest

from docqa_engine.ingest import DocumentChunk
from docqa_engine.retriever import (
    BM25Index,
    DenseIndex,
    HybridRetriever,
    SearchResult,
    reciprocal_rank_fusion,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_chunks() -> list[DocumentChunk]:
    """Three chunks with distinct keywords for deterministic BM25 tests."""
    return [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            content="The python programming language is popular for data science.",
            metadata={"source": "doc1.txt"},
        ),
        DocumentChunk(
            chunk_id="c2",
            document_id="d1",
            content="Real estate prices in California have risen sharply this year.",
            metadata={"source": "doc2.txt"},
        ),
        DocumentChunk(
            chunk_id="c3",
            document_id="d1",
            content="Machine learning models require large datasets for training.",
            metadata={"source": "doc3.txt"},
        ),
    ]


@pytest.fixture()
def sample_embeddings() -> np.ndarray:
    """Deterministic 4-dimensional embeddings aligned with sample_chunks."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------


class TestBM25Index:
    """Tests for the BM25 keyword retriever."""

    def test_tokenize_lowercases_and_splits(self) -> None:
        tokens = BM25Index._tokenize("Hello World! Python-3 rocks.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens
        assert "rocks" in tokens

    def test_tokenize_empty_string(self) -> None:
        assert BM25Index._tokenize("") == []

    def test_add_chunks_and_search(self, sample_chunks: list[DocumentChunk]) -> None:
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("python programming", top_k=3)
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk.chunk_id == "c1"
        assert results[0].source == "bm25"

    def test_search_returns_ranked(self, sample_chunks: list[DocumentChunk]) -> None:
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("real estate california prices", top_k=3)
        assert results[0].rank == 1
        if len(results) > 1:
            assert results[1].rank == 2

    def test_search_no_match(self, sample_chunks: list[DocumentChunk]) -> None:
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("xylophone zebra quantum", top_k=5)
        assert results == []

    def test_search_respects_top_k(self, sample_chunks: list[DocumentChunk]) -> None:
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("data science machine learning", top_k=1)
        assert len(results) <= 1

    def test_empty_index_returns_empty(self) -> None:
        idx = BM25Index()
        results = idx.search("anything")
        assert results == []

    def test_scores_descending(self, sample_chunks: list[DocumentChunk]) -> None:
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("data science", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# DenseIndex
# ---------------------------------------------------------------------------


class TestDenseIndex:
    """Tests for the dense (vector) retriever."""

    def test_empty_index_returns_empty(self) -> None:
        idx = DenseIndex()
        query_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert idx.search(query_emb) == []

    def test_add_chunks_and_search(
        self,
        sample_chunks: list[DocumentChunk],
        sample_embeddings: np.ndarray,
    ) -> None:
        idx = DenseIndex()
        idx.add_chunks(sample_chunks, sample_embeddings)

        query_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = idx.search(query_emb, top_k=2)

        assert len(results) >= 1
        assert results[0].chunk.chunk_id == "c1"
        assert results[0].source == "dense"
        assert results[0].score > 0

    def test_search_ranks_correctly(
        self,
        sample_chunks: list[DocumentChunk],
        sample_embeddings: np.ndarray,
    ) -> None:
        idx = DenseIndex()
        idx.add_chunks(sample_chunks, sample_embeddings)

        # Query closest to c2 (second basis vector)
        query_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        results = idx.search(query_emb, top_k=3)

        assert results[0].chunk.chunk_id == "c2"
        assert results[0].rank == 1

    def test_self_retrieval_with_identity(
        self,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        idx = DenseIndex()
        embs = np.eye(len(sample_chunks), dtype=np.float32)
        idx.add_chunks(sample_chunks, embs)
        results = idx.search(embs[2], top_k=1)
        assert results[0].chunk.chunk_id == "c3"


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    """Tests for the RRF combiner."""

    def test_combines_two_result_lists(self, sample_chunks: list[DocumentChunk]) -> None:
        list_a = [
            SearchResult(chunk=sample_chunks[0], score=3.0, rank=1, source="bm25"),
            SearchResult(chunk=sample_chunks[1], score=2.0, rank=2, source="bm25"),
        ]
        list_b = [
            SearchResult(chunk=sample_chunks[1], score=0.9, rank=1, source="dense"),
            SearchResult(chunk=sample_chunks[2], score=0.5, rank=2, source="dense"),
        ]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=10)
        assert len(fused) == 3
        # c2 appears in both lists and should rank highest
        assert fused[0].chunk.chunk_id == "c2"
        assert fused[0].source == "hybrid"

    def test_single_list_passthrough(self, sample_chunks: list[DocumentChunk]) -> None:
        single = [
            SearchResult(chunk=sample_chunks[0], score=5.0, rank=1, source="bm25"),
            SearchResult(chunk=sample_chunks[2], score=1.0, rank=2, source="bm25"),
        ]
        fused = reciprocal_rank_fusion([single], top_k=10)
        assert len(fused) == 2
        assert fused[0].chunk.chunk_id == "c1"

    def test_respects_top_k(self, sample_chunks: list[DocumentChunk]) -> None:
        results = [
            SearchResult(chunk=sample_chunks[i], score=float(3 - i), rank=i + 1, source="bm25") for i in range(3)
        ]
        fused = reciprocal_rank_fusion([results], top_k=1)
        assert len(fused) == 1

    def test_empty_lists(self) -> None:
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    """Tests for the combined BM25 + Dense retriever."""

    @pytest.mark.asyncio
    async def test_bm25_only_fallback(self, sample_chunks: list[DocumentChunk]) -> None:
        retriever = HybridRetriever(embed_fn=None)
        retriever.add_chunks(sample_chunks)
        results = await retriever.search("python programming", top_k=3)
        assert len(results) >= 1
        # Falls back to BM25 when no embeddings provided
        assert results[0].source == "bm25"

    @pytest.mark.asyncio
    async def test_hybrid_with_both_indices(
        self,
        sample_chunks: list[DocumentChunk],
        sample_embeddings: np.ndarray,
    ) -> None:
        async def mock_embed_fn(texts: list[str]) -> np.ndarray:
            # Return the first basis vector for any query
            return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        retriever = HybridRetriever(embed_fn=mock_embed_fn)
        retriever.add_chunks(sample_chunks, embeddings=sample_embeddings)

        results = await retriever.search("python data science", top_k=5)
        assert len(results) >= 1
        # With both indices, results come from RRF fusion
        assert results[0].source == "hybrid"

    @pytest.mark.asyncio
    async def test_custom_dense_backend(
        self,
        sample_chunks: list[DocumentChunk],
        sample_embeddings: np.ndarray,
    ) -> None:
        """HybridRetriever accepts a custom dense_backend."""
        custom_backend = DenseIndex()
        retriever = HybridRetriever(dense_backend=custom_backend)
        assert retriever.dense is custom_backend

    def test_default_dense_backend_is_dense_index(self) -> None:
        retriever = HybridRetriever()
        assert isinstance(retriever.dense, DenseIndex)


class TestDenseIndexResetAndLen:
    """Tests for DenseIndex.reset() and DenseIndex.__len__()."""

    def test_len_empty(self) -> None:
        idx = DenseIndex()
        assert len(idx) == 0

    def test_len_with_data(self, sample_chunks: list[DocumentChunk], sample_embeddings: np.ndarray) -> None:
        idx = DenseIndex()
        idx.add_chunks(sample_chunks, sample_embeddings)
        assert len(idx) == 3

    def test_reset(self, sample_chunks: list[DocumentChunk], sample_embeddings: np.ndarray) -> None:
        idx = DenseIndex()
        idx.add_chunks(sample_chunks, sample_embeddings)
        idx.reset()
        assert len(idx) == 0
        assert idx.embeddings is None
        assert idx.chunks == []
