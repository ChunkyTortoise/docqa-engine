"""Tests for vector store adapters (Protocol, ChromaDB, Pinecone, factory)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from docqa_engine.ingest import DocumentChunk
from docqa_engine.retriever import DenseIndex, SearchResult
from docqa_engine.vector_store import (
    VectorStore,
    create_vector_store,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(chunk_id="c1", document_id="d1", content="Python programming language"),
        DocumentChunk(chunk_id="c2", document_id="d1", content="Real estate market trends"),
        DocumentChunk(chunk_id="c3", document_id="d1", content="Machine learning models"),
    ]


@pytest.fixture()
def sample_embeddings() -> np.ndarray:
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


class TestVectorStoreProtocol:
    """Verify protocol structural subtyping."""

    def test_dense_index_satisfies_protocol(self) -> None:
        idx = DenseIndex()
        assert isinstance(idx, VectorStore)

    def test_chroma_satisfies_protocol(self) -> None:
        chromadb = pytest.importorskip("chromadb")  # noqa: F841
        from docqa_engine.vector_store import ChromaVectorStore

        store = ChromaVectorStore(collection_name="test_proto")
        assert isinstance(store, VectorStore)


# ---------------------------------------------------------------------------
# DenseIndex extensions
# ---------------------------------------------------------------------------


class TestDenseIndexExtensions:
    """Tests for reset() and __len__() added to DenseIndex."""

    def test_len_empty(self) -> None:
        idx = DenseIndex()
        assert len(idx) == 0

    def test_len_after_add(self, sample_chunks, sample_embeddings) -> None:
        idx = DenseIndex()
        idx.add_chunks(sample_chunks, sample_embeddings)
        assert len(idx) == 3

    def test_reset_clears_state(self, sample_chunks, sample_embeddings) -> None:
        idx = DenseIndex()
        idx.add_chunks(sample_chunks, sample_embeddings)
        assert len(idx) == 3

        idx.reset()
        assert len(idx) == 0
        assert idx.chunks == []
        assert idx.embeddings is None

    def test_len_after_reset(self, sample_chunks, sample_embeddings) -> None:
        idx = DenseIndex()
        idx.add_chunks(sample_chunks, sample_embeddings)
        idx.reset()
        assert len(idx) == 0

    def test_search_after_reset_returns_empty(self, sample_chunks, sample_embeddings) -> None:
        idx = DenseIndex()
        idx.add_chunks(sample_chunks, sample_embeddings)
        idx.reset()
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert idx.search(query) == []


# ---------------------------------------------------------------------------
# ChromaDB Vector Store
# ---------------------------------------------------------------------------


class TestChromaVectorStore:
    """Tests for ChromaDB adapter using real ephemeral client."""

    @pytest.fixture(autouse=True)
    def _skip_without_chromadb(self) -> None:
        pytest.importorskip("chromadb")

    @pytest.fixture()
    def chroma_store(self):
        from docqa_engine.vector_store import ChromaVectorStore

        return ChromaVectorStore(collection_name="test_chroma")

    def test_add_and_search(self, chroma_store, sample_chunks, sample_embeddings) -> None:
        chroma_store.add_chunks(sample_chunks, sample_embeddings)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = chroma_store.search(query, top_k=2)

        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk.chunk_id == "c1"
        assert results[0].source == "dense"

    def test_empty_search(self, chroma_store) -> None:
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = chroma_store.search(query, top_k=5)
        assert results == []

    def test_top_k_respected(self, chroma_store, sample_chunks, sample_embeddings) -> None:
        chroma_store.add_chunks(sample_chunks, sample_embeddings)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = chroma_store.search(query, top_k=1)
        assert len(results) == 1

    def test_reset(self, chroma_store, sample_chunks, sample_embeddings) -> None:
        chroma_store.add_chunks(sample_chunks, sample_embeddings)
        assert len(chroma_store) == 3
        chroma_store.reset()
        assert len(chroma_store) == 0

    def test_len(self, chroma_store, sample_chunks, sample_embeddings) -> None:
        assert len(chroma_store) == 0
        chroma_store.add_chunks(sample_chunks, sample_embeddings)
        assert len(chroma_store) == 3

    def test_cosine_scores_range(self, chroma_store, sample_chunks, sample_embeddings) -> None:
        chroma_store.add_chunks(sample_chunks, sample_embeddings)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = chroma_store.search(query, top_k=3)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_persistence(self, tmp_path, sample_chunks, sample_embeddings) -> None:
        from docqa_engine.vector_store import ChromaVectorStore

        persist_dir = str(tmp_path / "chroma_persist")
        store1 = ChromaVectorStore(
            collection_name="persist_test", persist_directory=persist_dir
        )
        store1.add_chunks(sample_chunks, sample_embeddings)
        assert len(store1) == 3

        # Re-open from disk — data should still be there
        store2 = ChromaVectorStore(
            collection_name="persist_test", persist_directory=persist_dir
        )
        assert len(store2) == 3


# ---------------------------------------------------------------------------
# Pinecone Vector Store (fully mocked)
# ---------------------------------------------------------------------------


class TestPineconeVectorStore:
    """Tests for Pinecone adapter with mocked client."""

    @pytest.fixture()
    def mock_pinecone(self):
        mock_index = MagicMock()
        mock_index.query.return_value = {
            "matches": [
                {"id": "c1", "score": 0.95},
                {"id": "c2", "score": 0.80},
            ]
        }
        mock_index.upsert.return_value = None
        mock_index.delete.return_value = None

        mock_pc = MagicMock()
        mock_pc.Index.return_value = mock_index

        return mock_pc, mock_index

    @pytest.fixture()
    def pinecone_store(self, mock_pinecone):
        mock_pc, _ = mock_pinecone
        with patch("docqa_engine.vector_store.PineconeVectorStore.__init__", return_value=None) as _:
            from docqa_engine.vector_store import PineconeVectorStore

            store = PineconeVectorStore.__new__(PineconeVectorStore)
            store._pc = mock_pc
            store._index = mock_pc.Index("test-index")
            store._namespace = ""
            store._chunks = {}
            store._count = 0
        return store

    def test_add_calls_upsert(self, pinecone_store, sample_chunks, sample_embeddings, mock_pinecone) -> None:
        _, mock_index = mock_pinecone
        pinecone_store.add_chunks(sample_chunks, sample_embeddings)
        mock_index.upsert.assert_called_once()
        assert pinecone_store._count == 3

    def test_batch_upsert_250_items(self, pinecone_store, mock_pinecone) -> None:
        _, mock_index = mock_pinecone
        chunks = [
            DocumentChunk(chunk_id=f"c{i}", document_id="d1", content=f"chunk {i}")
            for i in range(250)
        ]
        embeddings = np.random.rand(250, 4).astype(np.float32)
        pinecone_store.add_chunks(chunks, embeddings)
        # 250 / 100 = 3 batches
        assert mock_index.upsert.call_count == 3

    def test_search_returns_results(self, pinecone_store, sample_chunks, sample_embeddings) -> None:
        pinecone_store.add_chunks(sample_chunks, sample_embeddings)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = pinecone_store.search(query, top_k=2)
        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[0].chunk.chunk_id == "c1"

    def test_reset(self, pinecone_store, sample_chunks, sample_embeddings, mock_pinecone) -> None:
        _, mock_index = mock_pinecone
        pinecone_store.add_chunks(sample_chunks, sample_embeddings)
        pinecone_store.reset()
        mock_index.delete.assert_called_once_with(delete_all=True, namespace="")
        assert len(pinecone_store) == 0

    def test_len(self, pinecone_store, sample_chunks, sample_embeddings) -> None:
        assert len(pinecone_store) == 0
        pinecone_store.add_chunks(sample_chunks, sample_embeddings)
        assert len(pinecone_store) == 3

    def test_search_empty_returns_empty(self, pinecone_store) -> None:
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = pinecone_store.search(query, top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateVectorStore:
    """Tests for the create_vector_store factory function."""

    def test_memory_backend(self) -> None:
        store = create_vector_store("memory")
        assert isinstance(store, DenseIndex)
        assert isinstance(store, VectorStore)

    def test_chroma_backend(self) -> None:
        pytest.importorskip("chromadb")
        store = create_vector_store("chroma", collection_name="factory_test")
        assert isinstance(store, VectorStore)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown vector backend"):
            create_vector_store("milvus")


# ---------------------------------------------------------------------------
# HybridRetriever with custom backends
# ---------------------------------------------------------------------------


class TestHybridRetrieverWithBackends:
    """Integration tests for HybridRetriever with pluggable backends."""

    @pytest.mark.asyncio
    async def test_default_is_dense_index(self) -> None:
        from docqa_engine.retriever import HybridRetriever

        retriever = HybridRetriever()
        assert isinstance(retriever.dense, DenseIndex)

    @pytest.mark.asyncio
    async def test_custom_mock_backend(self, sample_chunks, sample_embeddings) -> None:
        from docqa_engine.retriever import HybridRetriever

        # Create a mock backend that satisfies the protocol
        mock_backend = MagicMock(spec=["add_chunks", "search", "reset", "__len__"])
        mock_backend.__len__ = MagicMock(return_value=3)
        mock_backend.search.return_value = [
            SearchResult(chunk=sample_chunks[0], score=0.9, rank=1, source="dense"),
        ]

        async def mock_embed(texts):
            return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        retriever = HybridRetriever(embed_fn=mock_embed, dense_backend=mock_backend)
        retriever.add_chunks(sample_chunks, sample_embeddings)

        results = await retriever.search("python", top_k=3)
        assert len(results) >= 1
        mock_backend.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_with_chroma_backend(self, sample_chunks, sample_embeddings) -> None:
        pytest.importorskip("chromadb")
        from docqa_engine.retriever import HybridRetriever
        from docqa_engine.vector_store import ChromaVectorStore

        chroma = ChromaVectorStore(collection_name="hybrid_test")

        async def mock_embed(texts):
            return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        retriever = HybridRetriever(embed_fn=mock_embed, dense_backend=chroma)
        retriever.add_chunks(sample_chunks, sample_embeddings)

        results = await retriever.search("python programming", top_k=3)
        assert len(results) >= 1
