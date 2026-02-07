"""Tests for the hybrid retriever module."""

import asyncio

import numpy as np
import pytest

from docqa_engine.ingest import DocumentChunk
from docqa_engine.retriever import (
    BM25Index,
    DenseIndex,
    HybridRetriever,
    reciprocal_rank_fusion,
)


def make_chunk(content: str, chunk_id: str = "") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id or f"c_{hash(content) % 10000}",
        document_id="doc1",
        content=content,
    )


@pytest.fixture
def sample_chunks():
    return [
        make_chunk("Python is a programming language used for data science.", "c1"),
        make_chunk("Machine learning models require training data.", "c2"),
        make_chunk("FastAPI is a modern web framework for building APIs.", "c3"),
        make_chunk("PostgreSQL is a relational database management system.", "c4"),
        make_chunk("Redis is an in-memory data structure store for caching.", "c5"),
    ]


class TestBM25Index:
    def test_basic_search(self, sample_chunks):
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("Python programming language")
        assert len(results) > 0
        assert results[0].chunk.chunk_id == "c1"
        assert results[0].source == "bm25"

    def test_no_match(self, sample_chunks):
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("quantum physics entanglement")
        # BM25 may still return some results with low scores
        if results:
            assert results[0].score < 1.0

    def test_top_k(self, sample_chunks):
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("data", top_k=2)
        assert len(results) <= 2

    def test_empty_index(self):
        idx = BM25Index()
        results = idx.search("anything")
        assert results == []

    def test_ranking_order(self, sample_chunks):
        idx = BM25Index()
        idx.add_chunks(sample_chunks)
        results = idx.search("database")
        assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))


class TestDenseIndex:
    def test_basic_search(self, sample_chunks):
        idx = DenseIndex()
        # Random embeddings for testing
        embs = np.random.rand(len(sample_chunks), 64).astype(np.float32)
        idx.add_chunks(sample_chunks, embs)

        query_emb = embs[0]  # Search with first chunk's embedding
        results = idx.search(query_emb, top_k=3)
        assert len(results) <= 3
        assert results[0].source == "dense"

    def test_empty_index(self):
        idx = DenseIndex()
        results = idx.search(np.random.rand(64))
        assert results == []

    def test_self_retrieval(self, sample_chunks):
        idx = DenseIndex()
        embs = np.eye(len(sample_chunks))  # Identity matrix — each chunk unique
        idx.add_chunks(sample_chunks, embs)
        results = idx.search(embs[2], top_k=1)
        assert results[0].chunk.chunk_id == "c3"


class TestReciprocalRankFusion:
    def test_basic(self, sample_chunks):
        list1 = [
            type("SR", (), {"chunk": sample_chunks[0], "score": 0.9, "rank": 1, "source": "a"})(),
            type("SR", (), {"chunk": sample_chunks[1], "score": 0.7, "rank": 2, "source": "a"})(),
        ]
        list2 = [
            type("SR", (), {"chunk": sample_chunks[1], "score": 0.8, "rank": 1, "source": "b"})(),
            type("SR", (), {"chunk": sample_chunks[2], "score": 0.6, "rank": 2, "source": "b"})(),
        ]
        # Use actual SearchResult objects
        from docqa_engine.retriever import SearchResult
        list1 = [SearchResult(chunk=sample_chunks[0], score=0.9, rank=1, source="a"),
                 SearchResult(chunk=sample_chunks[1], score=0.7, rank=2, source="a")]
        list2 = [SearchResult(chunk=sample_chunks[1], score=0.8, rank=1, source="b"),
                 SearchResult(chunk=sample_chunks[2], score=0.6, rank=2, source="b")]

        results = reciprocal_rank_fusion([list1, list2], top_k=3)
        assert len(results) <= 3
        # c2 appears in both lists, should rank high
        chunk_ids = [r.chunk.chunk_id for r in results]
        assert "c2" in chunk_ids

    def test_empty_lists(self):
        results = reciprocal_rank_fusion([[], []])
        assert results == []


class TestHybridRetriever:
    def test_bm25_only(self, sample_chunks):
        retriever = HybridRetriever()
        retriever.add_chunks(sample_chunks)

        results = asyncio.run(
            retriever.search("Python programming")
        )
        assert len(results) > 0

    def test_with_embeddings(self, sample_chunks):
        async def mock_embed(texts):
            return np.random.rand(len(texts), 64).astype(np.float32)

        retriever = HybridRetriever(embed_fn=mock_embed)
        embs = np.random.rand(len(sample_chunks), 64).astype(np.float32)
        retriever.add_chunks(sample_chunks, embs)

        results = asyncio.run(
            retriever.search("Python programming")
        )
        assert len(results) > 0
