"""Hybrid Retriever: BM25 (keyword) + dense vectors, Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from docqa_engine.ingest import DocumentChunk


@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float
    rank: int = 0
    source: str = ""  # "bm25", "dense", "hybrid"


# ---------- BM25 ----------


class BM25Index:
    """Okapi BM25 keyword retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[DocumentChunk] = []
        self.tokenized: list[list[str]] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.avg_dl: float = 0.0
        self.idf: dict[str, float] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Index chunks for BM25 search."""
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            self.chunks.append(chunk)
            self.tokenized.append(tokens)
            seen = set()
            for token in tokens:
                if token not in seen:
                    self.doc_freqs[token] += 1
                    seen.add(token)

        total_tokens = sum(len(t) for t in self.tokenized)
        self.avg_dl = total_tokens / max(len(self.tokenized), 1)
        self._compute_idf()

    def _compute_idf(self) -> None:
        n = len(self.tokenized)
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search using BM25 scoring."""
        query_tokens = self._tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self.tokenized):
            score = 0.0
            dl = len(doc_tokens)
            tf_map: dict[str, int] = defaultdict(int)
            for t in doc_tokens:
                tf_map[t] += 1

            for qt in query_tokens:
                if qt in self.idf:
                    tf = tf_map.get(qt, 0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                    score += self.idf[qt] * numerator / max(denominator, 1e-10)

            scores.append(score)

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            SearchResult(chunk=self.chunks[i], score=s, rank=rank + 1, source="bm25")
            for rank, (i, s) in enumerate(indexed)
            if s > 0
        ]


# ---------- Dense (Vector) ----------


class DenseIndex:
    """Dense vector retrieval using cosine similarity."""

    def __init__(self):
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """Index chunks with their embeddings."""
        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def reset(self) -> None:
        """Clear all chunks and embeddings."""
        self.chunks = []
        self.embeddings = None

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return 0 if self.embeddings is None else len(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Search using cosine similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Normalize
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-10)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        doc_norms = self.embeddings / norms

        similarities = doc_norms @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            SearchResult(
                chunk=self.chunks[i],
                score=float(similarities[i]),
                rank=rank + 1,
                source="dense",
            )
            for rank, i in enumerate(top_indices)
            if similarities[i] > 0
        ]


# ---------- Hybrid (RRF) ----------


def reciprocal_rank_fusion(result_lists: list[list[SearchResult]], k: int = 60, top_k: int = 10) -> list[SearchResult]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion."""
    chunk_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, DocumentChunk] = {}

    for results in result_lists:
        for result in results:
            cid = result.chunk.chunk_id
            chunk_scores[cid] += 1.0 / (k + result.rank)
            chunk_map[cid] = result.chunk

    sorted_ids = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        SearchResult(chunk=chunk_map[cid], score=score, rank=rank + 1, source="hybrid")
        for rank, (cid, score) in enumerate(sorted_ids)
    ]


class HybridRetriever:
    """Hybrid BM25 + Dense retriever with Reciprocal Rank Fusion."""

    def __init__(self, embed_fn=None, dense_backend=None):
        self.bm25 = BM25Index()
        self.dense = dense_backend if dense_backend is not None else DenseIndex()
        self.embed_fn = embed_fn  # async callable: list[str] -> np.ndarray

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray | None = None) -> None:
        """Add chunks to both indices."""
        self.bm25.add_chunks(chunks)
        if embeddings is not None:
            self.dense.add_chunks(chunks, embeddings)

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search both indices and fuse results."""
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        dense_results = []
        if self.embed_fn and len(self.dense) > 0:
            query_emb = await self.embed_fn([query])
            dense_results = self.dense.search(query_emb[0], top_k=top_k * 2)

        if not dense_results:
            # BM25-only fallback
            return bm25_results[:top_k]

        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)
