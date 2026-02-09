"""Vector Store Adapters: Protocol + ChromaDB + Pinecone backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from docqa_engine.ingest import DocumentChunk
from docqa_engine.retriever import SearchResult


@runtime_checkable
class VectorStore(Protocol):
    """Structural interface for vector storage backends.

    Any class implementing ``add_chunks``, ``search``, ``reset``, and
    ``__len__`` satisfies this protocol — no inheritance required.
    """

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None: ...

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]: ...

    def reset(self) -> None: ...

    def __len__(self) -> int: ...


# ---------- ChromaDB ----------


class ChromaVectorStore:
    """ChromaDB-backed vector store.

    Uses ephemeral mode by default (no persistence) for dev/testing.
    Pass ``persist_directory`` for on-disk persistence.
    """

    def __init__(
        self,
        collection_name: str = "docqa",
        persist_directory: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb required for ChromaDB vector store: pip install chromadb"
            )

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks: dict[str, DocumentChunk] = {}

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks with pre-computed embeddings to ChromaDB."""
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        emb_list = embeddings.tolist()

        self._collection.add(
            ids=ids,
            embeddings=emb_list,
            documents=documents,
        )
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Query ChromaDB by embedding vector."""
        if len(self) == 0:
            return []

        n_results = min(top_k, len(self))
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )

        search_results = []
        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for rank, (chunk_id, distance) in enumerate(zip(ids, distances), start=1):
            if chunk_id in self._chunks:
                # ChromaDB cosine distance = 1 - similarity
                score = max(1.0 - distance, 0.0)
                search_results.append(
                    SearchResult(
                        chunk=self._chunks[chunk_id],
                        score=score,
                        rank=rank,
                        source="dense",
                    )
                )

        return search_results

    def reset(self) -> None:
        """Delete all vectors and chunks."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks.clear()

    def __len__(self) -> int:
        return self._collection.count()


# ---------- Pinecone ----------


class PineconeVectorStore:
    """Pinecone-backed vector store for cloud-hosted vector search."""

    BATCH_SIZE = 100

    def __init__(
        self,
        index_name: str,
        api_key: str,
        environment: str = "us-east-1",
        namespace: str = "",
    ) -> None:
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "pinecone required for Pinecone vector store: pip install pinecone"
            )

        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)
        self._namespace = namespace
        self._chunks: dict[str, DocumentChunk] = {}
        self._count = 0

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """Upsert chunks to Pinecone in batches."""
        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            metadata = {"content": chunk.content[:1000]}
            vectors.append((chunk.chunk_id, emb.tolist(), metadata))
            self._chunks[chunk.chunk_id] = chunk

        # Batch upsert
        for i in range(0, len(vectors), self.BATCH_SIZE):
            batch = vectors[i : i + self.BATCH_SIZE]
            self._index.upsert(vectors=batch, namespace=self._namespace)

        self._count += len(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Query Pinecone by embedding vector."""
        if self._count == 0:
            return []

        response = self._index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            namespace=self._namespace,
        )

        results = []
        for rank, match in enumerate(response.get("matches", []), start=1):
            chunk_id = match["id"]
            if chunk_id in self._chunks:
                results.append(
                    SearchResult(
                        chunk=self._chunks[chunk_id],
                        score=match.get("score", 0.0),
                        rank=rank,
                        source="dense",
                    )
                )

        return results

    def reset(self) -> None:
        """Delete all vectors in the namespace."""
        self._index.delete(delete_all=True, namespace=self._namespace)
        self._chunks.clear()
        self._count = 0

    def __len__(self) -> int:
        return self._count


# ---------- Factory ----------


def create_vector_store(backend: str = "memory", **kwargs: Any) -> Any:
    """Create a vector store backend.

    Args:
        backend: One of "memory", "chroma", "pinecone".
        **kwargs: Backend-specific configuration.

    Returns:
        A vector store instance satisfying the ``VectorStore`` protocol.
    """
    if backend == "memory":
        from docqa_engine.retriever import DenseIndex

        return DenseIndex()
    elif backend == "chroma":
        return ChromaVectorStore(**kwargs)
    elif backend == "pinecone":
        return PineconeVectorStore(**kwargs)
    else:
        raise ValueError(
            f"Unknown vector backend: {backend!r}. "
            f"Choose from 'memory', 'chroma', 'pinecone'."
        )
