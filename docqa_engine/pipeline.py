"""End-to-end Document Q&A Pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docqa_engine.answer import Answer, build_context, generate_answer
from docqa_engine.embedder import TfidfEmbedder, embed_fn_factory
from docqa_engine.ingest import DocumentChunk, IngestResult, ingest_file, ingest_txt
from docqa_engine.prompt_lab import PromptComparison, PromptLibrary, compare_prompts
from docqa_engine.retriever import HybridRetriever
from docqa_engine.vector_store import create_vector_store


class DocQAPipeline:
    """Unified pipeline: ingest -> embed -> retrieve -> answer.

    Ties together every layer of the docqa-engine into a single, easy-to-use
    interface.  The TF-IDF embedder is automatically refitted whenever new
    documents are ingested (lazy, via a dirty flag so repeated ``ask`` calls
    are cheap).
    """

    def __init__(
        self,
        vector_backend: str = "memory",
        vector_kwargs: dict | None = None,
    ) -> None:
        self._embedder = TfidfEmbedder()
        self._retriever: HybridRetriever | None = None
        self._library = PromptLibrary()
        self._chunks: list[DocumentChunk] = []
        self._documents: list[IngestResult] = []
        self._dirty = True
        self._vector_backend = vector_backend
        self._vector_kwargs = vector_kwargs or {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_path: str | Path) -> IngestResult:
        """Ingest a file (PDF, DOCX, TXT, CSV) into the pipeline."""
        result = ingest_file(file_path)
        self._documents.append(result)
        self._chunks.extend(result.chunks)
        self._dirty = True
        return result

    def ingest_text(self, text: str, filename: str = "document.txt") -> IngestResult:
        """Ingest raw text content."""
        result = ingest_txt(text, filename=filename)
        self._documents.append(result)
        self._chunks.extend(result.chunks)
        self._dirty = True
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_fitted(self) -> None:
        """Refit the embedder and rebuild retriever indices when chunks change."""
        if not self._dirty or not self._chunks:
            return

        texts = [c.content for c in self._chunks]
        self._embedder.fit(texts)
        embeddings = self._embedder.embed(texts)

        dense_backend = create_vector_store(self._vector_backend, **self._vector_kwargs)
        embed_fn = embed_fn_factory(self._embedder)
        self._retriever = HybridRetriever(embed_fn=embed_fn, dense_backend=dense_backend)
        self._retriever.add_chunks(self._chunks, embeddings)
        self._dirty = False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def ask(
        self,
        question: str,
        top_k: int = 5,
        template: str = "qa_detailed",
        llm_fn: Any = None,
    ) -> Answer:
        """Full pipeline: retrieve relevant chunks and generate an answer.

        Args:
            question: The user's natural-language question.
            top_k: Number of chunks to retrieve.
            template: Name of the prompt template to use (from the PromptLibrary).
            llm_fn: Async LLM callable ``(prompt, provider) -> (text, model, tokens, cost)``.
                     Pass ``None`` for mock mode.
        """
        await self._ensure_fitted()
        assert self._retriever is not None  # noqa: S101

        results = await self._retriever.search(question, top_k=top_k)

        # Render the chosen template and override the default prompt
        context = build_context(results)
        rendered = self._library.render(template, context=context, question=question)

        if llm_fn is not None:

            async def _override(prompt: str, provider: str) -> tuple:
                return await llm_fn(rendered, provider)

            return await generate_answer(question, results, llm_fn=_override)

        return await generate_answer(question, results, llm_fn=None)

    async def compare_templates(
        self,
        question: str,
        template_a: str,
        template_b: str,
        top_k: int = 5,
        llm_fn: Any = None,
    ) -> PromptComparison:
        """A/B test two prompt templates on the same question and retrieval set."""
        await self._ensure_fitted()
        assert self._retriever is not None  # noqa: S101

        results = await self._retriever.search(question, top_k=top_k)
        return await compare_prompts(
            question,
            results,
            template_a,
            template_b,
            self._library,
            llm_fn=llm_fn,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return document count, chunk count, and total characters ingested."""
        doc_count = len(self._documents)
        return {
            "documents": doc_count,
            "document_count": doc_count,
            "total_documents": doc_count,
            "chunk_count": len(self._chunks),
            "total_chars": sum(d.total_chars for d in self._documents),
            "embedder_fitted": self._embedder.is_fitted,
        }

    @property
    def prompt_library(self) -> PromptLibrary:
        """Access the prompt template library for listing or adding templates."""
        return self._library
