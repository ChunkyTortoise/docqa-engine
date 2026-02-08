"""Batch Processing: parallel document ingestion and query execution."""

from __future__ import annotations

import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from docqa_engine.pipeline import DocQAPipeline

# ---------------------------------------------------------------------------
# Result data types
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Result of a single query through the pipeline."""

    query: str
    answer: str
    sources: list[str]
    confidence: float
    elapsed_ms: float


@dataclass
class BatchError:
    """An error that occurred while processing one item in a batch."""

    item: str
    error: str
    traceback: str = ""


@dataclass
class BatchResult:
    """Summary of a batch processing run."""

    total: int
    succeeded: int
    failed: int
    errors: list[BatchError] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------


class BatchProcessor:
    """Process multiple documents or queries in parallel.

    Uses :class:`~concurrent.futures.ThreadPoolExecutor` for document
    ingestion (which is CPU/IO-bound) and :func:`asyncio.gather` for query
    execution (which is already async).

    Args:
        pipeline: A :class:`DocQAPipeline` instance.
        max_workers: Maximum thread pool size for document ingestion.
    """

    def __init__(self, pipeline: DocQAPipeline, max_workers: int = 4) -> None:
        self._pipeline = pipeline
        self._max_workers = max_workers

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def process_documents(
        self,
        file_paths: list[str],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Ingest multiple documents into the pipeline.

        Processes files in parallel using a thread pool.  If one file fails,
        the rest continue.

        Args:
            file_paths: List of file paths to ingest.
            on_progress: Optional callback ``(completed, total)`` called after
                each file finishes (success or failure).

        Returns:
            A :class:`BatchResult` with counts and error details.
        """
        total = len(file_paths)
        if total == 0:
            return BatchResult(total=0, succeeded=0, failed=0, elapsed_seconds=0.0)

        succeeded = 0
        errors: list[BatchError] = []
        start = time.monotonic()

        def _ingest_one(path: str) -> str | BatchError:
            try:
                self._pipeline.ingest(path)
                return path
            except Exception as exc:
                return BatchError(
                    item=path,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(_ingest_one, p): p for p in file_paths}
            completed = 0
            for future in futures:
                result = future.result()
                completed += 1
                if isinstance(result, BatchError):
                    errors.append(result)
                else:
                    succeeded += 1
                if on_progress is not None:
                    on_progress(completed, total)

        elapsed = time.monotonic() - start
        return BatchResult(
            total=total,
            succeeded=succeeded,
            failed=len(errors),
            errors=errors,
            elapsed_seconds=round(elapsed, 3),
        )

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    async def process_queries(
        self,
        queries: list[str],
        on_progress: Callable[[int, int], None] | None = None,
        top_k: int = 5,
        llm_fn: Any = None,
    ) -> list[QueryResult]:
        """Run multiple queries through the pipeline.

        Executes queries concurrently using :func:`asyncio.gather`.

        Args:
            queries: List of question strings.
            on_progress: Optional callback ``(completed, total)`` invoked
                after each query completes.
            top_k: Number of chunks to retrieve per query.
            llm_fn: Optional async LLM callable for answer generation.

        Returns:
            List of :class:`QueryResult` objects (one per query).
        """
        total = len(queries)
        if total == 0:
            return []

        completed_count = 0

        async def _run_one(question: str) -> QueryResult:
            nonlocal completed_count
            t0 = time.monotonic()
            try:
                answer = await self._pipeline.ask(
                    question,
                    top_k=top_k,
                    llm_fn=llm_fn,
                )
                elapsed_ms = (time.monotonic() - t0) * 1000

                sources = [c.source for c in answer.citations if c.source]
                # Deduplicate while preserving order
                seen: set[str] = set()
                unique_sources: list[str] = []
                for s in sources:
                    if s not in seen:
                        seen.add(s)
                        unique_sources.append(s)

                confidence = max(c.relevance_score for c in answer.citations) if answer.citations else 0.0

                result = QueryResult(
                    query=question,
                    answer=answer.answer_text,
                    sources=unique_sources,
                    confidence=round(confidence, 4),
                    elapsed_ms=round(elapsed_ms, 2),
                )
            except Exception as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                result = QueryResult(
                    query=question,
                    answer=f"Error: {exc}",
                    sources=[],
                    confidence=0.0,
                    elapsed_ms=round(elapsed_ms, 2),
                )
            finally:
                completed_count += 1
                if on_progress is not None:
                    on_progress(completed_count, total)

            return result

        results = await asyncio.gather(*[_run_one(q) for q in queries])
        return list(results)
