"""Tests for batch document ingestion and query processing."""

from __future__ import annotations

from pathlib import Path

import pytest

from docqa_engine.batch import BatchProcessor, BatchResult, QueryResult
from docqa_engine.pipeline import DocQAPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline() -> DocQAPipeline:
    return DocQAPipeline()


@pytest.fixture()
def loaded_pipeline(pipeline: DocQAPipeline) -> DocQAPipeline:
    """Pipeline with documents already ingested for query tests."""
    pipeline.ingest_text(
        "The housing market in Rancho Cucamonga has grown steadily. "
        "Home prices rose 12% year-over-year.",
        filename="market.txt",
    )
    pipeline.ingest_text(
        "Machine learning models require large datasets for training. "
        "Neural networks have shown remarkable performance on NLP tasks.",
        filename="ml_intro.txt",
    )
    return pipeline


@pytest.fixture()
def processor(pipeline: DocQAPipeline) -> BatchProcessor:
    return BatchProcessor(pipeline, max_workers=2)


@pytest.fixture()
def loaded_processor(loaded_pipeline: DocQAPipeline) -> BatchProcessor:
    return BatchProcessor(loaded_pipeline, max_workers=2)


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------


class TestBatchDocuments:
    """Tests for process_documents."""

    def test_batch_ingest_multiple_files(
        self, processor: BatchProcessor, tmp_path: Path
    ) -> None:
        """Successfully ingest multiple text files in batch."""
        paths = []
        for i in range(3):
            p = tmp_path / f"doc_{i}.txt"
            p.write_text(f"Document {i} content about topic {i}.", encoding="utf-8")
            paths.append(str(p))

        result = processor.process_documents(paths)

        assert isinstance(result, BatchResult)
        assert result.total == 3
        assert result.succeeded == 3
        assert result.failed == 0
        assert result.errors == []
        assert result.elapsed_seconds >= 0.0

    def test_partial_failure_recovery(
        self, processor: BatchProcessor, tmp_path: Path
    ) -> None:
        """One bad file should not block the rest."""
        good_file = tmp_path / "good.txt"
        good_file.write_text("Valid document content.", encoding="utf-8")

        paths = [str(good_file), "/nonexistent/bad_file.txt"]
        result = processor.process_documents(paths)

        assert result.total == 2
        assert result.succeeded == 1
        assert result.failed == 1
        assert len(result.errors) == 1
        assert "bad_file" in result.errors[0].item

    def test_progress_callback(
        self, processor: BatchProcessor, tmp_path: Path
    ) -> None:
        """Progress callback is invoked for each file."""
        paths = []
        for i in range(2):
            p = tmp_path / f"cb_{i}.txt"
            p.write_text(f"Callback test doc {i}.", encoding="utf-8")
            paths.append(str(p))

        progress_log: list[tuple[int, int]] = []
        processor.process_documents(paths, on_progress=lambda c, t: progress_log.append((c, t)))

        assert len(progress_log) == 2
        # Last call should report 2/2
        totals = [t for _, t in progress_log]
        assert all(t == 2 for t in totals)
        completeds = sorted(c for c, _ in progress_log)
        assert completeds == [1, 2]

    def test_empty_batch(self, processor: BatchProcessor) -> None:
        """Empty file list returns zero counts."""
        result = processor.process_documents([])

        assert result.total == 0
        assert result.succeeded == 0
        assert result.failed == 0
        assert result.elapsed_seconds == 0.0

    def test_single_item_batch(
        self, processor: BatchProcessor, tmp_path: Path
    ) -> None:
        """Batch with one file works correctly."""
        p = tmp_path / "single.txt"
        p.write_text("Single doc batch test.", encoding="utf-8")

        result = processor.process_documents([str(p)])

        assert result.total == 1
        assert result.succeeded == 1
        assert result.failed == 0


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------


class TestBatchQueries:
    """Tests for process_queries (async)."""

    @pytest.mark.asyncio
    async def test_batch_queries(self, loaded_processor: BatchProcessor) -> None:
        """Run multiple queries and get results back."""
        queries = ["What is the housing market trend?", "What about machine learning?"]
        results = await loaded_processor.process_queries(queries)

        assert len(results) == 2
        for r in results:
            assert isinstance(r, QueryResult)
            assert len(r.query) > 0
            assert len(r.answer) > 0
            assert r.elapsed_ms >= 0.0

    @pytest.mark.asyncio
    async def test_query_progress_callback(self, loaded_processor: BatchProcessor) -> None:
        """Progress callback fires for each query."""
        progress_log: list[tuple[int, int]] = []
        results = await loaded_processor.process_queries(
            ["housing prices"],
            on_progress=lambda c, t: progress_log.append((c, t)),
        )

        assert len(results) == 1
        assert len(progress_log) == 1
        assert progress_log[0] == (1, 1)

    @pytest.mark.asyncio
    async def test_empty_query_batch(self, loaded_processor: BatchProcessor) -> None:
        """Empty query list returns empty results."""
        results = await loaded_processor.process_queries([])
        assert results == []
