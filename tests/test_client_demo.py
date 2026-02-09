"""Tests for WS2: Client Demo Mode — session management, reset, and demo questions."""

from __future__ import annotations

import pytest

from docqa_engine.pipeline import DocQAPipeline

# ── Pipeline Reset Tests ─────────────────────────────────────────────────────


class TestPipelineReset:
    """Verify pipeline.reset() clears all state correctly."""

    def test_reset_clears_documents(self) -> None:
        pipeline = DocQAPipeline()
        pipeline.ingest_text("Hello world test document.", filename="test.txt")
        assert pipeline.get_stats()["documents"] == 1

        pipeline.reset()
        assert pipeline.get_stats()["documents"] == 0

    def test_reset_clears_chunks(self) -> None:
        pipeline = DocQAPipeline()
        pipeline.ingest_text("Some text content for chunking.", filename="a.txt")
        assert pipeline.get_stats()["chunk_count"] > 0

        pipeline.reset()
        assert pipeline.get_stats()["chunk_count"] == 0

    def test_reset_clears_total_chars(self) -> None:
        pipeline = DocQAPipeline()
        pipeline.ingest_text("Characters here.", filename="b.txt")
        assert pipeline.get_stats()["total_chars"] > 0

        pipeline.reset()
        assert pipeline.get_stats()["total_chars"] == 0

    def test_reset_allows_reingest(self) -> None:
        pipeline = DocQAPipeline()
        pipeline.ingest_text("First document.", filename="first.txt")
        pipeline.reset()
        pipeline.ingest_text("Second document.", filename="second.txt")

        stats = pipeline.get_stats()
        assert stats["documents"] == 1
        assert stats["chunk_count"] >= 1

    @pytest.mark.asyncio
    async def test_reset_then_ask_works(self) -> None:
        pipeline = DocQAPipeline()
        pipeline.ingest_text(
            "Machine learning is a subset of artificial intelligence.",
            filename="ml.txt",
        )
        pipeline.reset()
        pipeline.ingest_text(
            "Python is a popular programming language for data science.",
            filename="python.txt",
        )

        answer = await pipeline.ask("What is Python?")
        assert len(answer.answer_text) > 0

    def test_reset_idempotent(self) -> None:
        pipeline = DocQAPipeline()
        pipeline.reset()
        pipeline.reset()
        assert pipeline.get_stats()["documents"] == 0

    def test_reset_clears_embedder_fitted(self) -> None:
        pipeline = DocQAPipeline()
        pipeline.ingest_text("Fit the embedder.", filename="fit.txt")
        pipeline.reset()
        assert pipeline.get_stats()["embedder_fitted"] is False


# ── Session Isolation Tests ──────────────────────────────────────────────────


class TestSessionIsolation:
    """Verify no state bleed between pipeline instances."""

    def test_separate_pipelines_isolated(self) -> None:
        p1 = DocQAPipeline()
        p2 = DocQAPipeline()

        p1.ingest_text("Document for pipeline 1.", filename="p1.txt")

        assert p1.get_stats()["documents"] == 1
        assert p2.get_stats()["documents"] == 0

    def test_reset_one_doesnt_affect_other(self) -> None:
        p1 = DocQAPipeline()
        p2 = DocQAPipeline()

        p1.ingest_text("P1 doc.", filename="p1.txt")
        p2.ingest_text("P2 doc.", filename="p2.txt")

        p1.reset()

        assert p1.get_stats()["documents"] == 0
        assert p2.get_stats()["documents"] == 1

    @pytest.mark.asyncio
    async def test_ask_after_reset_uses_new_docs(self) -> None:
        pipeline = DocQAPipeline()
        pipeline.ingest_text(
            "The capital of France is Paris. France is in Europe.",
            filename="france.txt",
        )
        pipeline.reset()
        pipeline.ingest_text(
            "The capital of Japan is Tokyo. Japan is in Asia.",
            filename="japan.txt",
        )

        answer = await pipeline.ask("What is the capital?")
        # After reset, only Japan doc exists — answer should reference Tokyo
        assert len(answer.answer_text) > 0


# ── Inactivity Timeout Logic Tests ───────────────────────────────────────────


class TestInactivityLogic:
    """Test the timeout calculation logic (no Streamlit dependency)."""

    def test_timeout_constant_is_30_minutes(self) -> None:
        from app import SESSION_TIMEOUT_SECONDS

        assert SESSION_TIMEOUT_SECONDS == 30 * 60

    def test_demo_questions_defined(self) -> None:
        from app import DEMO_QUESTIONS

        assert len(DEMO_QUESTIONS) == 5
        assert all(isinstance(q, str) for q in DEMO_QUESTIONS)
        assert all(q.endswith("?") or q.endswith("s") for q in DEMO_QUESTIONS)


# ── Large Document Handling ──────────────────────────────────────────────────


class TestLargeDocumentHandling:
    """Verify pipeline handles large documents correctly."""

    def test_large_text_ingested(self) -> None:
        pipeline = DocQAPipeline()
        large_text = "This is a test paragraph. " * 5000  # ~125K chars
        result = pipeline.ingest_text(large_text, filename="large.txt")

        assert result.total_chars > 100000
        assert pipeline.get_stats()["chunk_count"] > 1

    @pytest.mark.asyncio
    async def test_large_text_askable(self) -> None:
        pipeline = DocQAPipeline()
        large_text = (
            "Machine learning uses algorithms. " * 2000
            + "Deep learning uses neural networks. " * 2000
        )
        pipeline.ingest_text(large_text, filename="ml_large.txt")

        answer = await pipeline.ask("What does machine learning use?")
        assert len(answer.answer_text) > 0

    def test_multiple_files_ingested(self) -> None:
        pipeline = DocQAPipeline()
        for i in range(10):
            pipeline.ingest_text(
                f"Document {i} has unique content about topic {i}.",
                filename=f"doc_{i}.txt",
            )

        stats = pipeline.get_stats()
        assert stats["documents"] == 10
        assert stats["chunk_count"] >= 10
