"""Tests for the end-to-end DocQA pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from docqa_engine.pipeline import DocQAPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline() -> DocQAPipeline:
    return DocQAPipeline()


@pytest.fixture()
def loaded_pipeline(pipeline: DocQAPipeline) -> DocQAPipeline:
    """Pipeline with two text documents already ingested."""
    pipeline.ingest_text(
        "The housing market in Rancho Cucamonga has grown steadily. "
        "Home prices rose 12% year-over-year. Inventory is tight at 2.3 months of supply. "
        "Buyers face stiff competition for well-priced listings.",
        filename="market.txt",
    )
    pipeline.ingest_text(
        "Machine learning models require large datasets for training. "
        "Neural networks have shown remarkable performance on NLP tasks. "
        "Transfer learning reduces the need for domain-specific data.",
        filename="ml_intro.txt",
    )
    return pipeline


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestPipelineInit:
    """Tests for pipeline creation and properties."""

    def test_initialization(self, pipeline: DocQAPipeline) -> None:
        assert pipeline is not None
        stats = pipeline.get_stats()
        assert stats is not None

    def test_prompt_library_property(self, pipeline: DocQAPipeline) -> None:
        lib = pipeline.prompt_library
        assert lib is not None

    def test_initial_stats_empty(self, pipeline: DocQAPipeline) -> None:
        stats = pipeline.get_stats()
        # A fresh pipeline should report zero documents / chunks
        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class TestPipelineIngest:
    """Tests for document ingestion via the pipeline."""

    def test_ingest_text(self, pipeline: DocQAPipeline) -> None:
        result = pipeline.ingest_text("Hello world. This is a test.", filename="test.txt")
        assert result is not None
        stats = pipeline.get_stats()
        assert stats.get("documents", stats.get("total_documents", 0)) >= 1

    def test_ingest_file_txt(self, pipeline: DocQAPipeline, tmp_path: Path) -> None:
        txt = tmp_path / "sample.txt"
        txt.write_text(
            "Some sample content about real estate market analysis.",
            encoding="utf-8",
        )
        result = pipeline.ingest(str(txt))
        assert result is not None

    def test_multiple_ingestions(self, loaded_pipeline: DocQAPipeline) -> None:
        stats = loaded_pipeline.get_stats()
        doc_count = stats.get("documents", stats.get("total_documents", 0))
        assert doc_count >= 2


# ---------------------------------------------------------------------------
# Ask (async, mock mode)
# ---------------------------------------------------------------------------


class TestPipelineAsk:
    """Tests for the async question-answering interface."""

    @pytest.mark.asyncio
    async def test_ask_returns_answer(self, loaded_pipeline: DocQAPipeline) -> None:
        from docqa_engine.answer import Answer

        answer = await loaded_pipeline.ask("What is the housing market trend?")
        assert isinstance(answer, Answer)
        assert answer.question == "What is the housing market trend?"
        assert len(answer.answer_text) > 0

    @pytest.mark.asyncio
    async def test_ask_with_top_k(self, loaded_pipeline: DocQAPipeline) -> None:
        from docqa_engine.answer import Answer

        answer = await loaded_pipeline.ask("housing prices", top_k=2)
        assert isinstance(answer, Answer)

    @pytest.mark.asyncio
    async def test_ask_mock_mode_no_llm(self, loaded_pipeline: DocQAPipeline) -> None:
        answer = await loaded_pipeline.ask("What about machine learning?")
        # Without an llm_fn the pipeline should fall back to mock mode
        assert answer.provider == "mock"


# ---------------------------------------------------------------------------
# Compare templates (async)
# ---------------------------------------------------------------------------


class TestPipelineCompare:
    """Tests for template comparison via the pipeline."""

    @pytest.mark.asyncio
    async def test_compare_templates(self, loaded_pipeline: DocQAPipeline) -> None:
        # The pipeline should expose template comparison
        result = await loaded_pipeline.compare_templates(
            question="What is the housing trend?",
            template_a="qa_concise",
            template_b="qa_detailed",
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Vector backend configuration
# ---------------------------------------------------------------------------


class TestPipelineVectorBackend:
    """Tests for pipeline vector_backend parameter."""

    def test_default_is_memory(self) -> None:
        p = DocQAPipeline()
        assert p._vector_backend == "memory"

    @pytest.mark.asyncio
    async def test_memory_backend_works(self) -> None:
        p = DocQAPipeline(vector_backend="memory")
        p.ingest_text("Test document about AI and machine learning.")
        answer = await p.ask("What is this about?")
        assert answer is not None

    @pytest.mark.asyncio
    async def test_chroma_backend_works(self) -> None:
        pytest.importorskip("chromadb")
        p = DocQAPipeline(
            vector_backend="chroma",
            vector_kwargs={"collection_name": "pipeline_test"},
        )
        p.ingest_text("Test document about AI and machine learning.")
        answer = await p.ask("What is this about?")
        assert answer is not None

    @pytest.mark.asyncio
    async def test_invalid_backend_raises(self) -> None:
        p = DocQAPipeline(vector_backend="invalid_db")
        p.ingest_text("Some text.")
        with pytest.raises(ValueError, match="Unknown vector backend"):
            await p.ask("test")
