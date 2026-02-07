"""Tests for multi-LLM answer generation with citations."""

from __future__ import annotations

import pytest

from docqa_engine.answer import (
    Answer,
    Citation,
    build_context,
    build_qa_prompt,
    extract_citations,
    generate_answer,
)
from docqa_engine.ingest import DocumentChunk
from docqa_engine.retriever import SearchResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def search_results() -> list[SearchResult]:
    """Three search results with distinct sources and pages."""
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            content="Home prices rose 12% year-over-year in Rancho Cucamonga.",
            metadata={"source": "market_report.pdf"},
            page_number=3,
        ),
        DocumentChunk(
            chunk_id="c2",
            document_id="d1",
            content="Inventory is at 2.3 months of supply, well below the 6-month balanced mark.",
            metadata={"source": "market_report.pdf"},
            page_number=7,
        ),
        DocumentChunk(
            chunk_id="c3",
            document_id="d2",
            content="New construction permits increased by 18% in the Inland Empire.",
            metadata={"source": "construction_data.csv"},
        ),
    ]
    return [
        SearchResult(chunk=chunks[0], score=0.95, rank=1, source="hybrid"),
        SearchResult(chunk=chunks[1], score=0.82, rank=2, source="hybrid"),
        SearchResult(chunk=chunks[2], score=0.71, rank=3, source="hybrid"),
    ]


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    """Tests for context construction from search results."""

    def test_formats_with_source_annotations(self, search_results: list[SearchResult]) -> None:
        ctx = build_context(search_results)
        assert "[Source 1: market_report.pdf, p.3]" in ctx
        assert "[Source 2: market_report.pdf, p.7]" in ctx
        assert "[Source 3: construction_data.csv]" in ctx

    def test_includes_chunk_content(self, search_results: list[SearchResult]) -> None:
        ctx = build_context(search_results)
        assert "Home prices rose 12%" in ctx
        assert "Inventory is at 2.3 months" in ctx

    def test_respects_max_chars(self, search_results: list[SearchResult]) -> None:
        ctx = build_context(search_results, max_chars=100)
        assert len(ctx) <= 200  # generous bound (header + content of first entry)
        # Should include at most 1-2 sources before hitting the limit
        assert "[Source 1:" in ctx

    def test_empty_results(self) -> None:
        ctx = build_context([])
        assert ctx == ""


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


class TestExtractCitations:
    """Tests for citation extraction from search results."""

    def test_returns_correct_count(self, search_results: list[SearchResult]) -> None:
        citations = extract_citations(search_results)
        assert len(citations) == 3
        assert all(isinstance(c, Citation) for c in citations)

    def test_citation_fields_populated(self, search_results: list[SearchResult]) -> None:
        citations = extract_citations(search_results)
        c = citations[0]
        assert c.chunk_id == "c1"
        assert "Home prices" in c.content_snippet
        assert c.page_number == 3
        assert c.source == "market_report.pdf"
        assert c.relevance_score == pytest.approx(0.95)

    def test_no_page_number(self, search_results: list[SearchResult]) -> None:
        citations = extract_citations(search_results)
        # Third result has no page_number
        assert citations[2].page_number is None

    def test_scores_preserved(self, search_results: list[SearchResult]) -> None:
        citations = extract_citations(search_results)
        assert citations[0].relevance_score == pytest.approx(0.95)
        assert citations[1].relevance_score == pytest.approx(0.82)
        assert citations[2].relevance_score == pytest.approx(0.71)


# ---------------------------------------------------------------------------
# build_qa_prompt
# ---------------------------------------------------------------------------


class TestBuildQaPrompt:
    """Tests for prompt construction."""

    def test_includes_question_and_context(self) -> None:
        prompt = build_qa_prompt("What is the price trend?", "Context: prices rose 12%.")
        assert "What is the price trend?" in prompt
        assert "prices rose 12%" in prompt

    def test_contains_instructions_and_answer_marker(self) -> None:
        prompt = build_qa_prompt("Q?", "C")
        assert "Answer the following question" in prompt
        assert "Answer:" in prompt


# ---------------------------------------------------------------------------
# Citation / Answer dataclass fields
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Tests for Citation and Answer dataclass structures."""

    def test_citation_defaults(self) -> None:
        c = Citation(chunk_id="abc", content_snippet="text")
        assert c.page_number is None
        assert c.source == ""
        assert c.relevance_score == 0.0

    def test_answer_defaults(self) -> None:
        a = Answer(
            question="Q?",
            answer_text="A.",
            citations=[],
            provider="mock",
            model="mock-v1",
        )
        assert a.tokens_used == 0
        assert a.cost_estimate == 0.0
        assert a.metadata == {}


# ---------------------------------------------------------------------------
# generate_answer (async)
# ---------------------------------------------------------------------------


class TestGenerateAnswer:
    """Tests for the async answer generation function."""

    @pytest.mark.asyncio
    async def test_mock_mode(self, search_results: list[SearchResult]) -> None:
        answer = await generate_answer(
            question="What is the housing market trend?",
            results=search_results,
        )
        assert isinstance(answer, Answer)
        assert answer.provider == "mock"
        assert answer.model == "mock-v1"
        assert len(answer.citations) == 3
        assert answer.tokens_used > 0

    @pytest.mark.asyncio
    async def test_custom_llm_fn(self, search_results: list[SearchResult]) -> None:
        async def fake_llm(prompt: str, provider: str):
            return ("Fake answer text.", "test-model", 42, 0.001)

        answer = await generate_answer(
            question="What happened?",
            results=search_results,
            llm_fn=fake_llm,
            provider="test",
        )
        assert answer.answer_text == "Fake answer text."
        assert answer.model == "test-model"
        assert answer.tokens_used == 42
        assert answer.cost_estimate == pytest.approx(0.001)
        assert answer.provider == "test"

    @pytest.mark.asyncio
    async def test_respects_max_context_chars(self, search_results: list[SearchResult]) -> None:
        answer = await generate_answer(
            question="Short context test",
            results=search_results,
            max_context_chars=50,
        )
        assert isinstance(answer, Answer)

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        answer = await generate_answer(
            question="What is Python?",
            results=[],
        )
        assert isinstance(answer, Answer)
        assert len(answer.citations) == 0
