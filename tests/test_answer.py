"""Tests for the answer generation module."""

import asyncio

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


@pytest.fixture
def search_results():
    chunks = [
        DocumentChunk(
            chunk_id="c1", document_id="d1", content="Python is great for data science.",
            metadata={"source": "guide.pdf", "page": 3}, page_number=3,
        ),
        DocumentChunk(
            chunk_id="c2", document_id="d1", content="FastAPI enables async web development.",
            metadata={"source": "guide.pdf", "page": 7}, page_number=7,
        ),
        DocumentChunk(
            chunk_id="c3", document_id="d2", content="Redis provides in-memory caching.",
            metadata={"source": "infra.txt"},
        ),
    ]
    return [
        SearchResult(chunk=chunks[0], score=0.95, rank=1, source="hybrid"),
        SearchResult(chunk=chunks[1], score=0.82, rank=2, source="hybrid"),
        SearchResult(chunk=chunks[2], score=0.71, rank=3, source="hybrid"),
    ]


class TestBuildContext:
    def test_basic(self, search_results):
        context = build_context(search_results)
        assert "guide.pdf" in context
        assert "p.3" in context
        assert "Python is great" in context

    def test_max_chars(self, search_results):
        context = build_context(search_results, max_chars=50)
        # Should limit to at most one chunk
        assert len(context) < 200

    def test_empty_results(self):
        context = build_context([])
        assert context == ""


class TestExtractCitations:
    def test_basic(self, search_results):
        citations = extract_citations(search_results)
        assert len(citations) == 3
        assert isinstance(citations[0], Citation)
        assert citations[0].page_number == 3
        assert citations[0].source == "guide.pdf"

    def test_scores_preserved(self, search_results):
        citations = extract_citations(search_results)
        assert citations[0].relevance_score == 0.95


class TestBuildQAPrompt:
    def test_contains_question(self):
        prompt = build_qa_prompt("What is Python?", "Python is a language.")
        assert "What is Python?" in prompt
        assert "Python is a language." in prompt
        assert "Answer:" in prompt


class TestGenerateAnswer:
    def test_mock_mode(self, search_results):
        result = asyncio.run(
            generate_answer("What is Python?", search_results)
        )
        assert isinstance(result, Answer)
        assert result.provider == "mock"
        assert len(result.citations) == 3
        assert result.tokens_used > 0
        assert "guide.pdf" in result.answer_text

    def test_with_llm_fn(self, search_results):
        async def fake_llm(prompt, provider):
            return ("Python is a programming language.", "test-model", 42, 0.001)

        result = asyncio.run(
            generate_answer("What is Python?", search_results, llm_fn=fake_llm, provider="test")
        )
        assert result.answer_text == "Python is a programming language."
        assert result.provider == "test"
        assert result.model == "test-model"
        assert result.tokens_used == 42

    def test_empty_results(self):
        result = asyncio.run(
            generate_answer("What is Python?", [])
        )
        assert isinstance(result, Answer)
        assert len(result.citations) == 0
