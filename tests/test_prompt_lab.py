"""Tests for the Prompt Template Lab (library, rendering, A/B comparison)."""

from __future__ import annotations

import pytest

from docqa_engine.ingest import DocumentChunk
from docqa_engine.prompt_lab import (
    PromptComparison,
    PromptLibrary,
    PromptTemplate,
    compare_prompts,
)
from docqa_engine.retriever import SearchResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def library() -> PromptLibrary:
    """A fresh PromptLibrary (includes 5 built-in templates)."""
    return PromptLibrary()


@pytest.fixture()
def search_results() -> list[SearchResult]:
    """Minimal search results for compare_prompts tests."""
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            content="Home prices rose 12% year-over-year.",
            metadata={"source": "report.pdf"},
            page_number=1,
        ),
    ]
    return [
        SearchResult(chunk=chunks[0], score=0.9, rank=1, source="hybrid"),
    ]


# ---------------------------------------------------------------------------
# PromptTemplate dataclass
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    """Tests for PromptTemplate auto-variable extraction and fields."""

    def test_auto_extracts_variables(self) -> None:
        pt = PromptTemplate(name="test", template="Hello {name}, welcome to {place}!")
        assert "name" in pt.variables
        assert "place" in pt.variables

    def test_no_duplicates_in_variables(self) -> None:
        pt = PromptTemplate(name="dup", template="{x} and {x} and {y}")
        assert pt.variables == ["x", "y"]

    def test_explicit_variables_not_overridden(self) -> None:
        pt = PromptTemplate(name="explicit", template="{a} {b}", variables=["a", "b", "c"])
        assert pt.variables == ["a", "b", "c"]

    def test_description_default(self) -> None:
        pt = PromptTemplate(name="t", template="T")
        assert pt.description == ""


# ---------------------------------------------------------------------------
# PromptLibrary
# ---------------------------------------------------------------------------


class TestPromptLibrary:
    """Tests for the template library with built-in templates."""

    def test_has_five_builtin_templates(self, library: PromptLibrary) -> None:
        templates = library.list_templates()
        assert len(templates) == 5

    def test_builtin_names(self, library: PromptLibrary) -> None:
        names = {t.name for t in library.list_templates()}
        expected = {"qa_concise", "qa_detailed", "summarize", "extract_facts", "compare"}
        assert names == expected

    def test_list_templates_returns_prompt_templates(self, library: PromptLibrary) -> None:
        templates = library.list_templates()
        assert all(isinstance(t, PromptTemplate) for t in templates)

    def test_get_template_returns_correct(self, library: PromptLibrary) -> None:
        t = library.get_template("qa_concise")
        assert isinstance(t, PromptTemplate)
        assert t.name == "qa_concise"

    def test_get_template_unknown_raises(self, library: PromptLibrary) -> None:
        with pytest.raises(KeyError, match="not found"):
            library.get_template("nonexistent_template")

    def test_render_substitutes_variables(self, library: PromptLibrary) -> None:
        rendered = library.render("qa_concise", context="Some context here.", question="What?")
        assert "Some context here." in rendered
        assert "What?" in rendered

    def test_render_missing_variable_raises(self, library: PromptLibrary) -> None:
        with pytest.raises(KeyError):
            library.render("qa_concise")  # missing context and question

    def test_add_custom_template(self, library: PromptLibrary) -> None:
        pt = library.add_template(
            "custom_qa",
            "Custom: {context}\nQ: {question}\nA:",
            description="My custom template",
        )
        assert isinstance(pt, PromptTemplate)
        assert library.get_template("custom_qa").name == "custom_qa"
        assert len(library.list_templates()) == 6  # 5 builtins + 1 custom

    def test_add_template_overrides_existing(self, library: PromptLibrary) -> None:
        library.add_template("qa_concise", "Override: {question}", description="overridden")
        t = library.get_template("qa_concise")
        assert "Override" in t.template


# ---------------------------------------------------------------------------
# compare_prompts (async)
# ---------------------------------------------------------------------------


class TestComparePrompts:
    """Tests for the async prompt A/B comparison function."""

    @pytest.mark.asyncio
    async def test_returns_prompt_comparison(
        self,
        library: PromptLibrary,
        search_results: list[SearchResult],
    ) -> None:
        result = await compare_prompts(
            question="What is the trend?",
            results=search_results,
            template_a_name="qa_concise",
            template_b_name="qa_detailed",
            library=library,
        )
        assert isinstance(result, PromptComparison)
        assert result.template_a_name == "qa_concise"
        assert result.template_b_name == "qa_detailed"
        assert result.question == "What is the trend?"

    @pytest.mark.asyncio
    async def test_comparison_produces_two_answers(
        self,
        library: PromptLibrary,
        search_results: list[SearchResult],
    ) -> None:
        result = await compare_prompts(
            question="Price trend?",
            results=search_results,
            template_a_name="summarize",
            template_b_name="extract_facts",
            library=library,
        )
        assert result.answer_a is not None
        assert result.answer_b is not None
        assert len(result.answer_a.answer_text) > 0
        assert len(result.answer_b.answer_text) > 0
