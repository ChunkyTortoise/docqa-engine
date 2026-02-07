"""Prompt Template Lab: design, test, and compare prompt strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from docqa_engine.answer import Answer, build_context, generate_answer
from docqa_engine.retriever import SearchResult

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PromptTemplate:
    """A reusable prompt template with named ``{placeholder}`` variables."""

    name: str
    template: str
    description: str = ""
    variables: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Auto-extract placeholder names from the template string."""
        if not self.variables:
            self.variables = list(dict.fromkeys(re.findall(r"\{(\w+)\}", self.template)))


@dataclass
class PromptComparison:
    """Side-by-side comparison of two prompt template outputs."""

    template_a_name: str
    template_b_name: str
    question: str
    answer_a: Answer
    answer_b: Answer


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

_BUILTINS: dict[str, tuple[str, str]] = {
    "qa_concise": (
        "Answer the question briefly and cite your sources.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a concise answer with source references:",
        "Brief answer with citations",
    ),
    "qa_detailed": (
        "You are a knowledgeable research assistant. Analyze the provided context "
        "thoroughly and give a detailed answer with analysis and source references.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a thorough, well-structured answer:",
        "Thorough answer with analysis and source references",
    ),
    "summarize": (
        "Summarize the following context clearly and concisely. "
        "Highlight the most important points.\n\n"
        "Context:\n{context}\n\n"
        "Topic/Focus: {question}\n\n"
        "Summary:",
        "Summarize the context",
    ),
    "extract_facts": (
        "Extract the key facts from the context below as bullet points. "
        "Each fact should be specific and verifiable.\n\n"
        "Context:\n{context}\n\n"
        "Topic: {question}\n\n"
        "Key facts:",
        "Extract key facts as bullet points",
    ),
    "compare": (
        "Compare the information provided across different sources in the context. "
        "Note agreements, contradictions, and unique contributions from each source.\n\n"
        "Context:\n{context}\n\n"
        "Comparison focus: {question}\n\n"
        "Comparison:",
        "Compare information across sources",
    ),
}


# ---------------------------------------------------------------------------
# Prompt Library
# ---------------------------------------------------------------------------


class PromptLibrary:
    """Collection of prompt templates with five built-in strategies.

    Built-in templates: ``qa_concise``, ``qa_detailed``, ``summarize``,
    ``extract_facts``, ``compare``.  All use ``{context}`` and ``{question}``
    placeholders.
    """

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        for name, (template, desc) in _BUILTINS.items():
            self._templates[name] = PromptTemplate(name=name, template=template, description=desc)

    def add_template(self, name: str, template: str, description: str = "") -> PromptTemplate:
        """Register a custom prompt template."""
        pt = PromptTemplate(name=name, template=template, description=description)
        self._templates[name] = pt
        return pt

    def get_template(self, name: str) -> PromptTemplate:
        """Retrieve a template by name, raising ``KeyError`` if missing."""
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found. Available: {list(self._templates)}")
        return self._templates[name]

    def render(self, name: str, **kwargs: Any) -> str:
        """Render a template by substituting its placeholder variables."""
        template = self.get_template(name)
        try:
            return template.template.format(**kwargs)
        except KeyError as exc:
            raise KeyError(f"Missing variable {exc} for template '{name}'. Required: {template.variables}") from exc

    def list_templates(self) -> list[PromptTemplate]:
        """Return all registered templates."""
        return list(self._templates.values())


# ---------------------------------------------------------------------------
# Prompt comparison
# ---------------------------------------------------------------------------


def _make_prompt_override(rendered: str, original_fn: Any) -> Any:
    """Wrap an llm_fn so it substitutes a pre-rendered prompt."""
    if original_fn is None:
        return None

    async def wrapper(prompt: str, provider: str) -> tuple:
        return await original_fn(rendered, provider)

    return wrapper


async def compare_prompts(
    question: str,
    results: list[SearchResult],
    template_a_name: str,
    template_b_name: str,
    library: PromptLibrary,
    llm_fn: Any = None,
) -> PromptComparison:
    """Run the same query with two different templates and compare answers.

    Uses :func:`generate_answer` from *answer.py* internally but overrides the
    prompt via template rendering so each answer reflects a different strategy.

    Args:
        question: The user's question.
        results: Search results from the retriever.
        template_a_name: First template name (from *library*).
        template_b_name: Second template name (from *library*).
        library: ``PromptLibrary`` containing both templates.
        llm_fn: Async LLM callable, or ``None`` for mock mode.
    """
    context = build_context(results)

    rendered_a = library.render(template_a_name, context=context, question=question)
    rendered_b = library.render(template_b_name, context=context, question=question)

    answer_a = await generate_answer(
        question,
        results,
        llm_fn=_make_prompt_override(rendered_a, llm_fn),
    )
    answer_b = await generate_answer(
        question,
        results,
        llm_fn=_make_prompt_override(rendered_b, llm_fn),
    )

    return PromptComparison(
        template_a_name=template_a_name,
        template_b_name=template_b_name,
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )
