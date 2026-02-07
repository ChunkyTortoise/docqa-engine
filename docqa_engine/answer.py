"""Multi-LLM Answer Generation with source citations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from docqa_engine.retriever import SearchResult


@dataclass
class Citation:
    chunk_id: str
    content_snippet: str
    page_number: int | None = None
    source: str = ""
    relevance_score: float = 0.0


@dataclass
class Answer:
    question: str
    answer_text: str
    citations: list[Citation]
    provider: str
    model: str
    tokens_used: int = 0
    cost_estimate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def build_context(results: list[SearchResult], max_chars: int = 4000) -> str:
    """Build context string from search results with source annotations."""
    context_parts = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        chunk = result.chunk
        source = chunk.metadata.get("source", "unknown")
        page = chunk.page_number
        ref = f"[Source {i}: {source}"
        if page:
            ref += f", p.{page}"
        ref += "]"

        entry = f"{ref}\n{chunk.content}\n"
        if total_chars + len(entry) > max_chars:
            break
        context_parts.append(entry)
        total_chars += len(entry)

    return "\n".join(context_parts)


def extract_citations(results: list[SearchResult]) -> list[Citation]:
    """Extract citations from search results."""
    return [
        Citation(
            chunk_id=r.chunk.chunk_id,
            content_snippet=r.chunk.content[:200],
            page_number=r.chunk.page_number,
            source=r.chunk.metadata.get("source", ""),
            relevance_score=r.score,
        )
        for r in results
    ]


def build_qa_prompt(question: str, context: str) -> str:
    """Build the Q&A prompt with context."""
    return (
        "Answer the following question based on the provided context. "
        "Include specific references to the sources when possible. "
        "If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


async def generate_answer(
    question: str,
    results: list[SearchResult],
    llm_fn=None,
    provider: str = "mock",
    max_context_chars: int = 4000,
) -> Answer:
    """Generate an answer using retrieved context and an LLM.

    Args:
        question: The user's question
        results: Search results from the retriever
        llm_fn: Async callable (prompt, provider) -> (text, model, tokens, cost)
        provider: LLM provider name
        max_context_chars: Max characters for context window
    """
    context = build_context(results, max_chars=max_context_chars)
    citations = extract_citations(results)
    prompt = build_qa_prompt(question, context)

    if llm_fn is None:
        # Mock mode — generate a synthetic answer
        source_refs = [c.source for c in citations[:3]]
        answer_text = (
            f"Based on the provided documents ({', '.join(source_refs) or 'available sources'}), "
            f"here is what I found regarding your question about "
            f"'{question[:50]}{'...' if len(question) > 50 else ''}':\n\n"
            f"The documents indicate that the relevant information can be found "
            f"in the referenced sources. The key points are covered across "
            f"{len(citations)} source passages."
        )
        return Answer(
            question=question,
            answer_text=answer_text,
            citations=citations,
            provider="mock",
            model="mock-v1",
            tokens_used=len(prompt.split()) + len(answer_text.split()),
        )

    text, model, tokens, cost = await llm_fn(prompt, provider)
    return Answer(
        question=question,
        answer_text=text,
        citations=citations,
        provider=provider,
        model=model,
        tokens_used=tokens,
        cost_estimate=cost,
    )
