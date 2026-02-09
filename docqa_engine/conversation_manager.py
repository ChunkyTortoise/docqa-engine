"""Conversation Manager: multi-turn conversation tracking and context-aware query expansion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from docqa_engine.query_expansion import QueryExpander


@dataclass
class ConversationTurn:
    query: str
    answer: str
    retrieved_chunks: list[str]
    timestamp: datetime
    turn_number: int


class ConversationHistory:
    """Store and manage multi-turn conversations."""

    def __init__(self) -> None:
        self._turns: list[ConversationTurn] = []

    def add_turn(self, query: str, answer: str, chunks: list[str] | None = None) -> ConversationTurn:
        """Record a new conversation turn."""
        turn = ConversationTurn(
            query=query,
            answer=answer,
            retrieved_chunks=chunks if chunks is not None else [],
            timestamp=datetime.now(tz=timezone.utc),
            turn_number=len(self._turns) + 1,
        )
        self._turns.append(turn)
        return turn

    def get_turns(self, last_n: int | None = None) -> list[ConversationTurn]:
        """Return conversation turns, optionally limited to the last N."""
        if last_n is None:
            return list(self._turns)
        return list(self._turns[-last_n:])

    def get_context_window(self, max_turns: int = 5) -> str:
        """Format recent history as a context string for injection into prompts."""
        recent = self.get_turns(last_n=max_turns)
        if not recent:
            return ""
        lines: list[str] = []
        for turn in recent:
            lines.append(f"Q: {turn.query}")
            lines.append(f"A: {turn.answer}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all turns."""
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)


_PRONOUN_PATTERN = re.compile(r"\b(it|that|those|this|these|them)\b", re.IGNORECASE)
_FOLLOWUP_PATTERN = re.compile(
    r"\b(also|too|another|more|else|additionally|furthermore|"
    r"what about|how about|and\b.*\?|follow.?up)\b",
    re.IGNORECASE,
)


class ConversationContext:
    """Track active topic and entity references across turns."""

    def __init__(self) -> None:
        self._entities: list[str] = []
        self._topic_words: list[str] = []
        self._recent_turns: list[ConversationTurn] = []

    def update(self, turn: ConversationTurn) -> None:
        """Extract entities from a turn and update topic tracking."""
        self._recent_turns.append(turn)
        new_entities = self.extract_entities(turn.query + " " + turn.answer)
        self._entities.extend(new_entities)
        # Keep topic words from recent turns (last 5)
        self._topic_words.extend(_extract_keywords(turn.query))
        # Limit memory
        if len(self._recent_turns) > 10:
            self._recent_turns = self._recent_turns[-10:]
        if len(self._entities) > 50:
            self._entities = self._entities[-50:]
        if len(self._topic_words) > 100:
            self._topic_words = self._topic_words[-100:]

    @property
    def active_topic(self) -> str:
        """Return the most frequent topic keyword from recent turns."""
        if not self._topic_words:
            return ""
        freq: dict[str, int] = {}
        for w in self._topic_words:
            key = w.lower()
            freq[key] = freq.get(key, 0) + 1
        return max(freq, key=freq.get)  # type: ignore[arg-type]

    def resolve_references(self, query: str) -> str:
        """Replace pronouns with the most recent entity from context."""
        if not self._entities:
            return query
        last_entity = self._entities[-1]

        def _replace(match: re.Match[str]) -> str:
            return last_entity

        return _PRONOUN_PATTERN.sub(_replace, query)

    def detect_followup(self, query: str) -> bool:
        """Heuristic: is this query a follow-up to the previous conversation?"""
        if not self._recent_turns:
            return False
        # Pronouns referencing prior context
        if _PRONOUN_PATTERN.search(query):
            return True
        # Explicit follow-up markers
        if _FOLLOWUP_PATTERN.search(query):
            return True
        # Very short queries often are follow-ups
        word_count = len(query.split())
        if word_count <= 2 and self._recent_turns:
            return True
        return False

    @staticmethod
    def extract_entities(text: str) -> list[str]:
        """Simple NER: extract capitalized words, quoted phrases, and numbers."""
        entities: list[str] = []
        # Quoted phrases
        for match in re.finditer(r'"([^"]+)"', text):
            entities.append(match.group(1))
        for match in re.finditer(r"'([^']+)'", text):
            entities.append(match.group(1))
        # Capitalized multi-word names (e.g. "New York", "Machine Learning")
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
            entities.append(match.group(0))
        # Single capitalized words (skip sentence starts heuristically)
        words = text.split()
        for i, word in enumerate(words):
            cleaned = re.sub(r"[^\w]", "", word)
            if not cleaned:
                continue
            if cleaned[0].isupper() and len(cleaned) > 1 and i > 0:
                if cleaned not in entities:
                    entities.append(cleaned)
        # Numbers (standalone)
        for match in re.finditer(r"\b\d+(?:\.\d+)?\b", text):
            entities.append(match.group(0))
        return entities


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful words from text (skip stop words and short words)."""
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "between",
        "through",
        "after",
        "before",
        "above",
        "below",
        "and",
        "or",
        "but",
        "not",
        "no",
        "if",
        "then",
        "than",
        "that",
        "this",
        "it",
        "its",
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "whom",
    }
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if len(w) > 2 and w not in stop_words]


class ContextAwareExpander:
    """Modify queries using conversation history context."""

    def __init__(self, expander: QueryExpander | None = None) -> None:
        self._expander = expander or QueryExpander()

    def expand(self, query: str, context: ConversationContext) -> str:
        """Prepend relevant context to the query for better retrieval."""
        if not query or not query.strip():
            return query
        # Resolve pronoun references
        resolved = context.resolve_references(query)
        # Add active topic if it's not already in the query
        topic = context.active_topic
        if topic and topic.lower() not in resolved.lower():
            resolved = f"{topic} {resolved}"
        return resolved
