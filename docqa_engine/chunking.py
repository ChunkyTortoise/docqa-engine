"""Pluggable document chunking strategies for RAG pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single text chunk."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ChunkingResult:
    """Result of chunking a document."""

    chunks: list[Chunk]
    strategy: str
    avg_chunk_size: float
    total_chunks: int


@dataclass
class ChunkingComparison:
    """Comparison of chunking strategies."""

    results: dict[str, ChunkingResult]
    best_strategy: str
    best_avg_size: float


class Chunker:
    """Pluggable document chunking with multiple strategies."""

    def fixed_size(self, text: str, chunk_size: int = 500, overlap: int = 50) -> ChunkingResult:
        """Split text into fixed-size character chunks with optional overlap.

        Respects word boundaries (doesn't split mid-word).
        """
        if not text:
            return ChunkingResult(chunks=[], strategy="fixed_size", avg_chunk_size=0.0, total_chunks=0)

        chunks: list[Chunk] = []
        pos = 0
        index = 0

        while pos < len(text):
            end = min(pos + chunk_size, len(text))

            # Respect word boundaries: if we're not at the end of text and
            # we're in the middle of a word, back up to the last space.
            if end < len(text) and end > pos and text[end] not in (" ", "\n", "\t"):
                space_pos = text.rfind(" ", pos, end)
                if space_pos > pos:
                    end = space_pos + 1  # include the space in the current chunk

            chunk_text = text[pos:end]
            if chunk_text.strip():  # skip empty chunks
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=index,
                        start_char=pos,
                        end_char=end,
                        metadata={"strategy": "fixed_size"},
                    )
                )
                index += 1

            # If we reached the end of text, stop
            if end >= len(text):
                break

            # Move forward by chunk_size - overlap, but at least 1 char to avoid infinite loop
            step = max(end - pos - overlap, 1)
            pos = pos + step

        avg_size = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0.0
        return ChunkingResult(
            chunks=chunks,
            strategy="fixed_size",
            avg_chunk_size=avg_size,
            total_chunks=len(chunks),
        )

    def sentence(self, text: str, sentences_per_chunk: int = 5) -> ChunkingResult:
        """Split text by sentences, grouping N sentences per chunk.

        Sentence detection: split on '. ', '! ', '? ', or newline.
        """
        if not text:
            return ChunkingResult(chunks=[], strategy="sentence", avg_chunk_size=0.0, total_chunks=0)

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return ChunkingResult(chunks=[], strategy="sentence", avg_chunk_size=0.0, total_chunks=0)

        chunks: list[Chunk] = []
        pos = 0

        for i in range(0, len(sentences), sentences_per_chunk):
            group = sentences[i : i + sentences_per_chunk]
            chunk_text = " ".join(group)

            # Find the actual start position in the original text
            start_char = text.find(group[0], pos)
            if start_char == -1:
                start_char = pos

            # End position: find the end of the last sentence in this group
            last_sentence = group[-1]
            end_search_start = start_char + len(group[0]) if len(group) > 1 else start_char
            end_char = text.find(last_sentence, end_search_start)
            if end_char == -1:
                end_char = start_char + len(chunk_text)
            else:
                end_char = end_char + len(last_sentence)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_char=start_char,
                    end_char=end_char,
                    metadata={"strategy": "sentence"},
                )
            )
            pos = end_char

        avg_size = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0.0
        return ChunkingResult(
            chunks=chunks,
            strategy="sentence",
            avg_chunk_size=avg_size,
            total_chunks=len(chunks),
        )

    def sliding_window(self, text: str, window_size: int = 500, step: int = 250) -> ChunkingResult:
        """Sliding window chunking with configurable step size.

        Creates overlapping chunks: each chunk starts ``step`` characters after
        the previous.
        """
        if not text:
            return ChunkingResult(
                chunks=[],
                strategy="sliding_window",
                avg_chunk_size=0.0,
                total_chunks=0,
            )

        chunks: list[Chunk] = []
        pos = 0
        index = 0

        while pos < len(text):
            end = min(pos + window_size, len(text))
            chunk_text = text[pos:end]

            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=index,
                        start_char=pos,
                        end_char=end,
                        metadata={"strategy": "sliding_window"},
                    )
                )
                index += 1

            pos += step
            # If we've gone past the end, stop
            if pos >= len(text) and end >= len(text):
                break

        avg_size = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0.0
        return ChunkingResult(
            chunks=chunks,
            strategy="sliding_window",
            avg_chunk_size=avg_size,
            total_chunks=len(chunks),
        )

    def semantic(self, text: str, min_chunk_size: int = 100) -> ChunkingResult:
        """Paragraph-based semantic chunking.

        Splits on double newlines (paragraph boundaries).
        Merges short paragraphs (< min_chunk_size) with the next one.
        """
        if not text:
            return ChunkingResult(chunks=[], strategy="semantic", avg_chunk_size=0.0, total_chunks=0)

        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\n+", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return ChunkingResult(chunks=[], strategy="semantic", avg_chunk_size=0.0, total_chunks=0)

        # Merge short paragraphs
        merged: list[str] = []
        buffer = ""
        for para in paragraphs:
            if buffer:
                buffer = buffer + "\n\n" + para
            else:
                buffer = para

            if len(buffer) >= min_chunk_size:
                merged.append(buffer)
                buffer = ""

        # Don't lose trailing buffer
        if buffer:
            if merged:
                merged[-1] = merged[-1] + "\n\n" + buffer
            else:
                merged.append(buffer)

        # Build chunks with positions
        chunks: list[Chunk] = []
        pos = 0
        for i, chunk_text in enumerate(merged):
            # Find position in original text (search from current pos)
            # Use the first paragraph's text to locate
            first_line = chunk_text.split("\n\n")[0]
            start_char = text.find(first_line, pos)
            if start_char == -1:
                start_char = pos

            # Find end: locate the last paragraph of this merged chunk
            last_line = chunk_text.split("\n\n")[-1]
            end_search = start_char + len(first_line) if first_line != last_line else start_char
            end_pos = text.find(last_line, end_search)
            if end_pos == -1:
                end_char = start_char + len(chunk_text)
            else:
                end_char = end_pos + len(last_line)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={"strategy": "semantic"},
                )
            )
            pos = end_char

        avg_size = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0.0
        return ChunkingResult(
            chunks=chunks,
            strategy="semantic",
            avg_chunk_size=avg_size,
            total_chunks=len(chunks),
        )

    def compare_strategies(self, text: str) -> ChunkingComparison:
        """Run all 4 strategies and compare. Best = closest to 500 avg chunk size."""
        results: dict[str, ChunkingResult] = {
            "fixed_size": self.fixed_size(text),
            "sentence": self.sentence(text),
            "sliding_window": self.sliding_window(text),
            "semantic": self.semantic(text),
        }

        target = 500.0
        best_strategy = ""
        best_distance = float("inf")

        for name, result in results.items():
            distance = abs(result.avg_chunk_size - target)
            if distance < best_distance:
                best_distance = distance
                best_strategy = name

        best_avg = results[best_strategy].avg_chunk_size if best_strategy else 0.0

        return ChunkingComparison(
            results=results,
            best_strategy=best_strategy,
            best_avg_size=best_avg,
        )

    def semantic_tfidf(self, text: str, similarity_threshold: float = 0.3) -> ChunkingResult:
        """TF-IDF-based semantic chunking that detects topic boundaries.

        Splits text into paragraphs, computes TF-IDF vectors for adjacent paragraphs,
        and creates chunk boundaries where cosine similarity drops below threshold.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if not text:
            return ChunkingResult(chunks=[], strategy="semantic_tfidf", avg_chunk_size=0.0, total_chunks=0)

        # Split into paragraphs
        paragraphs = re.split(r"\n\n+", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return ChunkingResult(chunks=[], strategy="semantic_tfidf", avg_chunk_size=0.0, total_chunks=0)

        if len(paragraphs) == 1:
            # Single paragraph -> single chunk
            chunks = [
                Chunk(
                    text=paragraphs[0],
                    index=0,
                    start_char=0,
                    end_char=len(paragraphs[0]),
                    metadata={"strategy": "semantic_tfidf"},
                )
            ]
            return ChunkingResult(
                chunks=chunks,
                strategy="semantic_tfidf",
                avg_chunk_size=len(paragraphs[0]),
                total_chunks=1,
            )

        # Compute TF-IDF vectors
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(paragraphs)
        except ValueError:
            # No features (all stop words) -> treat as single chunk
            all_text = "\n\n".join(paragraphs)
            chunks = [
                Chunk(
                    text=all_text,
                    index=0,
                    start_char=0,
                    end_char=len(all_text),
                    metadata={"strategy": "semantic_tfidf"},
                )
            ]
            return ChunkingResult(
                chunks=chunks,
                strategy="semantic_tfidf",
                avg_chunk_size=len(all_text),
                total_chunks=1,
            )

        # Find topic boundaries: where similarity drops below threshold
        boundaries = [0]  # Start of first chunk
        for i in range(len(paragraphs) - 1):
            sim = cosine_similarity(tfidf_matrix[i : i + 1], tfidf_matrix[i + 1 : i + 2])[0][0]
            if sim < similarity_threshold:
                boundaries.append(i + 1)
        boundaries.append(len(paragraphs))  # End boundary

        # Build chunks from boundaries
        chunks: list[Chunk] = []
        pos = 0
        for idx, (start_idx, end_idx) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            chunk_paragraphs = paragraphs[start_idx:end_idx]
            chunk_text = "\n\n".join(chunk_paragraphs)

            # Find position in original text
            start_char = text.find(chunk_paragraphs[0], pos)
            if start_char == -1:
                start_char = pos

            end_char = start_char + len(chunk_text)
            pos = end_char

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={"strategy": "semantic_tfidf"},
                )
            )

        avg_size = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0.0
        return ChunkingResult(
            chunks=chunks,
            strategy="semantic_tfidf",
            avg_chunk_size=avg_size,
            total_chunks=len(chunks),
        )

    def sliding_window_ratio(self, text: str, chunk_size: int = 500, overlap_ratio: float = 0.5) -> ChunkingResult:
        """Sliding window chunking with overlap specified as a ratio.

        Args:
            text: Text to chunk.
            chunk_size: Size of each chunk in characters.
            overlap_ratio: Fraction of chunk_size to overlap (0.0-1.0).

        Returns:
            ChunkingResult with overlapping chunks.
        """
        if not text:
            return ChunkingResult(
                chunks=[],
                strategy="sliding_window_ratio",
                avg_chunk_size=0.0,
                total_chunks=0,
            )

        overlap = int(chunk_size * overlap_ratio)
        step = chunk_size - overlap

        # Delegate to existing sliding_window method
        result = self.sliding_window(text, window_size=chunk_size, step=step)

        # Update strategy name and metadata
        result.strategy = "sliding_window_ratio"
        for chunk in result.chunks:
            chunk.metadata["strategy"] = "sliding_window_ratio"

        return result

    def recursive(self, text: str, max_chunk_size: int = 1000, separators: list[str] | None = None) -> ChunkingResult:
        """Recursive chunking with hierarchical separators.

        Tries splitting on "\n\n" first, then "\n", then ". ", then " ".
        If a chunk exceeds max_chunk_size, recursively splits with next separator.
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " "]

        if not text:
            return ChunkingResult(chunks=[], strategy="recursive", avg_chunk_size=0.0, total_chunks=0)

        def _recursive_split(text_segment: str, sep_idx: int) -> list[str]:
            """Recursively split text using hierarchical separators."""
            if sep_idx >= len(separators):
                # No more separators, must use this chunk even if too large
                return [text_segment] if text_segment else []

            separator = separators[sep_idx]
            parts = text_segment.split(separator)

            result: list[str] = []
            for part in parts:
                if len(part) <= max_chunk_size:
                    if part.strip():
                        result.append(part)
                else:
                    # Part too large, recurse with next separator
                    sub_parts = _recursive_split(part, sep_idx + 1)
                    result.extend(sub_parts)

            return result

        # Perform recursive splitting
        chunk_texts = _recursive_split(text, 0)

        # Build Chunk objects with positions
        chunks: list[Chunk] = []
        pos = 0
        for idx, chunk_text in enumerate(chunk_texts):
            # Find position in original text
            start_char = text.find(chunk_text, pos)
            if start_char == -1:
                # Fallback if exact match not found (shouldn't happen)
                start_char = pos

            end_char = start_char + len(chunk_text)
            pos = end_char

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={"strategy": "recursive"},
                )
            )

        avg_size = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0.0
        return ChunkingResult(
            chunks=chunks,
            strategy="recursive",
            avg_chunk_size=avg_size,
            total_chunks=len(chunks),
        )
