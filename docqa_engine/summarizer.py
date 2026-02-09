"""Document summarization: extractive TF-IDF sentence scoring and key phrase extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KeyPhrase:
    """A key phrase with its TF-IDF importance score."""

    phrase: str
    score: float


@dataclass
class SummaryResult:
    """Result of a document summarization."""

    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    key_phrases: list[KeyPhrase] = field(default_factory=list)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class DocumentSummarizer:
    """Extractive document summarizer using TF-IDF sentence scoring.

    Scores sentences by their TF-IDF similarity to the overall document,
    then selects top-scoring sentences to form the summary.
    """

    def _score_sentences(self, text: str) -> list[tuple[str, float]]:
        """Score each sentence by TF-IDF cosine similarity to the full document.

        Returns list of (sentence, score) tuples sorted by score descending.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [(sentences[0], 1.0)]

        try:
            vectorizer = TfidfVectorizer()
            # Fit on sentences + full document
            corpus = sentences + [text]
            tfidf = vectorizer.fit_transform(corpus)
            doc_vec = tfidf[-1:]
            sent_vecs = tfidf[:-1]
            sims = cosine_similarity(sent_vecs, doc_vec).flatten()
        except ValueError:
            return [(s, 1.0 / len(sentences)) for s in sentences]

        scored = [(sentences[i], float(sims[i])) for i in range(len(sentences))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def extract_key_phrases(self, text: str, top_k: int = 10) -> list[KeyPhrase]:
        """Extract top TF-IDF terms from text.

        Args:
            text: Input text.
            top_k: Number of top phrases to return.

        Returns:
            List of KeyPhrase sorted by score descending.
        """
        if not text or not text.strip():
            return []
        try:
            vectorizer = TfidfVectorizer(max_features=top_k * 2, stop_words="english")
            tfidf = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf.toarray()[0]
        except ValueError:
            return []

        phrases = []
        for idx in scores.argsort()[::-1][:top_k]:
            if scores[idx] > 0:
                phrases.append(KeyPhrase(phrase=feature_names[idx], score=round(float(scores[idx]), 4)))
        return phrases

    def summarize(self, text: str, ratio: float = 0.3) -> SummaryResult:
        """Produce an extractive summary of a single document.

        Selects top-scoring sentences to approximate the target compression ratio.

        Args:
            text: Input document text.
            ratio: Target ratio of summary to original length (0.0 to 1.0).

        Returns:
            SummaryResult with summary text, lengths, and key phrases.
        """
        if not text or not text.strip():
            return SummaryResult(
                summary="",
                original_length=0,
                summary_length=0,
                compression_ratio=0.0,
            )

        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return SummaryResult(
                summary=text.strip(),
                original_length=len(text),
                summary_length=len(text.strip()),
                compression_ratio=1.0,
                key_phrases=self.extract_key_phrases(text),
            )

        scored = self._score_sentences(text)
        target_count = max(1, int(len(sentences) * ratio))

        # Select top sentences, preserve original order
        top_sentences = {s for s, _ in scored[:target_count]}
        ordered = [s for s in sentences if s in top_sentences]

        summary_text = " ".join(ordered)
        key_phrases = self.extract_key_phrases(text)

        original_length = len(text)
        summary_length = len(summary_text)
        compression = round(summary_length / original_length, 4) if original_length > 0 else 0.0

        return SummaryResult(
            summary=summary_text,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression,
            key_phrases=key_phrases,
        )

    def summarize_multi(self, documents: list[str], ratio: float = 0.3) -> SummaryResult:
        """Summarize multiple documents into a single summary.

        Concatenates documents and applies extractive summarization.

        Args:
            documents: List of document texts.
            ratio: Target compression ratio.

        Returns:
            SummaryResult over all documents combined.
        """
        if not documents:
            return SummaryResult(
                summary="",
                original_length=0,
                summary_length=0,
                compression_ratio=0.0,
            )

        combined = " ".join(doc.strip() for doc in documents if doc.strip())
        return self.summarize(combined, ratio=ratio)
