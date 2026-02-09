"""Answer Quality Metrics: coherence, conciseness, completeness, groundedness."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
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
        "shall",
        "can",
        "need",
        "dare",
        "ought",
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
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "but",
        "and",
        "or",
        "if",
        "while",
        "about",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "it",
        "its",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
    }
)


@dataclass
class QualityReport:
    coherence: float
    conciseness: float
    completeness: float
    groundedness: float
    overall: float


@dataclass
class AnswerComparison:
    report_a: QualityReport
    report_b: QualityReport
    winner: str
    deltas: dict[str, float]


class AnswerQualityScorer:
    """Score answer quality across four dimensions."""

    def coherence(self, answer: str) -> float:
        """Sentence transition quality: average TF-IDF cosine similarity between consecutive sentences."""
        if not answer or not answer.strip():
            return 0.0

        sentences = re.split(r"[.!?]+", answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 1.0

        similarities = []
        for i in range(len(sentences) - 1):
            s1 = sentences[i]
            s2 = sentences[i + 1]
            try:
                vectorizer = TfidfVectorizer()
                tfidf = vectorizer.fit_transform([s1, s2])
                sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
                similarities.append(float(sim[0][0]))
            except ValueError:
                similarities.append(0.0)

        if not similarities:
            return 0.0

        return sum(similarities) / len(similarities)

    def conciseness(self, answer: str, max_words: int = 200) -> float:
        """Score based on word count: 1.0 if under max_words, linearly to 0.0 at 3x max_words."""
        if not answer or not answer.strip():
            return 1.0

        word_count = len(answer.split())

        if word_count <= max_words:
            return 1.0

        if word_count >= 3 * max_words:
            return 0.0

        return 1.0 - (word_count - max_words) / (2 * max_words)

    def completeness(self, answer: str, query: str) -> float:
        """Keyword coverage: fraction of query keywords found in answer (excluding stopwords)."""
        if not query or not query.strip():
            return 0.0

        query_words = set(re.findall(r"[a-zA-Z]+", query.lower()))
        query_words = {w for w in query_words if w not in STOPWORDS and len(w) > 1}

        if not query_words:
            return 0.0

        if not answer or not answer.strip():
            return 0.0

        answer_words = set(re.findall(r"[a-zA-Z]+", answer.lower()))

        found = query_words & answer_words
        return len(found) / len(query_words)

    def groundedness(self, answer: str, context: str) -> float:
        """Factual grounding: fraction of answer sentences supported by context (>50% keyword overlap)."""
        if not answer or not answer.strip() or not context or not context.strip():
            return 0.0

        sentences = re.split(r"[.!?]+", answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        context_words = set(re.findall(r"[a-zA-Z]+", context.lower()))
        context_words = {w for w in context_words if len(w) > 2}

        supported = 0
        for sentence in sentences:
            sent_words = set(re.findall(r"[a-zA-Z]+", sentence.lower()))
            sent_words = {w for w in sent_words if len(w) > 2}

            if not sent_words:
                continue

            overlap = sent_words & context_words
            if len(overlap) / len(sent_words) > 0.5:
                supported += 1

        return supported / len(sentences)

    def score(
        self,
        answer: str,
        query: str,
        context: str,
        max_words: int = 200,
    ) -> QualityReport:
        """Compute all four dimensions and weighted overall score."""
        coh = self.coherence(answer)
        con = self.conciseness(answer, max_words)
        com = self.completeness(answer, query)
        gro = self.groundedness(answer, context)

        overall = 0.25 * coh + 0.15 * con + 0.30 * com + 0.30 * gro

        return QualityReport(
            coherence=coh,
            conciseness=con,
            completeness=com,
            groundedness=gro,
            overall=overall,
        )

    def compare_answers(
        self,
        answer_a: str,
        answer_b: str,
        query: str,
        context: str,
    ) -> AnswerComparison:
        """Compare two answers and determine a winner."""
        report_a = self.score(answer_a, query, context)
        report_b = self.score(answer_b, query, context)

        deltas = {
            "coherence": report_a.coherence - report_b.coherence,
            "conciseness": report_a.conciseness - report_b.conciseness,
            "completeness": report_a.completeness - report_b.completeness,
            "groundedness": report_a.groundedness - report_b.groundedness,
            "overall": report_a.overall - report_b.overall,
        }

        if report_a.overall > report_b.overall:
            winner = "a"
        elif report_b.overall > report_a.overall:
            winner = "b"
        else:
            winner = "tie"

        return AnswerComparison(
            report_a=report_a,
            report_b=report_b,
            winner=winner,
            deltas=deltas,
        )

    def score_batch(
        self,
        items: list[tuple[str, str, str]],
    ) -> list[QualityReport]:
        """Score multiple (answer, query, context) tuples."""
        return [self.score(answer, query, context) for answer, query, context in items]
