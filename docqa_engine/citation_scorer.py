"""Citation accuracy scoring: faithfulness, coverage, redundancy."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CitationScore:
    """Score for a single citation."""

    citation_text: str
    source_text: str
    faithfulness: float  # 0-1, keyword overlap ratio
    relevance: float  # 0-1, how relevant to the query


@dataclass
class CitationReport:
    """Overall citation quality report."""

    scores: list[CitationScore]
    avg_faithfulness: float
    avg_relevance: float
    coverage: float  # fraction of source content referenced
    redundancy: float  # fraction of duplicate citations
    overall_score: float  # weighted combination


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase keywords from text, filtering short words."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) > 2}


class CitationScorer:
    """Score citation accuracy and quality."""

    def score_citation(self, citation: str, source: str, query: str = "") -> CitationScore:
        """Score a single citation against its source.

        Faithfulness: fraction of citation keywords found in source.
        Relevance: fraction of query keywords found in citation.
        """
        citation_keywords = _extract_keywords(citation)
        source_keywords = _extract_keywords(source)

        # Faithfulness: how many citation keywords are in the source
        if not citation_keywords:
            faithfulness = 0.0
        else:
            overlap = citation_keywords & source_keywords
            faithfulness = len(overlap) / len(citation_keywords)

        # Relevance: how many query keywords are in the citation
        if not query or not query.strip():
            relevance = 0.0
        else:
            query_keywords = _extract_keywords(query)
            if not query_keywords:
                relevance = 0.0
            else:
                query_overlap = query_keywords & citation_keywords
                relevance = len(query_overlap) / len(query_keywords)

        return CitationScore(
            citation_text=citation,
            source_text=source,
            faithfulness=faithfulness,
            relevance=relevance,
        )

    def coverage_analysis(self, citations: list[str], source: str) -> float:
        """What fraction of source content is covered by citations?

        Uses keyword overlap: unique source words covered by any citation
        divided by total source words.
        """
        source_keywords = _extract_keywords(source)
        if not source_keywords:
            return 0.0

        covered: set[str] = set()
        for citation in citations:
            citation_keywords = _extract_keywords(citation)
            covered |= citation_keywords & source_keywords

        return len(covered) / len(source_keywords)

    def redundancy_detection(self, citations: list[str]) -> float:
        """Detect redundant citations. Returns 0-1 (1 = all identical).

        Compares each pair of citations using keyword overlap.
        Redundancy = fraction of pairs with >80% overlap.
        """
        if len(citations) < 2:
            return 0.0

        keyword_sets = [_extract_keywords(c) for c in citations]
        total_pairs = 0
        redundant_pairs = 0

        for i in range(len(keyword_sets)):
            for j in range(i + 1, len(keyword_sets)):
                total_pairs += 1
                set_a = keyword_sets[i]
                set_b = keyword_sets[j]

                if not set_a and not set_b:
                    # Both empty = identical
                    redundant_pairs += 1
                    continue

                if not set_a or not set_b:
                    continue

                union = set_a | set_b
                intersection = set_a & set_b
                overlap_ratio = len(intersection) / len(union) if union else 0.0

                if overlap_ratio > 0.8:
                    redundant_pairs += 1

        return redundant_pairs / total_pairs if total_pairs > 0 else 0.0

    def score_all(self, citations: list[str], source: str, query: str = "") -> CitationReport:
        """Score all citations and produce a comprehensive report."""
        scores = [self.score_citation(citation, source, query) for citation in citations]

        avg_faithfulness = sum(s.faithfulness for s in scores) / len(scores) if scores else 0.0
        avg_relevance = sum(s.relevance for s in scores) / len(scores) if scores else 0.0

        coverage = self.coverage_analysis(citations, source)
        redundancy = self.redundancy_detection(citations)

        # Overall score: weighted combination
        # Higher faithfulness and coverage are good; higher redundancy is bad.
        overall_score = 0.4 * avg_faithfulness + 0.3 * coverage + 0.2 * avg_relevance + 0.1 * (1.0 - redundancy)
        overall_score = max(0.0, min(1.0, overall_score))

        return CitationReport(
            scores=scores,
            avg_faithfulness=avg_faithfulness,
            avg_relevance=avg_relevance,
            coverage=coverage,
            redundancy=redundancy,
            overall_score=overall_score,
        )

    def relevance_score_tfidf(self, query: str, citation_text: str) -> float:
        """Compute TF-IDF cosine similarity between query and citation.

        Args:
            query: The search query.
            citation_text: The citation text to score.

        Returns:
            Float 0-1 representing relevance score.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if not query or not query.strip() or not citation_text or not citation_text.strip():
            return 0.0

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([query, citation_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except ValueError:
            # No features (all stop words or empty)
            return 0.0

    def deduplicate_citations(self, citations: list[str], similarity_threshold: float = 0.8):
        """Merge citations with high overlap.

        Args:
            citations: List of citation texts.
            similarity_threshold: Threshold for considering citations duplicates (0-1).

        Returns:
            dict with 'citations', 'removed_count', 'original_count'.
        """
        if not citations:
            return {"citations": [], "removed_count": 0, "original_count": 0}

        original_count = len(citations)
        keyword_sets = [_extract_keywords(c) for c in citations]

        # Track which citations to keep
        keep = [True] * len(citations)

        for i in range(len(citations)):
            if not keep[i]:
                continue

            for j in range(i + 1, len(citations)):
                if not keep[j]:
                    continue

                set_a = keyword_sets[i]
                set_b = keyword_sets[j]

                # Calculate Jaccard similarity
                if not set_a and not set_b:
                    # Both empty = identical
                    keep[j] = False
                    continue

                if not set_a or not set_b:
                    continue

                union = set_a | set_b
                intersection = set_a & set_b
                jaccard = len(intersection) / len(union) if union else 0.0

                if jaccard > similarity_threshold:
                    # Mark duplicate for removal (keep the first one)
                    keep[j] = False

        deduplicated = [citations[i] for i in range(len(citations)) if keep[i]]
        removed_count = original_count - len(deduplicated)

        return {
            "citations": deduplicated,
            "removed_count": removed_count,
            "original_count": original_count,
        }

    def rank_citations(self, citations: list[str], query: str) -> list[str]:
        """Rank citations by relevance to query using TF-IDF.

        Args:
            citations: List of citation texts.
            query: The search query.

        Returns:
            Citations sorted by relevance (most relevant first).
        """
        if not citations:
            return []

        # Score each citation
        scores = [(citation, self.relevance_score_tfidf(query, citation)) for citation in citations]

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return [citation for citation, _ in scores]

    def source_coverage(self, citations: list[str], source_texts: list[str]):
        """Check which sources are covered by at least one citation.

        Args:
            citations: List of citation texts.
            source_texts: List of source document texts.

        Returns:
            dict with 'coverage_ratio', 'covered_sources', 'uncovered_sources'.
        """
        if not source_texts:
            return {"coverage_ratio": 0.0, "covered_sources": [], "uncovered_sources": []}

        citation_keywords = set()
        for citation in citations:
            citation_keywords.update(_extract_keywords(citation))

        covered = []
        uncovered = []

        for source in source_texts:
            source_keywords = _extract_keywords(source)
            # Check if there's any overlap
            if citation_keywords & source_keywords:
                covered.append(source)
            else:
                uncovered.append(source)

        coverage_ratio = len(covered) / len(source_texts) if source_texts else 0.0

        return {
            "coverage_ratio": coverage_ratio,
            "covered_sources": covered,
            "uncovered_sources": uncovered,
        }
