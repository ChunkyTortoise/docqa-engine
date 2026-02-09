"""Query Expansion: synonym, pseudo-relevance feedback, and decomposition strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer

SYNONYM_DICT: dict[str, list[str]] = {
    "machine learning": ["ML", "artificial intelligence", "deep learning"],
    "database": ["DB", "data store", "repository"],
    "search": ["query", "lookup", "find", "retrieve"],
    "document": ["file", "text", "record", "page"],
    "error": ["bug", "issue", "fault", "defect"],
    "fast": ["quick", "rapid", "speedy"],
    "big": ["large", "huge", "massive"],
    "small": ["tiny", "little", "compact"],
    "create": ["make", "build", "generate"],
    "delete": ["remove", "erase", "drop"],
    "update": ["modify", "change", "edit"],
    "user": ["person", "client", "customer"],
    "api": ["interface", "endpoint", "service"],
    "test": ["check", "verify", "validate"],
    "performance": ["speed", "efficiency", "throughput"],
    "data": ["information", "records", "content"],
    "analyze": ["examine", "evaluate", "assess"],
    "deploy": ["release", "launch", "publish"],
    "config": ["configuration", "settings", "options"],
    "log": ["record", "trace", "journal"],
}


@dataclass
class ExpandedQuery:
    original_query: str
    expanded_terms: list[tuple[str, float]] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    strategy: str = ""


class QueryExpander:
    """Query expansion with synonym, PRF, and decomposition strategies."""

    def __init__(self, synonyms: dict[str, list[str]] | None = None):
        self.synonyms = synonyms if synonyms is not None else SYNONYM_DICT

    def synonym_expand(self, query: str) -> ExpandedQuery:
        """Expand query using a built-in synonym dictionary.

        Matched synonyms are added with a lower boost factor (0.5).
        """
        if not query or not query.strip():
            return ExpandedQuery(original_query=query, strategy="synonym")

        query_lower = query.lower()
        expanded_terms: list[tuple[str, float]] = []

        for term, syns in self.synonyms.items():
            if term in query_lower:
                for syn in syns:
                    expanded_terms.append((syn, 0.5))

        return ExpandedQuery(
            original_query=query,
            expanded_terms=expanded_terms,
            strategy="synonym",
        )

    def prf_expand(
        self,
        query: str,
        feedback_docs: list[str],
        top_terms: int = 5,
    ) -> ExpandedQuery:
        """Pseudo-relevance feedback expansion using Rocchio on TF-IDF.

        Extracts top TF-IDF terms from feedback docs and adds as expansion terms.
        """
        if not query or not query.strip() or not feedback_docs:
            return ExpandedQuery(original_query=query, strategy="prf")

        non_empty = [d for d in feedback_docs if d and d.strip()]
        if not non_empty:
            return ExpandedQuery(original_query=query, strategy="prf")

        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(non_empty)
        except ValueError:
            return ExpandedQuery(original_query=query, strategy="prf")

        feature_names = vectorizer.get_feature_names_out()

        # Average TF-IDF scores across feedback docs
        avg_scores = tfidf_matrix.mean(axis=0).A1

        # Get top terms by score
        top_indices = avg_scores.argsort()[::-1][:top_terms]

        query_words = set(re.findall(r"\w+", query.lower()))
        expanded_terms: list[tuple[str, float]] = []

        for idx in top_indices:
            term = feature_names[idx]
            score = float(avg_scores[idx])
            if term not in query_words and score > 0:
                expanded_terms.append((term, round(score, 4)))

        return ExpandedQuery(
            original_query=query,
            expanded_terms=expanded_terms,
            strategy="prf",
        )

    def decompose(self, query: str) -> ExpandedQuery:
        """Split compound queries into sub-queries.

        Splits on 'and', 'or', and question patterns (what/how/why/when/where/which/who).
        """
        if not query or not query.strip():
            return ExpandedQuery(original_query=query, strategy="decompose")

        # Split on " and " or " or "
        parts = re.split(r"\s+(?:and|or)\s+", query, flags=re.IGNORECASE)

        sub_queries = [p.strip() for p in parts if p.strip()]

        # If no splitting occurred, try question pattern decomposition
        if len(sub_queries) <= 1:
            question_parts = re.split(
                r"(?:^|\s)(what|how|why|when|where|which|who)\s",
                query,
                flags=re.IGNORECASE,
            )
            if len(question_parts) > 2:
                # Reconstruct question fragments
                sub_queries = []
                i = 1
                while i < len(question_parts) - 1:
                    q = (question_parts[i] + " " + question_parts[i + 1]).strip()
                    if q:
                        sub_queries.append(q)
                    i += 2
                if not sub_queries:
                    sub_queries = [query]
            else:
                sub_queries = [query]

        return ExpandedQuery(
            original_query=query,
            sub_queries=sub_queries,
            strategy="decompose",
        )

    def expand_all(
        self,
        query: str,
        feedback_docs: list[str] | None = None,
    ) -> ExpandedQuery:
        """Combine all expansion strategies."""
        synonym_result = self.synonym_expand(query)
        decompose_result = self.decompose(query)

        all_terms = list(synonym_result.expanded_terms)

        if feedback_docs:
            prf_result = self.prf_expand(query, feedback_docs)
            all_terms.extend(prf_result.expanded_terms)

        return ExpandedQuery(
            original_query=query,
            expanded_terms=all_terms,
            sub_queries=decompose_result.sub_queries,
            strategy="all",
        )

    def merge_expansions(self, expansions: list[ExpandedQuery]) -> ExpandedQuery:
        """Merge multiple expanded queries, deduplicating terms and keeping highest boost."""
        if not expansions:
            return ExpandedQuery(original_query="", strategy="merged")

        original = expansions[0].original_query
        term_map: dict[str, float] = {}
        all_sub_queries: list[str] = []
        seen_sub: set[str] = set()

        for exp in expansions:
            for term, boost in exp.expanded_terms:
                key = term.lower()
                if key not in term_map or boost > term_map[key]:
                    term_map[key] = boost
            for sq in exp.sub_queries:
                sq_lower = sq.lower()
                if sq_lower not in seen_sub:
                    seen_sub.add(sq_lower)
                    all_sub_queries.append(sq)

        merged_terms = [(term, boost) for term, boost in term_map.items()]

        return ExpandedQuery(
            original_query=original,
            expanded_terms=merged_terms,
            sub_queries=all_sub_queries,
            strategy="merged",
        )
