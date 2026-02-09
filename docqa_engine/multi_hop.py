"""Multi-hop retrieval: iterative query decomposition and result merging."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedPassage:
    """A passage retrieved during a hop."""

    text: str
    score: float
    hop_number: int
    source_query: str


@dataclass
class MultiHopResult:
    """Result of multi-hop retrieval."""

    passages: list[RetrievedPassage] = field(default_factory=list)
    hops_used: int = 0
    sub_queries: list[str] = field(default_factory=list)
    merged_scores: list[float] = field(default_factory=list)


class MultiHopRetriever:
    """Iterative multi-hop retrieval with query decomposition.

    Splits compound queries into sub-queries, retrieves relevant passages
    for each, and merges/deduplicates results across hops.
    """

    def decompose_query(self, query: str) -> list[str]:
        """Split a compound query into sub-queries.

        Handles:
        - Explicit conjunctions: "and", "also", "as well as"
        - Question separators: "?", semicolons
        - Comparative structures: "compared to", "versus", "vs"

        Returns at least one sub-query (the original if no split is found).
        """
        if not query or not query.strip():
            return []

        # Split on question marks (multiple questions)
        parts = re.split(r"\?\s*", query.strip())
        parts = [p.strip().rstrip("?").strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return parts

        # Split on semicolons
        parts = [p.strip() for p in query.split(";") if p.strip()]
        if len(parts) > 1:
            return parts

        # Split on conjunctions
        for conj in [" and also ", " and ", " as well as ", " compared to ", " versus ", " vs "]:
            if conj in query.lower():
                idx = query.lower().index(conj)
                part1 = query[:idx].strip()
                part2 = query[idx + len(conj) :].strip()
                if part1 and part2:
                    return [part1, part2]

        return [query.strip()]

    def _single_hop(self, query: str, documents: list[str], top_k: int = 5) -> list[RetrievedPassage]:
        """One retrieval step using TF-IDF cosine similarity.

        Args:
            query: Search query.
            documents: List of document texts.
            top_k: Maximum results to return.

        Returns:
            List of RetrievedPassage sorted by score descending.
        """
        if not documents or not query.strip():
            return []

        non_empty = [(i, d) for i, d in enumerate(documents) if d.strip()]
        if not non_empty:
            return []

        corpus = [d for _, d in non_empty] + [query]
        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(corpus)
        except ValueError:
            return []

        query_vec = tfidf[-1:]
        doc_vecs = tfidf[:-1]
        sims = cosine_similarity(query_vec, doc_vecs)[0]

        scored = []
        for idx, (orig_idx, doc_text) in enumerate(non_empty):
            scored.append((doc_text, float(sims[idx])))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for text, score in scored[:top_k]:
            if score > 0:
                results.append(RetrievedPassage(text=text, score=score, hop_number=0, source_query=query))
        return results

    def merge_results(self, results: list[RetrievedPassage]) -> list[RetrievedPassage]:
        """Deduplicate and rank merged results from multiple hops.

        If the same passage appears in multiple hops, keep the one with
        the highest score. Final list is sorted by score descending.
        """
        seen: dict[str, RetrievedPassage] = {}
        for passage in results:
            key = passage.text.strip()
            if key not in seen or passage.score > seen[key].score:
                seen[key] = passage
        merged = sorted(seen.values(), key=lambda p: p.score, reverse=True)
        return merged

    def retrieve(
        self,
        query: str,
        documents: list[str],
        max_hops: int = 3,
        top_k_per_hop: int = 5,
    ) -> MultiHopResult:
        """Iterative multi-hop retrieval.

        Decomposes the query into sub-queries and retrieves passages for each.
        Each sub-query is one "hop". Results are merged and deduplicated.

        Args:
            query: The compound or simple query.
            documents: Corpus of document texts.
            max_hops: Maximum number of retrieval hops.
            top_k_per_hop: Maximum passages per hop.

        Returns:
            MultiHopResult with all passages, hops used, and sub-queries.
        """
        if not query or not query.strip() or not documents:
            return MultiHopResult()

        sub_queries = self.decompose_query(query)
        sub_queries = sub_queries[:max_hops]

        all_passages: list[RetrievedPassage] = []
        for hop_num, sub_q in enumerate(sub_queries, start=1):
            hop_results = self._single_hop(sub_q, documents, top_k=top_k_per_hop)
            for p in hop_results:
                p.hop_number = hop_num
            all_passages.extend(hop_results)

        merged = self.merge_results(all_passages)
        merged_scores = [p.score for p in merged]

        return MultiHopResult(
            passages=merged,
            hops_used=len(sub_queries),
            sub_queries=sub_queries,
            merged_scores=merged_scores,
        )
