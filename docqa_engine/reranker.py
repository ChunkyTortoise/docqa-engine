"""Cross-Encoder Re-Ranker: two-stage retrieval with TF-IDF cross-similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from docqa_engine.retriever import SearchResult


@dataclass
class RerankResult:
    original_rank: int
    new_rank: int
    score: float
    search_result: SearchResult


@dataclass
class RerankReport:
    results: list[RerankResult]
    kendall_tau: float
    improvement_ratio: float


class CrossEncoderReranker:
    """Pointwise TF-IDF cross-similarity re-ranker."""

    def _score_pair(self, query: str, document: str) -> float:
        """Compute TF-IDF cosine similarity between query and document."""
        if not query or not query.strip() or not document or not document.strip():
            return 0.0
        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([query, document])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
            return float(sim[0][0])
        except ValueError:
            return 0.0

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Score each result against query using TF-IDF cosine similarity.

        Returns results sorted by new score (descending).
        """
        if not results:
            return []

        scored = []
        for result in results:
            score = self._score_pair(query, result.chunk.content)
            scored.append((result, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        reranked = []
        for new_rank_idx, (result, score) in enumerate(scored):
            reranked.append(
                RerankResult(
                    original_rank=result.rank,
                    new_rank=new_rank_idx + 1,
                    score=score,
                    search_result=result,
                )
            )

        return reranked

    def cascade_rerank(
        self,
        query: str,
        results: list[SearchResult],
        stages: list[Callable],
    ) -> list[RerankResult]:
        """Multi-stage re-ranking pipeline.

        Each stage is a scoring function: (query, document) -> float.
        Results are re-scored and re-sorted at each stage.
        """
        if not results:
            return []

        current = list(results)

        for stage_fn in stages:
            scored = []
            for result in current:
                score = stage_fn(query, result.chunk.content)
                scored.append((result, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            current = [r for r, _ in scored]

        # Build final RerankResults using last-stage scores
        final_scored = []
        for result in current:
            if stages:
                score = stages[-1](query, result.chunk.content)
            else:
                score = result.score
            final_scored.append((result, score))

        reranked = []
        for new_rank_idx, (result, score) in enumerate(final_scored):
            reranked.append(
                RerankResult(
                    original_rank=result.rank,
                    new_rank=new_rank_idx + 1,
                    score=score,
                    search_result=result,
                )
            )

        return reranked

    def _kendall_tau(self, original_ranks: list[int], new_ranks: list[int]) -> float:
        """Compute Kendall tau rank correlation between original and new rankings."""
        n = len(original_ranks)
        if n < 2:
            return 1.0

        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                orig_diff = original_ranks[i] - original_ranks[j]
                new_diff = new_ranks[i] - new_ranks[j]

                if orig_diff * new_diff > 0:
                    concordant += 1
                elif orig_diff * new_diff < 0:
                    discordant += 1

        pairs = n * (n - 1) / 2
        if pairs == 0:
            return 1.0

        return (concordant - discordant) / pairs

    def report(self, query: str, results: list[SearchResult]) -> RerankReport:
        """Generate a full re-ranking report with Kendall tau correlation."""
        reranked = self.rerank(query, results)

        if not reranked:
            return RerankReport(results=[], kendall_tau=1.0, improvement_ratio=0.0)

        original_ranks = [r.original_rank for r in reranked]
        new_ranks = [r.new_rank for r in reranked]

        tau = self._kendall_tau(original_ranks, new_ranks)

        # Improvement ratio: fraction of results that moved to a better (lower) rank
        improved = sum(1 for r in reranked if r.new_rank < r.original_rank)
        improvement_ratio = improved / len(reranked)

        return RerankReport(
            results=reranked,
            kendall_tau=tau,
            improvement_ratio=improvement_ratio,
        )
