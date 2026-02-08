"""Retrieval Evaluation Metrics: MRR, NDCG@K, Precision@K, Recall@K, Hit Rate@K."""

from __future__ import annotations

import math


class Evaluator:
    """Evaluate retrieval quality using standard information retrieval metrics.

    Computes five metrics over a set of queries:

    - **MRR** (Mean Reciprocal Rank): Average of ``1 / rank`` of the first
      relevant document across all queries.
    - **NDCG@K** (Normalized Discounted Cumulative Gain): Measures ranking
      quality by giving higher weight to relevant documents that appear earlier.
    - **Precision@K**: Fraction of the top-K results that are relevant.
    - **Recall@K**: Fraction of all relevant documents found in the top-K.
    - **Hit Rate@K**: Binary indicator -- did at least one relevant document
      appear in the top-K?
    """

    # ------------------------------------------------------------------
    # Single-query evaluation
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        retrieved: list[str],
        relevant: set[str],
        k: int = 5,
    ) -> dict[str, float]:
        """Evaluate retrieval quality for a single query.

        Args:
            retrieved: Ordered list of retrieved document identifiers (best first).
            relevant: Set of ground-truth relevant document identifiers.
            k: Cut-off rank for the metrics.

        Returns:
            Dict with keys ``mrr``, ``ndcg``, ``precision``, ``recall``,
            ``hit_rate``.
        """
        top_k = retrieved[:k]

        mrr = self._reciprocal_rank(top_k, relevant)
        ndcg = self._ndcg_at_k(top_k, relevant, k)
        precision = self._precision_at_k(top_k, relevant)
        recall = self._recall_at_k(top_k, relevant)
        hit_rate = self._hit_rate_at_k(top_k, relevant)

        return {
            "mrr": mrr,
            "ndcg": ndcg,
            "precision": precision,
            "recall": recall,
            "hit_rate": hit_rate,
        }

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        queries: list[str],
        retrieved_docs: list[list[str]],
        relevant_docs: list[set[str]],
        k: int = 5,
    ) -> dict[str, float]:
        """Evaluate retrieval quality averaged over multiple queries.

        Args:
            queries: List of query strings (used for alignment only).
            retrieved_docs: Per-query ordered list of retrieved doc identifiers.
            relevant_docs: Per-query set of ground-truth relevant doc identifiers.
            k: Cut-off rank.

        Returns:
            Dict with mean ``mrr``, ``ndcg``, ``precision``, ``recall``,
            ``hit_rate`` across all queries.

        Raises:
            ValueError: If the input lists have mismatched lengths.
        """
        if not (len(queries) == len(retrieved_docs) == len(relevant_docs)):
            raise ValueError(
                f"Length mismatch: queries={len(queries)}, "
                f"retrieved={len(retrieved_docs)}, "
                f"relevant={len(relevant_docs)}"
            )

        if len(queries) == 0:
            return {
                "mrr": 0.0,
                "ndcg": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "hit_rate": 0.0,
            }

        totals: dict[str, float] = {
            "mrr": 0.0,
            "ndcg": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
        }

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            single = self.evaluate_single(retrieved, relevant, k=k)
            for key in totals:
                totals[key] += single[key]

        n = len(queries)
        return {key: value / n for key, value in totals.items()}

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank(top_k: list[str], relevant: set[str]) -> float:
        """Return ``1 / rank`` of the first relevant document, or 0."""
        for i, doc_id in enumerate(top_k):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _ndcg_at_k(top_k: list[str], relevant: set[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain at rank *k*.

        Binary relevance: 1 if the document is relevant, 0 otherwise.
        """
        dcg = 0.0
        for i, doc_id in enumerate(top_k):
            if doc_id in relevant:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-indexed

        # Ideal DCG: all relevant docs at the top
        ideal_count = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def _precision_at_k(top_k: list[str], relevant: set[str]) -> float:
        """Fraction of the top-K results that are relevant."""
        if not top_k:
            return 0.0
        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        return hits / len(top_k)

    @staticmethod
    def _recall_at_k(top_k: list[str], relevant: set[str]) -> float:
        """Fraction of all relevant docs found in the top-K."""
        if not relevant:
            return 0.0
        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        return hits / len(relevant)

    @staticmethod
    def _hit_rate_at_k(top_k: list[str], relevant: set[str]) -> float:
        """Binary: 1.0 if at least one relevant doc appears in top-K, else 0.0."""
        for doc_id in top_k:
            if doc_id in relevant:
                return 1.0
        return 0.0
