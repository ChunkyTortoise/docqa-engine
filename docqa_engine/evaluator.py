"""Retrieval Evaluation Metrics: MRR, NDCG@K, Precision@K, Recall@K, Hit Rate@K.

Also includes RAGAS-style generation quality metrics: context relevancy,
answer relevancy, and faithfulness.
"""

from __future__ import annotations

import math
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

    # ------------------------------------------------------------------
    # RAGAS-style generation quality metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _tfidf_cosine(text_a: str, text_b: str) -> float:
        """Compute TF-IDF cosine similarity between two texts.

        Returns 0.0 for empty or non-overlapping texts.
        """
        if not text_a or not text_a.strip() or not text_b or not text_b.strip():
            return 0.0

        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([text_a, text_b])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
            return float(sim[0][0])
        except ValueError:
            # Can happen if both texts have only stop words or no features
            return 0.0

    def context_relevancy(self, context: str, query: str) -> float:
        """Score how relevant a context passage is to a query.

        Uses TF-IDF cosine similarity between query and context.
        """
        return self._tfidf_cosine(context, query)

    def answer_relevancy(self, answer: str, query: str) -> float:
        """Score how relevant an answer is to the original query.

        Uses TF-IDF cosine similarity.
        """
        return self._tfidf_cosine(answer, query)

    def faithfulness(self, answer: str, context: str) -> float:
        """Score how faithful an answer is to its source context.

        Keyword overlap: fraction of answer keywords found in context.
        """
        answer_words = set(re.findall(r"[a-zA-Z]+", answer.lower()))
        answer_words = {w for w in answer_words if len(w) > 2}

        if not answer_words:
            return 0.0

        context_words = set(re.findall(r"[a-zA-Z]+", context.lower()))
        context_words = {w for w in context_words if len(w) > 2}

        overlap = answer_words & context_words
        return len(overlap) / len(answer_words)

    def evaluate_rag(
        self,
        query: str,
        context: str,
        answer: str,
        retrieved: list[str] | None = None,
        relevant: set[str] | None = None,
        k: int = 5,
    ) -> dict[str, float]:
        """Full RAG evaluation combining retrieval and generation metrics.

        Returns dict with: context_relevancy, answer_relevancy, faithfulness,
        and optionally mrr, ndcg, precision, recall, hit_rate if
        retrieved/relevant provided.
        """
        result: dict[str, float] = {
            "context_relevancy": self.context_relevancy(context, query),
            "answer_relevancy": self.answer_relevancy(answer, query),
            "faithfulness": self.faithfulness(answer, context),
        }

        if retrieved is not None and relevant is not None:
            retrieval = self.evaluate_single(retrieved, relevant, k=k)
            result.update(retrieval)

        return result
