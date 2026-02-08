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

    # ------------------------------------------------------------------
    # Enhanced generation quality metrics
    # ------------------------------------------------------------------

    def check_faithfulness(self, answer: str, context: str) -> dict[str, any]:
        """Check what fraction of answer sentences are supported by context.

        A sentence is "supported" if >50% of its keywords appear in context.

        Returns:
            dict with 'score', 'supported_sentences', 'unsupported_sentences'.
        """
        if not answer or not answer.strip():
            return {
                "score": 0.0,
                "supported_sentences": [],
                "unsupported_sentences": [],
            }

        # Split answer into sentences
        sentences = re.split(r"[.!?]+", answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {
                "score": 0.0,
                "supported_sentences": [],
                "unsupported_sentences": [],
            }

        # Extract context keywords
        context_words = set(re.findall(r"[a-zA-Z]+", context.lower()))
        context_words = {w for w in context_words if len(w) > 2}

        supported = []
        unsupported = []

        for sentence in sentences:
            sentence_words = set(re.findall(r"[a-zA-Z]+", sentence.lower()))
            sentence_words = {w for w in sentence_words if len(w) > 2}

            if not sentence_words:
                # Empty sentence -> consider unsupported
                unsupported.append(sentence)
                continue

            # Check keyword overlap
            overlap = sentence_words & context_words
            overlap_ratio = len(overlap) / len(sentence_words)

            if overlap_ratio > 0.5:
                supported.append(sentence)
            else:
                unsupported.append(sentence)

        score = len(supported) / len(sentences) if sentences else 0.0

        return {
            "score": score,
            "supported_sentences": supported,
            "unsupported_sentences": unsupported,
        }

    def check_completeness(self, answer: str, question: str, context: str) -> dict[str, any]:
        """Check if answer addresses key terms from the question.

        Extracts key terms (>2 chars) from question and checks if answer contains them.

        Returns:
            dict with 'score', 'addressed_terms', 'missing_terms'.
        """
        if not question or not question.strip():
            return {
                "score": 0.0,
                "addressed_terms": [],
                "missing_terms": [],
            }

        # Extract question keywords
        question_words = set(re.findall(r"[a-zA-Z]+", question.lower()))
        question_words = {w for w in question_words if len(w) > 2}

        # Filter out common question words
        stop_words = {"what", "when", "where", "which", "who", "how", "why", "does", "did", "can", "will", "would"}
        question_words = question_words - stop_words

        if not question_words:
            return {
                "score": 0.0,
                "addressed_terms": [],
                "missing_terms": [],
            }

        # Extract answer keywords
        answer_words = set(re.findall(r"[a-zA-Z]+", answer.lower()))
        answer_words = {w for w in answer_words if len(w) > 2}

        # Check which question terms are in the answer
        addressed = sorted(list(question_words & answer_words))
        missing = sorted(list(question_words - answer_words))

        score = len(addressed) / len(question_words) if question_words else 0.0

        return {
            "score": score,
            "addressed_terms": addressed,
            "missing_terms": missing,
        }

    def detect_hallucinations(self, answer: str, context: str) -> dict[str, any]:
        """Detect claims in answer NOT supported by context.

        A sentence is a "hallucination" if <30% of its keywords appear in context.

        Returns:
            dict with 'hallucination_count', 'hallucinated_sentences',
            'total_sentences', 'hallucination_rate'.
        """
        if not answer or not answer.strip():
            return {
                "hallucination_count": 0,
                "hallucinated_sentences": [],
                "total_sentences": 0,
                "hallucination_rate": 0.0,
            }

        # Split answer into sentences
        sentences = re.split(r"[.!?]+", answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {
                "hallucination_count": 0,
                "hallucinated_sentences": [],
                "total_sentences": 0,
                "hallucination_rate": 0.0,
            }

        # Extract context keywords
        context_words = set(re.findall(r"[a-zA-Z]+", context.lower()))
        context_words = {w for w in context_words if len(w) > 2}

        hallucinated = []

        for sentence in sentences:
            sentence_words = set(re.findall(r"[a-zA-Z]+", sentence.lower()))
            sentence_words = {w for w in sentence_words if len(w) > 2}

            if not sentence_words:
                # Empty sentence -> not a hallucination
                continue

            # Check keyword overlap
            overlap = sentence_words & context_words
            overlap_ratio = len(overlap) / len(sentence_words)

            if overlap_ratio < 0.3:
                # Low overlap -> likely hallucination
                hallucinated.append(sentence)

        hallucination_count = len(hallucinated)
        total_sentences = len(sentences)
        hallucination_rate = hallucination_count / total_sentences if total_sentences > 0 else 0.0

        return {
            "hallucination_count": hallucination_count,
            "hallucinated_sentences": hallucinated,
            "total_sentences": total_sentences,
            "hallucination_rate": hallucination_rate,
        }

    def context_quality(self, retrieved_chunks: list[str], relevant_chunks: list[str]) -> dict[str, float]:
        """Compute precision, recall, and F1 for retrieved vs relevant chunks.

        Args:
            retrieved_chunks: List of retrieved chunk identifiers.
            relevant_chunks: List of ground-truth relevant chunk identifiers.

        Returns:
            dict with 'precision', 'recall', 'f1'.
        """
        if not retrieved_chunks and not relevant_chunks:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        retrieved_set = set(retrieved_chunks)
        relevant_set = set(relevant_chunks)

        if not retrieved_set:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if not relevant_set:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Calculate intersection
        intersection = retrieved_set & relevant_set

        # Precision: fraction of retrieved that are relevant
        precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0

        # Recall: fraction of relevant that were retrieved
        recall = len(intersection) / len(relevant_set) if relevant_set else 0.0

        # F1: harmonic mean
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
