"""Tests for query expansion: synonym, PRF, decomposition, merge."""

from __future__ import annotations

import pytest

from docqa_engine.query_expansion import ExpandedQuery, QueryExpander


@pytest.fixture()
def expander() -> QueryExpander:
    return QueryExpander()


# ---------------------------------------------------------------------------
# Synonym expansion
# ---------------------------------------------------------------------------


class TestSynonymExpand:
    """Tests for synonym_expand."""

    def test_synonym_found(self, expander: QueryExpander) -> None:
        """Known synonym terms are expanded."""
        result = expander.synonym_expand("machine learning model")
        assert len(result.expanded_terms) > 0
        terms = [t for t, _ in result.expanded_terms]
        assert any("ML" in t or "artificial intelligence" in t or "deep learning" in t for t in terms)

    def test_synonym_boost_factor(self, expander: QueryExpander) -> None:
        """Synonym terms have boost factor 0.5."""
        result = expander.synonym_expand("database query")
        for _, boost in result.expanded_terms:
            assert boost == 0.5

    def test_no_synonyms_found(self, expander: QueryExpander) -> None:
        """Unknown terms produce no expansions."""
        result = expander.synonym_expand("xylophone music")
        assert len(result.expanded_terms) == 0

    def test_synonym_strategy_label(self, expander: QueryExpander) -> None:
        """Strategy is set to 'synonym'."""
        result = expander.synonym_expand("test query")
        assert result.strategy == "synonym"

    def test_synonym_empty_query(self, expander: QueryExpander) -> None:
        """Empty query returns no expansions."""
        result = expander.synonym_expand("")
        assert len(result.expanded_terms) == 0
        assert result.original_query == ""

    def test_synonym_preserves_original(self, expander: QueryExpander) -> None:
        """Original query is preserved."""
        result = expander.synonym_expand("search for data")
        assert result.original_query == "search for data"

    def test_multiple_synonyms_matched(self, expander: QueryExpander) -> None:
        """Multiple terms can match synonyms."""
        result = expander.synonym_expand("search database")
        assert len(result.expanded_terms) >= 2


# ---------------------------------------------------------------------------
# PRF expansion
# ---------------------------------------------------------------------------


class TestPRFExpand:
    """Tests for prf_expand (pseudo-relevance feedback)."""

    def test_prf_extracts_terms(self, expander: QueryExpander) -> None:
        """PRF extracts top TF-IDF terms from feedback docs."""
        feedback = [
            "Python is excellent for machine learning and data science projects.",
            "Data science uses Python libraries like scikit-learn and pandas.",
            "Machine learning algorithms process large datasets efficiently.",
        ]
        result = expander.prf_expand("programming", feedback)
        assert len(result.expanded_terms) > 0
        assert result.strategy == "prf"

    def test_prf_terms_not_in_query(self, expander: QueryExpander) -> None:
        """PRF terms should not duplicate query words."""
        feedback = [
            "Machine learning uses neural networks for classification.",
            "Classification algorithms include decision trees and SVM.",
        ]
        result = expander.prf_expand("classification", feedback)
        terms = [t for t, _ in result.expanded_terms]
        assert "classification" not in terms

    def test_prf_top_terms_limit(self, expander: QueryExpander) -> None:
        """PRF respects top_terms limit."""
        feedback = [
            "Python numpy pandas scikit-learn tensorflow keras pytorch data science analytics visualization modeling.",
        ]
        result = expander.prf_expand("query", feedback, top_terms=3)
        assert len(result.expanded_terms) <= 3

    def test_prf_empty_feedback(self, expander: QueryExpander) -> None:
        """Empty feedback docs return no expansions."""
        result = expander.prf_expand("query", [])
        assert len(result.expanded_terms) == 0

    def test_prf_empty_query(self, expander: QueryExpander) -> None:
        """Empty query returns no expansions."""
        result = expander.prf_expand("", ["some doc"])
        assert len(result.expanded_terms) == 0

    def test_prf_boost_is_positive(self, expander: QueryExpander) -> None:
        """PRF boost values are positive."""
        feedback = ["Python data science machine learning analytics."]
        result = expander.prf_expand("query", feedback)
        for _, boost in result.expanded_terms:
            assert boost > 0


# ---------------------------------------------------------------------------
# Decompose
# ---------------------------------------------------------------------------


class TestDecompose:
    """Tests for decompose (sub-query splitting)."""

    def test_split_on_and(self, expander: QueryExpander) -> None:
        """Splits compound query on 'and'."""
        result = expander.decompose("Python and Java")
        assert len(result.sub_queries) == 2
        assert "Python" in result.sub_queries[0]
        assert "Java" in result.sub_queries[1]

    def test_split_on_or(self, expander: QueryExpander) -> None:
        """Splits compound query on 'or'."""
        result = expander.decompose("Python or Ruby")
        assert len(result.sub_queries) == 2

    def test_no_split_simple_query(self, expander: QueryExpander) -> None:
        """Simple query without connectors returns single sub-query."""
        result = expander.decompose("Python programming")
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "Python programming"

    def test_decompose_empty_query(self, expander: QueryExpander) -> None:
        """Empty query returns no sub-queries."""
        result = expander.decompose("")
        assert result.strategy == "decompose"

    def test_decompose_strategy_label(self, expander: QueryExpander) -> None:
        """Strategy is set to 'decompose'."""
        result = expander.decompose("some query")
        assert result.strategy == "decompose"


# ---------------------------------------------------------------------------
# Expand all
# ---------------------------------------------------------------------------


class TestExpandAll:
    """Tests for expand_all combining strategies."""

    def test_expand_all_combines_terms(self, expander: QueryExpander) -> None:
        """expand_all combines synonym and other expansions."""
        result = expander.expand_all("search database")
        assert result.strategy == "all"
        assert len(result.expanded_terms) > 0

    def test_expand_all_with_feedback(self, expander: QueryExpander) -> None:
        """expand_all includes PRF terms when feedback provided."""
        feedback = ["Python machine learning algorithms classification."]
        result = expander.expand_all("search", feedback_docs=feedback)
        assert result.strategy == "all"

    def test_expand_all_has_sub_queries(self, expander: QueryExpander) -> None:
        """expand_all includes decomposed sub-queries."""
        result = expander.expand_all("Python and Java")
        assert len(result.sub_queries) == 2


# ---------------------------------------------------------------------------
# Merge expansions
# ---------------------------------------------------------------------------


class TestMergeExpansions:
    """Tests for merge_expansions."""

    def test_merge_deduplicates_terms(self, expander: QueryExpander) -> None:
        """Duplicate terms are deduplicated."""
        exp1 = ExpandedQuery(original_query="q", expanded_terms=[("python", 0.5), ("data", 0.3)])
        exp2 = ExpandedQuery(original_query="q", expanded_terms=[("python", 0.8), ("java", 0.4)])
        merged = expander.merge_expansions([exp1, exp2])

        term_dict = dict(merged.expanded_terms)
        assert term_dict["python"] == 0.8  # Highest boost kept
        assert "data" in term_dict
        assert "java" in term_dict

    def test_merge_deduplicates_subqueries(self, expander: QueryExpander) -> None:
        """Duplicate sub-queries are removed."""
        exp1 = ExpandedQuery(original_query="q", sub_queries=["Python", "Java"])
        exp2 = ExpandedQuery(original_query="q", sub_queries=["python", "Ruby"])
        merged = expander.merge_expansions([exp1, exp2])

        # "python" (case-insensitive) should appear once
        lower_sqs = [sq.lower() for sq in merged.sub_queries]
        assert lower_sqs.count("python") == 1

    def test_merge_empty_list(self, expander: QueryExpander) -> None:
        """Empty list returns empty expansion."""
        merged = expander.merge_expansions([])
        assert merged.original_query == ""
        assert merged.expanded_terms == []
        assert merged.strategy == "merged"

    def test_merge_preserves_original_query(self, expander: QueryExpander) -> None:
        """First expansion's original query is used."""
        exp1 = ExpandedQuery(original_query="first query")
        exp2 = ExpandedQuery(original_query="second query")
        merged = expander.merge_expansions([exp1, exp2])
        assert merged.original_query == "first query"
