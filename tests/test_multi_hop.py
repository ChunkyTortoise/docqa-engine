"""Tests for multi-hop retrieval."""

from __future__ import annotations

from docqa_engine.multi_hop import MultiHopResult, MultiHopRetriever, RetrievedPassage

SAMPLE_DOCS = [
    "Python is a high-level programming language used for web development and data science.",
    "JavaScript is primarily used for frontend web development and browser scripting.",
    "Machine learning models require training data and computational resources.",
    "SQL databases store structured data in tables with rows and columns.",
    "Docker containers package applications with their dependencies for deployment.",
    "REST APIs use HTTP methods like GET, POST, PUT, and DELETE.",
    "Git version control tracks changes in source code during development.",
    "Cloud computing provides on-demand computing resources over the internet.",
]


class TestSingleHop:
    def setup_method(self):
        self.retriever = MultiHopRetriever()

    def test_basic_retrieval(self):
        results = self.retriever._single_hop("Python programming", SAMPLE_DOCS)
        assert len(results) > 0
        assert isinstance(results[0], RetrievedPassage)
        assert results[0].score > 0
        # Python doc should rank high
        assert "Python" in results[0].text or "python" in results[0].text.lower()

    def test_empty_docs(self):
        results = self.retriever._single_hop("test", [])
        assert results == []

    def test_empty_query(self):
        results = self.retriever._single_hop("", SAMPLE_DOCS)
        assert results == []

    def test_single_doc(self):
        results = self.retriever._single_hop("Python", ["Python is great."])
        assert len(results) == 1

    def test_top_k_limit(self):
        results = self.retriever._single_hop("development", SAMPLE_DOCS, top_k=2)
        assert len(results) <= 2

    def test_scores_sorted(self):
        results = self.retriever._single_hop("web development", SAMPLE_DOCS)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score


class TestQueryDecomposition:
    def setup_method(self):
        self.retriever = MultiHopRetriever()

    def test_compound_and(self):
        parts = self.retriever.decompose_query("What is Python and what is JavaScript?")
        assert len(parts) == 2

    def test_semicolon(self):
        parts = self.retriever.decompose_query("Python programming; JavaScript frameworks")
        assert len(parts) == 2

    def test_multiple_questions(self):
        parts = self.retriever.decompose_query("What is Python? What is JavaScript?")
        assert len(parts) == 2

    def test_simple_query_no_split(self):
        parts = self.retriever.decompose_query("What is Python?")
        assert len(parts) == 1

    def test_empty_query(self):
        assert self.retriever.decompose_query("") == []

    def test_compared_to(self):
        parts = self.retriever.decompose_query("Python compared to JavaScript")
        assert len(parts) == 2


class TestMergeResults:
    def setup_method(self):
        self.retriever = MultiHopRetriever()

    def test_dedup(self):
        results = [
            RetrievedPassage(text="Same text", score=0.8, hop_number=1, source_query="q1"),
            RetrievedPassage(text="Same text", score=0.6, hop_number=2, source_query="q2"),
            RetrievedPassage(text="Different text", score=0.5, hop_number=1, source_query="q1"),
        ]
        merged = self.retriever.merge_results(results)
        assert len(merged) == 2
        # Should keep higher score
        same_text_entry = [p for p in merged if p.text == "Same text"][0]
        assert same_text_entry.score == 0.8

    def test_sorted_by_score(self):
        results = [
            RetrievedPassage(text="A", score=0.3, hop_number=1, source_query="q"),
            RetrievedPassage(text="B", score=0.9, hop_number=1, source_query="q"),
            RetrievedPassage(text="C", score=0.6, hop_number=1, source_query="q"),
        ]
        merged = self.retriever.merge_results(results)
        assert merged[0].text == "B"
        assert merged[-1].text == "A"

    def test_empty(self):
        assert self.retriever.merge_results([]) == []


class TestMultiHopRetrieval:
    def setup_method(self):
        self.retriever = MultiHopRetriever()

    def test_multi_hop_compound(self):
        result = self.retriever.retrieve("What is Python and what is Docker?", SAMPLE_DOCS)
        assert isinstance(result, MultiHopResult)
        assert result.hops_used == 2
        assert len(result.sub_queries) == 2
        assert len(result.passages) > 0

    def test_single_query_one_hop(self):
        result = self.retriever.retrieve("Python programming", SAMPLE_DOCS)
        assert result.hops_used == 1

    def test_max_hops_limit(self):
        result = self.retriever.retrieve("A; B; C; D; E", SAMPLE_DOCS, max_hops=2)
        assert result.hops_used <= 2

    def test_empty_query(self):
        result = self.retriever.retrieve("", SAMPLE_DOCS)
        assert result.passages == []
        assert result.hops_used == 0

    def test_empty_docs(self):
        result = self.retriever.retrieve("test", [])
        assert result.passages == []

    def test_merged_scores_populated(self):
        result = self.retriever.retrieve("Python and Docker", SAMPLE_DOCS)
        assert len(result.merged_scores) == len(result.passages)

    def test_multi_hop_finds_more(self):
        # Compound query should potentially find passages from both topics
        compound = self.retriever.retrieve("Python programming and Docker containers", SAMPLE_DOCS)
        simple = self.retriever.retrieve("Python programming", SAMPLE_DOCS)
        # Compound should find at least as many unique topics
        compound_texts = {p.text for p in compound.passages}
        simple_texts = {p.text for p in simple.passages}
        assert len(compound_texts) >= len(simple_texts)
