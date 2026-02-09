"""Tests for the document summarizer."""

from __future__ import annotations

from docqa_engine.summarizer import DocumentSummarizer, KeyPhrase, SummaryResult


class TestSummarizeSingle:
    def setup_method(self):
        self.summarizer = DocumentSummarizer()

    def test_basic_summary(self):
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "It focuses on building systems that learn from data. "
            "Deep learning is a subset of machine learning. "
            "Neural networks are the foundation of deep learning. "
            "Training requires large datasets and compute resources."
        )
        result = self.summarizer.summarize(text, ratio=0.4)
        assert isinstance(result, SummaryResult)
        assert len(result.summary) > 0
        assert result.summary_length < result.original_length

    def test_compression_ratio(self):
        text = (
            "First sentence about data. Second sentence about analysis. "
            "Third sentence about results. Fourth sentence about conclusions. "
            "Fifth sentence about recommendations."
        )
        result = self.summarizer.summarize(text, ratio=0.3)
        assert 0 < result.compression_ratio <= 1.0

    def test_empty_input(self):
        result = self.summarizer.summarize("")
        assert result.summary == ""
        assert result.original_length == 0
        assert result.compression_ratio == 0.0

    def test_single_sentence(self):
        text = "This is the only sentence"
        result = self.summarizer.summarize(text)
        assert result.summary == text
        assert result.compression_ratio == 1.0

    def test_whitespace_only(self):
        result = self.summarizer.summarize("   ")
        assert result.summary == ""

    def test_very_long_text(self):
        sentences = [f"Sentence number {i} about topic {i % 5}." for i in range(50)]
        text = " ".join(sentences)
        result = self.summarizer.summarize(text, ratio=0.2)
        assert result.summary_length < result.original_length
        assert len(result.summary) > 0

    def test_key_phrases_in_result(self):
        text = (
            "Python programming is popular for data science. "
            "Data science uses Python for machine learning. "
            "Machine learning models are trained with Python libraries."
        )
        result = self.summarizer.summarize(text)
        assert len(result.key_phrases) > 0
        assert all(isinstance(kp, KeyPhrase) for kp in result.key_phrases)


class TestSummarizeMulti:
    def setup_method(self):
        self.summarizer = DocumentSummarizer()

    def test_multi_doc_merge(self):
        docs = [
            "Climate change affects weather patterns globally. Rising temperatures cause ice caps to melt.",
            "Renewable energy reduces carbon emissions. Solar and wind power are expanding rapidly.",
            "Electric vehicles help reduce urban pollution. Battery technology is improving steadily.",
        ]
        result = self.summarizer.summarize_multi(docs, ratio=0.4)
        assert isinstance(result, SummaryResult)
        assert len(result.summary) > 0

    def test_empty_docs(self):
        result = self.summarizer.summarize_multi([])
        assert result.summary == ""

    def test_single_doc_list(self):
        result = self.summarizer.summarize_multi(["One document with content here."])
        assert len(result.summary) > 0


class TestExtractKeyPhrases:
    def setup_method(self):
        self.summarizer = DocumentSummarizer()

    def test_extract_phrases(self):
        text = (
            "Python programming is great for data analysis. Data analysis with Python uses pandas and numpy libraries."
        )
        phrases = self.summarizer.extract_key_phrases(text, top_k=5)
        assert len(phrases) > 0
        assert all(isinstance(kp, KeyPhrase) for kp in phrases)
        assert all(kp.score > 0 for kp in phrases)

    def test_empty_text(self):
        assert self.summarizer.extract_key_phrases("") == []

    def test_top_k_limit(self):
        text = " ".join([f"topic_{i} content" for i in range(20)])
        phrases = self.summarizer.extract_key_phrases(text, top_k=3)
        assert len(phrases) <= 3


class TestScoreSentences:
    def setup_method(self):
        self.summarizer = DocumentSummarizer()

    def test_scoring(self):
        text = "First sentence. Second sentence. Third sentence."
        scored = self.summarizer._score_sentences(text)
        assert len(scored) == 3
        assert all(isinstance(s, tuple) and len(s) == 2 for s in scored)

    def test_single_sentence_score(self):
        scored = self.summarizer._score_sentences("Only one.")
        assert len(scored) == 1
        assert scored[0][1] == 1.0

    def test_empty_text(self):
        assert self.summarizer._score_sentences("") == []
