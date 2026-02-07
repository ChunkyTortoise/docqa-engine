"""Tests for the prompt engineering lab module."""

import pytest

from docqa_engine.prompt_lab import (
    ABTestResult,
    EvalScore,
    ExperimentResult,
    PromptLab,
    PromptVersion,
)


@pytest.fixture
def lab():
    lab = PromptLab()
    v1 = lab.create_version(
        "concise", "Answer briefly: {context}\nQ: {question}\nA:", temperature=0.3
    )
    v2 = lab.create_version(
        "detailed", "Provide a detailed answer with examples: {context}\nQ: {question}\nA:",
        temperature=0.7,
    )
    return lab, v1, v2


class TestPromptVersion:
    def test_create(self, lab):
        lab_obj, v1, v2 = lab
        assert v1.name == "concise"
        assert v2.name == "detailed"
        assert v1.temperature == 0.3
        assert v1.created_at

    def test_version_id(self, lab):
        lab_obj, v1, v2 = lab
        assert v1.version_id != v2.version_id

    def test_list_versions(self, lab):
        lab_obj, _, _ = lab
        versions = lab_obj.list_versions()
        assert len(versions) == 2


class TestRenderPrompt:
    def test_basic(self, lab):
        lab_obj, v1, _ = lab
        rendered = lab_obj.render_prompt(v1.version_id, context="Some context", question="What?")
        assert "Some context" in rendered
        assert "What?" in rendered

    def test_missing_version(self, lab):
        lab_obj, _, _ = lab
        with pytest.raises(ValueError, match="not found"):
            lab_obj.render_prompt("nonexistent", context="", question="")


class TestEvalScore:
    def test_overall_calculation(self):
        score = EvalScore(faithfulness=0.8, relevance=0.9, completeness=0.7)
        assert abs(score.overall - 0.8) < 0.01

    def test_perfect_score(self):
        score = EvalScore(faithfulness=1.0, relevance=1.0, completeness=1.0)
        assert score.overall == 1.0


class TestRecordResult:
    def test_basic(self, lab):
        lab_obj, v1, _ = lab
        result = lab_obj.record_result(
            v1.version_id, "What is Python?", "A programming language.",
            faithfulness=0.9, relevance=0.85, completeness=0.8,
        )
        assert isinstance(result, ExperimentResult)
        assert result.eval_score.overall > 0

    def test_stats(self, lab):
        lab_obj, v1, _ = lab
        for i in range(5):
            lab_obj.record_result(
                v1.version_id, f"Q{i}", f"A{i}",
                faithfulness=0.8 + i * 0.02, relevance=0.7, completeness=0.9,
            )
        stats = lab_obj.get_version_stats(v1.version_id)
        assert stats["runs"] == 5
        assert stats["avg_overall"] > 0

    def test_no_stats(self, lab):
        lab_obj, _, _ = lab
        stats = lab_obj.get_version_stats("nonexistent")
        assert stats["runs"] == 0


class TestABTest:
    def test_compare(self, lab):
        lab_obj, v1, v2 = lab
        # Record results for v1 (good)
        for _ in range(3):
            lab_obj.record_result(v1.version_id, "Q", "A", 0.9, 0.9, 0.9)
        # Record results for v2 (mediocre)
        for _ in range(3):
            lab_obj.record_result(v2.version_id, "Q", "A", 0.5, 0.5, 0.5)

        result = lab_obj.compare_versions(v1.version_id, v2.version_id)
        assert isinstance(result, ABTestResult)
        assert result.winner == v1.version_id
        assert result.margin > 0

    def test_export(self, lab):
        lab_obj, v1, _ = lab
        lab_obj.record_result(v1.version_id, "Q", "A", 0.8, 0.8, 0.8)
        exported = lab_obj.export_experiments()
        assert "faithfulness" in exported
        assert "0.8" in exported
