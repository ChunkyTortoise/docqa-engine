"""Tests for JSON and CSV export functionality."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from docqa_engine.batch import QueryResult
from docqa_engine.exporter import Exporter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def exporter() -> Exporter:
    return Exporter()


@pytest.fixture()
def sample_results() -> list[QueryResult]:
    return [
        QueryResult(
            query="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            sources=["doc1.txt", "doc2.pdf"],
            confidence=0.92,
            elapsed_ms=150.5,
        ),
        QueryResult(
            query="How does BM25 work?",
            answer="BM25 is a bag-of-words ranking function.",
            sources=["retrieval_guide.md"],
            confidence=0.85,
            elapsed_ms=98.3,
        ),
    ]


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


class TestJsonExport:
    """Tests for to_json."""

    def test_json_export(
        self, exporter: Exporter, sample_results: list[QueryResult], tmp_path: Path
    ) -> None:
        """Exported JSON contains metadata and all results."""
        out = tmp_path / "results.json"
        path = exporter.to_json(sample_results, str(out))

        assert Path(path).exists()
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        assert "metadata" in data
        assert data["metadata"]["total_queries"] == 2
        assert "version" in data["metadata"]
        assert "timestamp" in data["metadata"]

        assert len(data["results"]) == 2
        assert data["results"][0]["query"] == "What is RAG?"
        assert data["results"][0]["sources"] == ["doc1.txt", "doc2.pdf"]
        assert data["results"][1]["confidence"] == 0.85

    def test_json_empty_results(self, exporter: Exporter, tmp_path: Path) -> None:
        """Empty results list produces valid JSON with zero count."""
        out = tmp_path / "empty.json"
        path = exporter.to_json([], str(out))

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["metadata"]["total_queries"] == 0
        assert data["results"] == []

    def test_json_special_characters(self, exporter: Exporter, tmp_path: Path) -> None:
        """Special characters (quotes, newlines, unicode) are preserved."""
        results = [
            QueryResult(
                query='What is "RAG"?',
                answer="Line 1\nLine 2\tTabbed.\nUnicode: \u00e9\u00e0\u00fc\u00f1",
                sources=["doc with spaces.txt"],
                confidence=0.5,
                elapsed_ms=42.0,
            ),
        ]
        out = tmp_path / "special.json"
        path = exporter.to_json(results, str(out))

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["results"][0]["query"] == 'What is "RAG"?'
        assert "\n" in data["results"][0]["answer"]
        assert "\u00e9" in data["results"][0]["answer"]


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


class TestCsvExport:
    """Tests for to_csv."""

    def test_csv_export(
        self, exporter: Exporter, sample_results: list[QueryResult], tmp_path: Path
    ) -> None:
        """Exported CSV has correct headers and rows."""
        out = tmp_path / "results.csv"
        path = exporter.to_csv(sample_results, str(out))

        assert Path(path).exists()
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Header + 2 data rows
        assert len(rows) == 3
        assert rows[0] == ["query", "answer", "sources", "confidence", "elapsed_ms"]
        assert rows[1][0] == "What is RAG?"
        assert "doc1.txt" in rows[1][2]
        assert "doc2.pdf" in rows[1][2]

    def test_csv_empty_results(self, exporter: Exporter, tmp_path: Path) -> None:
        """Empty results list produces a CSV with only the header row."""
        out = tmp_path / "empty.csv"
        path = exporter.to_csv([], str(out))

        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1  # header only


# ---------------------------------------------------------------------------
# Evaluation export
# ---------------------------------------------------------------------------


class TestEvaluationExport:
    """Tests for export_evaluation."""

    def test_evaluation_export(self, exporter: Exporter, tmp_path: Path) -> None:
        """Evaluation metrics are wrapped with metadata and written to JSON."""
        metrics = {
            "mrr": 0.75,
            "ndcg": 0.82,
            "precision": 0.60,
            "recall": 0.90,
            "hit_rate": 1.0,
        }
        out = tmp_path / "eval.json"
        path = exporter.export_evaluation(metrics, str(out))

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "metadata" in data
        assert "timestamp" in data["metadata"]
        assert data["metrics"]["mrr"] == 0.75
        assert data["metrics"]["hit_rate"] == 1.0
