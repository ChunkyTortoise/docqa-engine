"""Export Functionality: JSON and CSV export for Q&A results and evaluation metrics."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from docqa_engine import __version__
from docqa_engine.batch import QueryResult


class Exporter:
    """Export Q&A results and evaluation metrics to JSON or CSV files.

    Supports three export modes:

    - **JSON export** of query results with metadata (timestamp, version,
      total queries).
    - **CSV export** of query results with columns: query, answer, sources,
      confidence, elapsed_ms.
    - **JSON export** of evaluation metric dictionaries.
    """

    # ------------------------------------------------------------------
    # Q&A Results -> JSON
    # ------------------------------------------------------------------

    def to_json(self, results: list[QueryResult], output_path: str) -> str:
        """Export Q&A results to a JSON file.

        Includes metadata: timestamp, total_queries, version.

        Args:
            results: List of :class:`QueryResult` objects.
            output_path: Destination file path.

        Returns:
            The absolute path of the written file.
        """
        data = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_queries": len(results),
                "version": __version__,
            },
            "results": [
                {
                    "query": r.query,
                    "answer": r.answer,
                    "sources": r.sources,
                    "confidence": r.confidence,
                    "elapsed_ms": r.elapsed_ms,
                }
                for r in results
            ],
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path.resolve())

    # ------------------------------------------------------------------
    # Q&A Results -> CSV
    # ------------------------------------------------------------------

    def to_csv(self, results: list[QueryResult], output_path: str) -> str:
        """Export Q&A results to a CSV file.

        Columns: ``query``, ``answer``, ``sources``, ``confidence``,
        ``elapsed_ms``.  The ``sources`` column is semicolon-delimited.

        Args:
            results: List of :class:`QueryResult` objects.
            output_path: Destination file path.

        Returns:
            The absolute path of the written file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query", "answer", "sources", "confidence", "elapsed_ms"])
            for r in results:
                writer.writerow(
                    [
                        r.query,
                        r.answer,
                        "; ".join(r.sources),
                        r.confidence,
                        r.elapsed_ms,
                    ]
                )

        return str(path.resolve())

    # ------------------------------------------------------------------
    # Evaluation Metrics -> JSON
    # ------------------------------------------------------------------

    def export_evaluation(self, eval_results: dict, output_path: str) -> str:
        """Export evaluation metrics to a JSON file.

        Wraps the raw metrics dict with a timestamp and version.

        Args:
            eval_results: Dictionary of metric names to values (e.g. from
                :meth:`Evaluator.evaluate`).
            output_path: Destination file path.

        Returns:
            The absolute path of the written file.
        """
        data = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": __version__,
            },
            "metrics": eval_results,
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path.resolve())
