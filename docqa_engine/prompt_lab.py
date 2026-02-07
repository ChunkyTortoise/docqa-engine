"""Prompt Engineering Lab: versioning, A/B testing, evaluation scoring."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class PromptVersion:
    version_id: str
    name: str
    template: str
    temperature: float = 0.7
    max_tokens: int = 1024
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class EvalScore:
    faithfulness: float  # 0-1: answer grounded in context
    relevance: float  # 0-1: answer addresses the question
    completeness: float  # 0-1: all aspects of question covered
    overall: float = 0.0

    def __post_init__(self):
        self.overall = round(
            (self.faithfulness + self.relevance + self.completeness) / 3, 4
        )


@dataclass
class ExperimentResult:
    version_id: str
    question: str
    answer: str
    eval_score: EvalScore
    provider: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0


@dataclass
class ABTestResult:
    version_a: str
    version_b: str
    results_a: list[ExperimentResult]
    results_b: list[ExperimentResult]
    winner: str = ""
    margin: float = 0.0

    def __post_init__(self):
        if self.results_a and self.results_b:
            avg_a = statistics.mean(r.eval_score.overall for r in self.results_a)
            avg_b = statistics.mean(r.eval_score.overall for r in self.results_b)
            self.margin = round(abs(avg_a - avg_b), 4)
            self.winner = self.version_a if avg_a >= avg_b else self.version_b


class PromptLab:
    """Prompt versioning and experimentation framework."""

    def __init__(self):
        self.versions: dict[str, PromptVersion] = {}
        self.experiments: list[ExperimentResult] = []

    def create_version(
        self, name: str, template: str, temperature: float = 0.7, max_tokens: int = 1024, **kwargs
    ) -> PromptVersion:
        """Create a new prompt version."""
        version = PromptVersion(
            version_id=str(uuid4())[:8],
            name=name,
            template=template,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=kwargs,
        )
        self.versions[version.version_id] = version
        return version

    def get_version(self, version_id: str) -> PromptVersion | None:
        return self.versions.get(version_id)

    def list_versions(self) -> list[PromptVersion]:
        return list(self.versions.values())

    def render_prompt(self, version_id: str, **kwargs) -> str:
        """Render a prompt template with variables."""
        version = self.versions.get(version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")
        return version.template.format(**kwargs)

    def record_result(
        self,
        version_id: str,
        question: str,
        answer: str,
        faithfulness: float,
        relevance: float,
        completeness: float,
        provider: str = "",
        tokens_used: int = 0,
        latency_ms: float = 0.0,
    ) -> ExperimentResult:
        """Record an experiment result."""
        result = ExperimentResult(
            version_id=version_id,
            question=question,
            answer=answer,
            eval_score=EvalScore(
                faithfulness=faithfulness, relevance=relevance, completeness=completeness
            ),
            provider=provider,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )
        self.experiments.append(result)
        return result

    def get_version_stats(self, version_id: str) -> dict[str, Any]:
        """Get aggregated stats for a prompt version."""
        results = [e for e in self.experiments if e.version_id == version_id]
        if not results:
            return {"version_id": version_id, "runs": 0}

        scores = [r.eval_score.overall for r in results]
        return {
            "version_id": version_id,
            "runs": len(results),
            "avg_overall": round(statistics.mean(scores), 4),
            "min_overall": round(min(scores), 4),
            "max_overall": round(max(scores), 4),
            "std_overall": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "avg_tokens": round(statistics.mean(r.tokens_used for r in results)),
            "avg_latency_ms": round(statistics.mean(r.latency_ms for r in results), 1),
        }

    def compare_versions(self, version_a_id: str, version_b_id: str) -> ABTestResult:
        """Compare two prompt versions based on recorded experiments."""
        results_a = [e for e in self.experiments if e.version_id == version_a_id]
        results_b = [e for e in self.experiments if e.version_id == version_b_id]
        return ABTestResult(
            version_a=version_a_id,
            version_b=version_b_id,
            results_a=results_a,
            results_b=results_b,
        )

    def export_experiments(self) -> str:
        """Export all experiments as JSON."""
        data = []
        for e in self.experiments:
            data.append({
                "version_id": e.version_id,
                "question": e.question,
                "answer": e.answer[:200],
                "faithfulness": e.eval_score.faithfulness,
                "relevance": e.eval_score.relevance,
                "completeness": e.eval_score.completeness,
                "overall": e.eval_score.overall,
                "provider": e.provider,
                "tokens_used": e.tokens_used,
                "latency_ms": e.latency_ms,
            })
        return json.dumps(data, indent=2)
