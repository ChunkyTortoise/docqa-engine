"""Cost Tracking Dashboard: per-query token usage and cumulative cost by provider."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Approximate pricing per 1K tokens (input/output averaged)
PROVIDER_PRICING = {
    "claude": 0.008,
    "openai": 0.010,
    "gemini": 0.003,
    "perplexity": 0.005,
    "mock": 0.000,
}


@dataclass
class QueryCost:
    query_id: str
    question: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class CostTracker:
    """Track per-query costs and aggregate by provider."""

    def __init__(self, pricing: dict[str, float] | None = None):
        self.pricing = pricing or PROVIDER_PRICING.copy()
        self.queries: list[QueryCost] = []

    def estimate_cost(self, provider: str, total_tokens: int) -> float:
        """Estimate cost for a given provider and token count."""
        rate = self.pricing.get(provider, 0.01)
        return round(rate * total_tokens / 1000, 6)

    def record_query(
        self,
        query_id: str,
        question: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> QueryCost:
        """Record a query with cost tracking."""
        total = input_tokens + output_tokens
        cost = self.estimate_cost(provider, total)
        entry = QueryCost(
            query_id=query_id,
            question=question,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            cost_estimate=cost,
        )
        self.queries.append(entry)
        return entry

    def total_cost(self) -> float:
        """Total cost across all queries."""
        return round(sum(q.cost_estimate for q in self.queries), 6)

    def cost_by_provider(self) -> dict[str, float]:
        """Cumulative cost grouped by provider."""
        costs: dict[str, float] = defaultdict(float)
        for q in self.queries:
            costs[q.provider] += q.cost_estimate
        return {k: round(v, 6) for k, v in costs.items()}

    def tokens_by_provider(self) -> dict[str, int]:
        """Total tokens grouped by provider."""
        tokens: dict[str, int] = defaultdict(int)
        for q in self.queries:
            tokens[q.provider] += q.total_tokens
        return dict(tokens)

    def query_count(self) -> int:
        return len(self.queries)

    def summary(self) -> dict[str, Any]:
        """Full cost summary."""
        return {
            "total_queries": self.query_count(),
            "total_cost": self.total_cost(),
            "total_tokens": sum(q.total_tokens for q in self.queries),
            "by_provider": {
                provider: {
                    "queries": sum(1 for q in self.queries if q.provider == provider),
                    "tokens": tokens,
                    "cost": self.cost_by_provider().get(provider, 0),
                }
                for provider, tokens in self.tokens_by_provider().items()
            },
        }
