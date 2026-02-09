# ADR 0003: Citation Scoring Framework

## Status

Accepted

## Context

RAG systems frequently produce answers that are:

1. **Unfaithful**: The answer makes claims not supported by the retrieved passages
2. **Incomplete**: The citations only address part of the question
3. **Redundant**: Multiple citations convey the same information, wasting context window budget

Without programmatic measurement, these failure modes are only detectable through manual review, which does not scale. We need automated quality scoring to:

- Compare retrieval strategies objectively (A/B testing)
- Set quality thresholds for production deployments
- Identify systematic failure patterns (e.g., certain query types consistently produce low-faithfulness answers)

## Decision

Implement a three-axis citation scoring framework:

1. **Faithfulness** (0.0-1.0): Measures whether the generated answer is supported by the cited passages. Computed via keyword overlap ratio between answer claims and source text.
2. **Coverage** (0.0-1.0): Measures whether the citations collectively address the full question. Computed as the fraction of query terms found across all cited passages.
3. **Redundancy** (0.0-1.0): Measures information duplication across citations. Computed via pairwise similarity between citation texts (lower is better).

Overall score = weighted combination: `0.4 * faithfulness + 0.4 * coverage + 0.2 * (1 - redundancy)`

## Consequences

### Positive

- **Quantified answer quality** enables automated A/B testing of retrieval strategies
- **Regression detection**: CI can fail if citation quality drops below threshold
- **Interpretable dimensions**: Each axis maps to a specific failure mode, making debugging actionable
- **No external API needed**: Scoring uses keyword overlap and TF-IDF similarity, not LLM judges

### Negative

- Keyword-based faithfulness is a proxy -- it cannot detect semantic faithfulness violations (e.g., negation)
- Coverage scoring assumes query terms map to information needs, which is imperfect for complex queries
- Adding scoring to every query adds ~5ms latency

### Mitigation

- Framework is extensible: LLM-based faithfulness scoring can be added as an optional high-accuracy mode
- Three separate scores allow users to weight dimensions based on their use case
- Scoring is opt-in per query, not mandatory
