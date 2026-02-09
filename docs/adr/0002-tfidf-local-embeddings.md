# ADR 0002: TF-IDF Local Embeddings

## Status

Accepted

## Context

Embedding generation is a core dependency of the retrieval pipeline. The main options are:

1. **External API embeddings** (OpenAI `text-embedding-3-small`, Cohere `embed-v3`): High semantic quality, but add latency (100-500ms per call), per-token cost ($0.02-0.10/1M tokens), and an external dependency that can fail.
2. **Local transformer models** (sentence-transformers): High quality, zero API cost, but require GPU for acceptable speed and add ~500MB+ model weight downloads.
3. **Local TF-IDF** (scikit-learn): Lower semantic quality than transformers, but zero cost, zero latency, zero external dependency, and works on any hardware.

For a document QA engine targeting moderate corpora (<100K documents) where the primary goal is demonstrating RAG architecture and retrieval strategies, TF-IDF provides adequate semantic representation.

## Decision

Use **TF-IDF vectorization** via scikit-learn as the default embedding method. Configuration:

- `max_features=5000` (vocabulary size)
- Sublinear TF scaling for better term discrimination
- Dense embeddings available as an opt-in via the `EmbedderBackend` abstraction

## Consequences

### Positive

- **Zero API cost**: No external billing or API key management
- **Zero latency overhead**: Embedding happens in-process (<1ms for typical documents)
- **Works offline**: No network dependency for core functionality
- **Reproducible**: Same input always produces same embeddings (deterministic)
- **Simple deployment**: No model downloads, GPU requirements, or API key provisioning

### Negative

- Lower semantic quality than transformer-based embeddings (no synonym/paraphrase understanding beyond co-occurrence)
- Vocabulary is corpus-dependent (new terms not seen in training get zero weight)
- Not suitable for cross-lingual or highly semantic queries

### Mitigation

- Hybrid retrieval (ADR-0001) compensates by combining BM25 keyword matching with TF-IDF dense similarity
- Query expansion (synonym, PRF) adds semantic coverage without changing the embedding model
- Architecture supports plugging in transformer embeddings when higher quality is needed
