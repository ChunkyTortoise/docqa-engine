# ADR 0004: REST API Wrapper Design

## Status

Accepted

## Context

The core docqa-engine is a Python library consumed via direct import. However, enterprise and SaaS deployment scenarios require:

1. **HTTP access**: Clients in other languages or on other machines need network-accessible endpoints
2. **Authentication**: Multi-tenant deployments need per-user API key validation
3. **Rate limiting**: Prevent abuse and ensure fair usage across tenants
4. **Usage metering**: Track queries/day and tokens consumed for billing and capacity planning

Building these concerns into the core library would violate separation of concerns. A wrapper approach keeps the core library clean while adding enterprise features at the HTTP boundary.

## Decision

Wrap the engine in a **FastAPI REST API** with:

- **JWT/API-key authentication** via `X-API-Key` header with configurable valid key sets
- **Per-user rate limiting** using a sliding-window algorithm (default: 100 requests/60s)
- **Usage metering** tracking request count and token consumption per API key
- **Factory pattern** (`create_app()`) for testability -- the app can be instantiated with or without a real pipeline

Endpoints:
- `POST /ingest` -- Upload and index a document
- `POST /ask` -- Query the knowledge base
- `GET /stats` -- Usage statistics
- `DELETE /reset` -- Clear all documents

## Consequences

### Positive

- **Enterprise-ready deployment** without modifying core library code
- **Testable**: Factory pattern allows full API testing with mock pipelines
- **Configurable**: Rate limits, API keys, and pipeline injection are all constructor parameters
- **Standard tooling**: FastAPI provides automatic OpenAPI docs, request validation, and async support

### Negative

- Adds ~20ms overhead per request (HTTP parsing, auth check, rate limit check)
- In-memory rate limiter and meter do not persist across restarts (acceptable for demo; production would use Redis)
- API key management is basic (set-based, no rotation or scoping)

### Mitigation

- 20ms overhead is negligible compared to retrieval + generation latency
- Architecture supports plugging in Redis-backed rate limiting and metering
- API key validation is behind a dependency injection boundary, easily swappable for JWT or OAuth2
