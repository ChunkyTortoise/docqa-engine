"""Tests for DocQA Engine REST API — auth, metering, and endpoints."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from docqa_engine.api import RateLimiter, UsageMeter, create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_KEY = "test-key-1"
VALID_KEY_2 = "test-key-2"
HEADERS = {"X-API-Key": VALID_KEY}


@pytest.fixture()
def demo_app():
    """Create a demo-mode app (no pipeline) with known API keys."""
    return create_app(pipeline=None, api_keys={VALID_KEY, VALID_KEY_2}, rate_limit=5)


@pytest.fixture()
def client(demo_app):
    """Async test client bound to the demo app."""
    transport = ASGITransport(app=demo_app)
    return AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# 1. Authentication Tests
# ---------------------------------------------------------------------------


class TestAuth:
    """Verify API key authentication and rejection."""

    async def test_valid_key_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/stats", headers=HEADERS)
        assert resp.status_code == 200

    async def test_invalid_key_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get("/stats", headers={"X-API-Key": "bad-key"})
        assert resp.status_code == 401
        assert "Invalid or missing API key" in resp.json()["detail"]

    async def test_missing_key_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get("/stats")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 2. Ingest Endpoint Tests
# ---------------------------------------------------------------------------


class TestIngest:
    """Verify document ingestion in demo mode."""

    async def test_ingest_success(self, client: AsyncClient) -> None:
        payload = {"content": "Hello world. This is a test document."}
        resp = await client.post("/ingest", json=payload, headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_id"]
        assert body["chunks"] >= 1
        assert body["message"] == "Document ingested successfully"

    async def test_ingest_with_custom_doc_id(self, client: AsyncClient) -> None:
        payload = {"content": "Custom ID doc.", "doc_id": "my-doc-42"}
        resp = await client.post("/ingest", json=payload, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["doc_id"] == "my-doc-42"

    async def test_ingest_with_metadata(self, client: AsyncClient) -> None:
        payload = {
            "content": "Metadata doc.",
            "metadata": {"author": "test", "year": 2026},
        }
        resp = await client.post("/ingest", json=payload, headers=HEADERS)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. Ask Endpoint Tests
# ---------------------------------------------------------------------------


class TestAsk:
    """Verify question-answering in demo mode."""

    async def test_ask_success(self, client: AsyncClient) -> None:
        payload = {"question": "What is machine learning?"}
        resp = await client.post("/ask", json=payload, headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert "Demo answer for:" in body["answer"]
        assert body["confidence"] == 0.5
        assert body["tokens_used"] > 0

    async def test_ask_with_top_k(self, client: AsyncClient) -> None:
        payload = {"question": "Test question", "top_k": 3}
        resp = await client.post("/ask", json=payload, headers=HEADERS)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 4. Stats Endpoint Tests
# ---------------------------------------------------------------------------


class TestStats:
    """Verify stats reflect accumulated usage."""

    async def test_stats_initial(self, client: AsyncClient) -> None:
        resp = await client.get("/stats", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        # The stats call itself is the first metered request
        assert body["total_documents"] == 0
        assert body["uptime_seconds"] >= 0

    async def test_stats_after_ingest(self, client: AsyncClient) -> None:
        await client.post(
            "/ingest",
            json={"content": "Doc one."},
            headers=HEADERS,
        )
        resp = await client.get("/stats", headers=HEADERS)
        body = resp.json()
        assert body["total_documents"] == 1
        assert body["total_tokens"] > 0


# ---------------------------------------------------------------------------
# 5. Reset Endpoint Tests
# ---------------------------------------------------------------------------


class TestReset:
    """Verify the reset endpoint clears documents."""

    async def test_reset_clears_docs(self, client: AsyncClient) -> None:
        await client.post(
            "/ingest",
            json={"content": "To be deleted."},
            headers=HEADERS,
        )
        resp = await client.delete("/reset", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        stats_resp = await client.get("/stats", headers=HEADERS)
        assert stats_resp.json()["total_documents"] == 0


# ---------------------------------------------------------------------------
# 6. Rate Limiting Tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Verify rate limiter rejects excess requests."""

    async def test_rate_limit_exceeded(self, client: AsyncClient) -> None:
        # App fixture has rate_limit=5
        for _ in range(5):
            resp = await client.get("/stats", headers=HEADERS)
            assert resp.status_code == 200

        # 6th request should be rate-limited
        resp = await client.get("/stats", headers=HEADERS)
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 7. UsageMeter Unit Tests
# ---------------------------------------------------------------------------


class TestUsageMeter:
    """Unit tests for the UsageMeter helper."""

    def test_initial_state(self) -> None:
        meter = UsageMeter()
        assert meter.request_count == 0
        assert meter.token_count == 0

    def test_record_increments_counts(self) -> None:
        meter = UsageMeter()
        meter.record("key-a", tokens=100)
        meter.record("key-a", tokens=50)
        assert meter.request_count == 2
        assert meter.token_count == 150

    def test_get_stats(self) -> None:
        meter = UsageMeter()
        meter.record("key-a", tokens=42)
        stats = meter.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 42
        assert stats["uptime_seconds"] >= 0

    def test_per_key_stats(self) -> None:
        meter = UsageMeter()
        meter.record("key-a", tokens=10)
        meter.record("key-b", tokens=20)
        meter.record("key-a", tokens=30)

        a_stats = meter.get_key_stats("key-a")
        assert a_stats["requests"] == 2
        assert a_stats["tokens"] == 40

        b_stats = meter.get_key_stats("key-b")
        assert b_stats["requests"] == 1
        assert b_stats["tokens"] == 20

    def test_unknown_key_stats(self) -> None:
        meter = UsageMeter()
        stats = meter.get_key_stats("nonexistent")
        assert stats["requests"] == 0
        assert stats["tokens"] == 0


# ---------------------------------------------------------------------------
# 8. RateLimiter Unit Tests
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """Unit tests for the RateLimiter helper."""

    def test_within_limit(self) -> None:
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.check("k") is True
        assert limiter.check("k") is True
        assert limiter.check("k") is True

    def test_exceeds_limit(self) -> None:
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.check("k") is True
        assert limiter.check("k") is True
        assert limiter.check("k") is False

    def test_separate_keys_independent(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.check("a") is True
        assert limiter.check("b") is True
        # Both keys exhausted
        assert limiter.check("a") is False
        assert limiter.check("b") is False
