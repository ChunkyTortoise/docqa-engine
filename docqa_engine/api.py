"""REST API wrapper for DocQA Engine with auth and metering."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_id: str | None = None


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    message: str = "Document ingested successfully"


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    tokens_used: int = 0


class StatsResponse(BaseModel):
    total_documents: int = 0
    total_queries: int = 0
    total_tokens: int = 0
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Metering
# ---------------------------------------------------------------------------


class UsageMeter:
    """Track API usage metrics per key and globally."""

    def __init__(self) -> None:
        self.request_count: int = 0
        self.token_count: int = 0
        self.start_time: float = time.time()
        self._per_key: dict[str, dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "tokens": 0},
        )

    def record(self, api_key: str, tokens: int = 0) -> None:
        self.request_count += 1
        self.token_count += tokens
        self._per_key[api_key]["requests"] += 1
        self._per_key[api_key]["tokens"] += tokens

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_requests": self.request_count,
            "total_tokens": self.token_count,
            "uptime_seconds": time.time() - self.start_time,
        }

    def get_key_stats(self, api_key: str) -> dict[str, int]:
        return dict(self._per_key.get(api_key, {"requests": 0, "tokens": 0}))


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > window_start]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------


def create_app(
    pipeline: Any = None,
    api_keys: set[str] | None = None,
    rate_limit: int = 100,
) -> FastAPI:
    """Create a FastAPI application wrapping a :class:`DocQAPipeline`.

    Parameters
    ----------
    pipeline:
        An optional ``DocQAPipeline`` instance.  When *None* the API runs in
        demo mode and returns synthetic answers.
    api_keys:
        Set of valid API keys.  Defaults to ``{"demo-key"}``.
    rate_limit:
        Maximum requests per key per 60-second window.
    """

    app = FastAPI(title="DocQA Engine API", version="1.0.0")
    _meter = UsageMeter()
    _limiter = RateLimiter(max_requests=rate_limit)
    _valid_keys = api_keys or {"demo-key"}
    _docs: dict[str, dict[str, Any]] = {}

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def verify_api_key(
        api_key: str | None = Depends(api_key_header),
    ) -> str:
        if not api_key or api_key not in _valid_keys:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        if not _limiter.check(api_key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return api_key

    # ── Endpoints ─────────────────────────────────────────────────────────

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        req: IngestRequest,
        api_key: str = Depends(verify_api_key),
    ) -> IngestResponse:
        doc_id = req.doc_id or hashlib.sha256(req.content[:200].encode()).hexdigest()[:12]
        chunks = 0

        if pipeline is not None:
            try:
                result = pipeline.ingest_text(req.content, filename=f"{doc_id}.txt")
                chunks = len(result.chunks)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
        else:
            chunks = max(1, len(req.content) // 500)

        _docs[doc_id] = {"chunks": chunks, **req.metadata}
        _meter.record(api_key, tokens=len(req.content))
        return IngestResponse(doc_id=doc_id, chunks=chunks)

    @app.post("/ask", response_model=AskResponse)
    async def ask(
        req: AskRequest,
        api_key: str = Depends(verify_api_key),
    ) -> AskResponse:
        if pipeline is not None:
            try:
                result = await pipeline.ask(req.question, top_k=req.top_k)
                answer = result.answer_text
                sources = [
                    {
                        "chunk_id": c.chunk_id,
                        "source": c.source,
                        "snippet": c.content_snippet,
                        "relevance": c.relevance_score,
                    }
                    for c in result.citations
                ]
                confidence = min(
                    1.0,
                    max(c.relevance_score for c in result.citations) if result.citations else 0.0,
                )
                tokens = result.tokens_used
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
        else:
            answer = f"Demo answer for: {req.question}"
            sources = []
            confidence = 0.5
            tokens = len(req.question) + len(answer)

        _meter.record(api_key, tokens=tokens)
        return AskResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            tokens_used=tokens,
        )

    @app.get("/stats", response_model=StatsResponse)
    async def stats(
        api_key: str = Depends(verify_api_key),
    ) -> StatsResponse:
        meter_stats = _meter.get_stats()
        return StatsResponse(
            total_documents=len(_docs),
            total_queries=meter_stats["total_requests"],
            total_tokens=meter_stats["total_tokens"],
            uptime_seconds=meter_stats["uptime_seconds"],
        )

    @app.delete("/reset")
    async def reset(
        api_key: str = Depends(verify_api_key),
    ) -> dict[str, str]:
        _docs.clear()
        if pipeline is not None:
            pipeline.reset()
        return {"message": "All documents cleared", "status": "ok"}

    # Store references for testing / introspection
    app.state.meter = _meter
    app.state.limiter = _limiter

    return app
