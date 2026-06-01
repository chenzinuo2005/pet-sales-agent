"""ASGI middleware: authentication, rate limiting, request ID tracking.

All middleware is pure ASGI (not BaseHTTPMiddleware) so StreamingResponse
(SSE) bodies pass through without buffering.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from datetime import datetime

from starlette.requests import Request

from app.common.config import settings
from app.common.logger import clear_request_context, get_logger, set_request_context

logger = get_logger(__name__)


def _error_body(status_code: int, error_code: str, message: str) -> bytes:
    """Build a JSON error response body matching the app's ErrorResponse shape."""
    return json.dumps(
        {
            "error": {"code": error_code, "message": message, "details": {}},
            "request_id": "",
            "timestamp": datetime.now().isoformat(),
        },
        ensure_ascii=False,
    ).encode("utf-8")


async def _send_error(send, status_code: int, error_code: str, message: str) -> None:
    """Send a JSON error response directly (bypasses FastAPI exception handlers)."""
    body = _error_body(status_code, error_code, message)
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [[b"content-type", b"application/json"]],
        }
    )
    await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------------------
# Pure ASGI middlewares
# ---------------------------------------------------------------------------


class RequestIDMiddleware:
    """Inject X-Request-ID header and set request context for logging.

    Pure ASGI — does not buffer streaming response bodies.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Read X-Request-ID from incoming headers
        request_id = str(uuid.uuid4())
        for key, value in scope.get("headers", []):
            if key == b"x-request-id":
                request_id = value.decode("latin-1")
                break

        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id
        set_request_context(request_id, "")

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append([b"x-request-id", request_id.encode()])
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            clear_request_context()


class RequestTimingMiddleware:
    """Log request duration and status code for every API call.

    Pure ASGI — measures time-to-first-byte for streaming responses,
    which is the meaningful metric for SSE endpoints.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        sent_start = False

        async def send_wrapper(message):
            nonlocal sent_start
            if message["type"] == "http.response.start" and not sent_start:
                sent_start = True
                elapsed_ms = (time.perf_counter() - start) * 1000
                headers = list(message.get("headers", []))
                headers.append(
                    [b"x-response-time-ms", str(round(elapsed_ms)).encode()]
                )
                message["headers"] = headers
                logger.info(
                    "request_completed",
                    extra={
                        "method": scope.get("method", ""),
                        "path": scope.get("path", ""),
                        "status_code": message.get("status", 0),
                        "duration_ms": round(elapsed_ms, 2),
                    },
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)


class AuthMiddleware:
    """Optional API-key authentication middleware.

    Pure ASGI — sends 401 directly instead of raising, so streaming responses
    on other paths are not affected.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not settings.api_key:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path.endswith("/health"):
            await self.app(scope, receive, send)
            return

        api_key = ""
        for key, value in scope.get("headers", []):
            if key == b"x-api-key":
                api_key = value.decode("latin-1")
                break

        if not api_key or api_key != settings.api_key:
            await _send_error(send, 401, "AUTHENTICATION_ERROR", "\u65e0\u6548\u6216\u7f3a\u5931\u7684 API Key")
            return

        await self.app(scope, receive, send)


class RateLimitMiddleware:
    """Simple in-memory sliding-window rate limiter.

    Pure ASGI — sends 429 directly instead of raising.
    """

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: dict[str, list[float]] = defaultdict(list)

    def _clean_window(self, ip: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._windows[ip] = [t for t in self._windows[ip] if t > cutoff]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path.endswith("/health"):
            await self.app(scope, receive, send)
            return

        # Get client IP
        client_ip = ""
        for key, value in scope.get("headers", []):
            if key == b"x-forwarded-for":
                client_ip = value.decode("latin-1").split(",")[0].strip()
                break
        if not client_ip:
            client = scope.get("client")
            if client:
                client_ip = client[0]
            else:
                client_ip = "unknown"

        now = time.time()
        self._clean_window(client_ip, now)

        if len(self._windows[client_ip]) >= self.max_requests:
            retry_after = max(
                int(self.window_seconds - (now - self._windows[client_ip][0])), 1
            )
            logger.warning("rate_limit_exceeded", extra={"client_ip": client_ip})
            await _send_error(
                send, 429, "RATE_LIMIT_EXCEEDED", "\u8bf7\u6c42\u592a\u9891\u7e41\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5"
            )
            return

        self._windows[client_ip].append(now)
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def add_middleware_stack(app):
    """Register all middleware on the FastAPI app in correct order.

    FastAPI's ``add_middleware`` inserts at ``user_middleware[0]``, so the
    **first** call produces the **innermost** middleware (closest to the app).

    Desired stack (outermost → innermost):
      RateLimit → Auth → Timing → RequestID → CORS → endpoint
    """
    app.add_middleware(RequestIDMiddleware)       # innermost
    app.add_middleware(RequestTimingMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)        # outermost
    logger.info(
        "middleware_stack_registered",
        extra={"middlewares": ["RateLimit", "Auth", "Timing", "RequestID"]},
    )
