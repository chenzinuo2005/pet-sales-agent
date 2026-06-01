"""Enterprise-grade structured logging with request-context propagation.

Provides:
- JSON-line and text formatters (selectable at startup).
- A ``contextvars``-based request context that flows through threads and asyncio.
- A ``LogContextAdapter`` that injects ``request_id`` / ``thread_id`` into
  ``extra``, so every log record carries tracing identifiers.
- A ``get_logger()`` factory that returns the adapter over the standard logger.
"""

import contextvars
import json
import logging
import sys
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_FORMAT_TEXT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# Standard ``LogRecord`` attribute names.  Anything else on the record (e.g.
# fields injected via ``extra``) is treated as user-provided metadata.
_LOG_RECORD_STD_ATTRS: frozenset[str] = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text",
    "filename", "funcName", "levelname", "levelno",
    "lineno", "module", "msecs", "message", "msg",
    "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "taskName",
    "thread", "threadName",
})

# ---------------------------------------------------------------------------
# Request context (contextvars-based, works with threads + asyncio)
# ---------------------------------------------------------------------------

class RequestContext:
    """Immutable holder for per-request tracing identifiers."""

    __slots__ = ("request_id", "thread_id")

    def __init__(self, request_id: str, thread_id: str) -> None:
        self.request_id = request_id
        self.thread_id = thread_id


_request_ctx_var: contextvars.ContextVar[RequestContext | None] = (
    contextvars.ContextVar("_request_ctx", default=None)
)


def set_request_context(request_id: str, thread_id: str) -> None:
    """Store *request_id* and *thread_id* in the current context.

    Safe to call from async coroutines and threads — ``contextvars``
    handles isolation automatically.
    """
    _request_ctx_var.set(RequestContext(request_id, thread_id))


def clear_request_context() -> None:
    """Remove the request context from the current scope."""
    _request_ctx_var.set(None)


def _get_request_context() -> RequestContext | None:
    """Return the active :class:`RequestContext` or *None*."""
    return _request_ctx_var.get(None)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Encode a :class:`~logging.LogRecord` as a single JSON line.

    The output always contains ``timestamp``, ``level``, ``logger``, and
    ``message``.  When a request context is active, ``request_id`` and
    ``thread_id`` are added.  Any extra kwargs passed through the adapter
    are also serialised.
    """

    def format(self, record: logging.LogRecord) -> str:
        # ISO-8601 UTC with milliseconds
        ts = (
            datetime.fromtimestamp(record.created, tz=UTC)
            .strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{record.msecs:03.0f}Z"
        )

        payload: dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        ctx = _get_request_context()
        if ctx is not None:
            payload["request_id"] = ctx.request_id
            payload["thread_id"] = ctx.thread_id

        # Merge any user-supplied extra fields (injected by LoggerAdapter /
        # explicit `extra={...}` kwargs).
        for key, value in record.__dict__.items():
            if key not in _LOG_RECORD_STD_ATTRS and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Log context adapter
# ---------------------------------------------------------------------------

class LogContextAdapter(logging.LoggerAdapter):
    """A :class:`~logging.LoggerAdapter` that injects ``request_id`` and
    ``thread_id`` from the active request context into every log call.

    Usage::

        log = get_logger(__name__)
        log.info("user logged in")
    """

    def __init__(
        self,
        logger: logging.Logger,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(logger, extra or {})

    def process(
        self,
        msg: Any,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[Any, MutableMapping[str, Any]]:
        ctx = _get_request_context()
        extra = dict(kwargs.get("extra", {}))
        if ctx is not None:
            extra.setdefault("request_id", ctx.request_id)
            extra.setdefault("thread_id", ctx.thread_id)
        for key, value in (self.extra or {}).items():
            extra.setdefault(key, value)
        kwargs["extra"] = extra
        return msg, kwargs


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

def get_logger(name: str) -> LogContextAdapter:
    """Return a :class:`LogContextAdapter` wrapping ``logging.getLogger(name)``.

    The returned object has the usual methods (``debug``, ``info``, ``warning``,
    ``error``, ``exception``, ``critical``) and automatically enriches records
    with ``request_id`` and ``thread_id`` when a request context is active.
    """
    return LogContextAdapter(logging.getLogger(name))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    log_format: Literal["text", "json"] = "text",
) -> None:
    """Configure the root logger.

    Parameters
    ----------
    level:
        Logging threshold — one of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, ``"CRITICAL"``.
    log_format:
        - ``"text"`` — human-readable format (default, great for dev).
        - ``"json"`` — machine-readable JSON lines (great for prod / log
          aggregators).
    """
    root = logging.getLogger()

    # Clear existing handlers so repeated calls are idempotent.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)  # let the logger's own level gate messages

    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(LOG_FORMAT_TEXT))

    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper()))


# ---------------------------------------------------------------------------
# Module-level convenience logger (backward-compatible)
# ---------------------------------------------------------------------------

logger: LogContextAdapter = get_logger("pet_agent")
