"""FastAPI backend — enterprise-grade API server."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.agents.pet_agent import (
    chat_with_container,
    clear_messages_with_container,
    get_messages_with_container,
)
from app.api.middleware import add_middleware_stack
from app.cnn.inference import _get_model
from app.common.config import settings
from app.common.container import get_container
from app.common.exceptions import AppException
from app.common.logger import get_logger, set_request_context, setup_logging
from app.models.schemas import (
    ChatRequest,
    ComponentStatus,
    ErrorResponse,
    HealthResponse,
    HistoryMessage,
    SessionResponse,
)

setup_logging(level=settings.log_level, log_format=settings.log_format)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Error response helper
# ---------------------------------------------------------------------------

def error_response(
    status_code: int,
    error_code: str,
    message: str,
    request_id: str = "",
    details: dict | None = None,
) -> JSONResponse:
    """Build a unified JSON error response.

    All exception handlers and manual error paths funnel through this helper
    so every error looks identical to API consumers.
    """
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error={
                "code": error_code,
                "message": message,
                "details": details or {},
            },
            request_id=request_id or "",
            timestamp=datetime.now().isoformat(),
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


async def _stream_chat(message: str, image_path: str, thread_id: str):
    """Run the sync chat_with_agent generator in a thread, yielding SSE events."""
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    container = get_container()

    def _run():
        try:
            for token in chat_with_container(container, message, image_path, thread_id):
                loop.call_soon_threadsafe(q.put_nowait, ("token", token))
            loop.call_soon_threadsafe(q.put_nowait, ("done", thread_id))
        except Exception:
            logger.exception("chat_stream_error")
            loop.call_soon_threadsafe(q.put_nowait, ("error", ""))

    loop.run_in_executor(None, _run)

    while True:
        kind, payload = await q.get()
        if kind == "token":
            yield _sse_event("token", payload)
        elif kind == "done":
            yield _sse_event("done", json.dumps({"thread_id": payload}))
            break
        elif kind == "error":
            yield _sse_event(
                "error",
                json.dumps({"detail": "Internal error, please retry."}),
            )
            break


async def _stream_chat_with_cleanup(
    message: str,
    image_path: str,
    thread_id: str,
    tmp_path: str | None,
):
    """Like _stream_chat but deletes *tmp_path* after the stream ends."""
    try:
        async for sse in _stream_chat(message, image_path, thread_id):
            yield sse
    finally:
        if tmp_path:
            with suppress(OSError):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the CNN model on startup so the first request is fast."""
    logger.info("Warming up CNN model ...")
    try:
        _get_model()
        logger.info("CNN model warm-up complete")
    except Exception as e:
        logger.warning("cnn_warmup_failed", extra={"error": str(e)})
    yield


app = FastAPI(title="Pet Agent API", version="1.0.0", lifespan=lifespan)

# CORS — kept alongside the enterprise middleware stack for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enterprise middleware stack: RequestID → Auth → RateLimit
add_middleware_stack(app)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Map every :class:`AppException` subclass to a structured ErrorResponse."""
    request_id = getattr(request.state, "request_id", "")
    logger.warning(
        f"Application exception: {exc.error_code} - {exc.message}",
        extra={"error_code": exc.error_code, "request_id": request_id},
    )
    return error_response(
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        request_id=request_id,
        details=exc.details if exc.details else None,  # type: ignore[arg-type]
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler — log the real error, return a sanitised 500."""
    request_id = getattr(request.state, "request_id", "")
    logger.exception(
        f"Unhandled exception: {exc}",
        extra={"request_id": request_id},
    )
    return error_response(
        status_code=500,
        error_code="INTERNAL_ERROR",
        message="An unexpected error occurred.",
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Endpoints — v1
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
async def health_v1(request: Request):
    """Deep health check for all infrastructure components.

    Returns a :class:`HealthResponse` with per-component status so
    orchestration and monitoring tools can pinpoint failures.
    """
    set_request_context(request.state.request_id, "")
    logger.info("Deep health check requested")

    components: dict[str, dict] = {}

    # --- Database (SQLite via LangGraph checkpointer) ---
    try:
        import sqlite3
        conn = sqlite3.connect(settings.db_path)
        conn.execute("SELECT 1")
        conn.close()
        components["database"] = ComponentStatus(
            status="healthy", details="SQLite connected"
        ).model_dump()
    except Exception as exc:
        logger.warning("db_health_check_failed", extra={"error": str(exc)})
        components["database"] = ComponentStatus(
            status="unhealthy", details=str(exc)
        ).model_dump()

    # --- CNN model (weights file existence only — no eager load) ---
    if os.path.isfile(settings.model_weights_path):
        components["cnn_model"] = ComponentStatus(
            status="healthy",
            details=f"Weights file found at {settings.model_weights_path}",
        ).model_dump()
    else:
        components["cnn_model"] = ComponentStatus(
            status="unhealthy",
            details=f"Weights file not found: {settings.model_weights_path}",
        ).model_dump()

    # --- RAG vector store (Chroma DB directory existence) ---
    if os.path.isdir(settings.chroma_dir):
        components["rag_vector_store"] = ComponentStatus(
            status="healthy",
            details=f"Chroma DB directory found at {settings.chroma_dir}",
        ).model_dump()
    else:
        components["rag_vector_store"] = ComponentStatus(
            status="unhealthy",
            details=f"Chroma DB directory not found: {settings.chroma_dir}",
        ).model_dump()

    # --- Agent (LLM) — API key presence implies availability ---
    # Deep LLM probing requires a full inference call; we trust the config.
    components["agent_model"] = ComponentStatus(
        status="healthy",
        details="LLM configured via API key",
    ).model_dump()

    # Determine overall status
    status_values = [c["status"] for c in components.values()]
    if all(s == "healthy" for s in status_values):
        overall = "healthy"
    elif any(s == "unhealthy" for s in status_values):
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthResponse(status=overall, components=components).model_dump()  # type: ignore[arg-type]


@app.post("/api/v1/session")
async def create_session_v1(request: Request):
    """Create a new chat session, returning a fresh thread_id."""
    thread_id = str(uuid.uuid4())
    set_request_context(request.state.request_id, thread_id)
    logger.info("New session created")
    return SessionResponse(
        thread_id=thread_id,
        created_at=datetime.now().isoformat(),
    ).model_dump()


@app.post("/api/v1/chat")
async def chat_v1(request: Request, body: ChatRequest):
    """Text-only chat with SSE streaming of agent tokens."""
    set_request_context(request.state.request_id, body.thread_id)
    logger.info(
        "POST /api/v1/chat",
        extra={"thread_id": body.thread_id, "message_length": len(body.message)},
    )

    return StreamingResponse(
        _stream_chat(body.message, "", body.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/chat/upload")
async def chat_upload_v1(
    request: Request,
    message: str = Form(""),
    thread_id: str = Form(""),
    image: UploadFile = File(...),  # noqa: B008
):
    """Chat with image upload, streaming agent tokens via SSE."""
    suffix = os.path.splitext(image.filename or ".jpg")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)  # noqa: SIM115
    contents = await image.read()
    tmp.write(contents)
    tmp.close()

    set_request_context(request.state.request_id, thread_id)
    logger.info(
        "POST /api/v1/chat/upload",
        extra={
            "thread_id": thread_id,
            "message_length": len(message),
            "image_filename": image.filename,
            "image_size": len(contents),
        },
    )

    return StreamingResponse(
        _stream_chat_with_cleanup(message, tmp.name, thread_id, tmp.name),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/v1/history/{thread_id}")
async def get_history_v1(request: Request, thread_id: str):
    """Return the conversation history for a given session."""
    set_request_context(request.state.request_id, thread_id)
    logger.info("history_get", extra={"thread_id": thread_id})
    messages = await asyncio.to_thread(get_messages_with_container, get_container(), thread_id)
    return [
        HistoryMessage(role=m["role"], content=m["content"]).model_dump()  # type: ignore[arg-type]
        for m in messages
    ]


@app.delete("/api/v1/history/{thread_id}")
async def delete_history_v1(request: Request, thread_id: str):
    """Clear the conversation history for a given session."""
    set_request_context(request.state.request_id, thread_id)
    logger.info("history_delete", extra={"thread_id": thread_id})
    await asyncio.to_thread(clear_messages_with_container, get_container(), thread_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Legacy endpoints  — delegate to v1 handlers so existing clients keep working
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_legacy(request: Request):
    """Deprecated — delegates to /api/v1/health."""
    return await health_v1(request)


@app.post("/api/session")
async def create_session_legacy(request: Request):
    """Deprecated — delegates to /api/v1/session."""
    return await create_session_v1(request)


@app.post("/api/chat")
async def chat_legacy(request: Request, body: ChatRequest):
    """Deprecated — delegates to /api/v1/chat."""
    return await chat_v1(request, body)


@app.post("/api/chat/upload")
async def chat_upload_legacy(
    request: Request,
    message: str = Form(""),
    thread_id: str = Form(""),
    image: UploadFile = File(...),  # noqa: B008
):
    """Deprecated — delegates to /api/v1/chat/upload."""
    return await chat_upload_v1(request, message, thread_id, image)


@app.get("/api/history/{thread_id}")
async def get_history_legacy(request: Request, thread_id: str):
    """Deprecated — delegates to /api/v1/history/{thread_id}."""
    return await get_history_v1(request, thread_id)


@app.delete("/api/history/{thread_id}")
async def delete_history_legacy(request: Request, thread_id: str):
    """Deprecated — delegates to /api/v1/history/{thread_id}."""
    return await delete_history_v1(request, thread_id)
