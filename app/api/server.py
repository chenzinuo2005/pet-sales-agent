"""FastAPI backend bridging the LangGraph pet agent to a Vue3 frontend via SSE."""

from __future__ import annotations

import os
import json
import uuid
import asyncio
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from app.agents.pet_agent import chat_with_agent, get_messages, clear_messages
from app.cnn.inference import _get_model
from app.common.logger import setup_logging, logger

# Ensure logger is configured even in uvicorn reloader subprocess
setup_logging()


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


async def _stream_chat(message: str, image_path: str, thread_id: str):
    """Run the sync chat_with_agent generator in a thread, yielding SSE events."""
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _run():
        try:
            for token in chat_with_agent(message, image_path, thread_id):
                loop.call_soon_threadsafe(q.put_nowait, ("token", token))
            loop.call_soon_threadsafe(q.put_nowait, ("done", thread_id))
        except Exception:
            logger.exception("chat_with_agent 异常")
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
            yield _sse_event("error", json.dumps({"detail": "内部错误，请稍后重试"}))
            break


async def _stream_chat_with_cleanup(
    message: str, image_path: str, thread_id: str, tmp_path: str | None
):
    """Like _stream_chat but deletes *tmp_path* after the stream ends."""
    try:
        async for sse in _stream_chat(message, image_path, thread_id):
            yield sse
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = ""
    thread_id: str = ""


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the CNN model on startup so the first request is fast."""
    logger.info("正在预热 CNN 模型 ...")
    try:
        _get_model()
        logger.info("CNN 模型预热完成")
    except Exception as e:
        logger.warning(f"CNN 模型预热失败（首次请求将重试）: {e}")
    yield


app = FastAPI(title="Pet Agent API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Health check for load balancers and orchestration."""
    return {"status": "ok"}


@app.post("/api/session")
async def create_session():
    """Create a new chat session, returning a fresh thread_id."""
    thread_id = str(uuid.uuid4())
    logger.info(f"新建会话: {thread_id}")
    return {"thread_id": thread_id}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Text-only chat with SSE streaming of agent tokens."""
    logger.info(f"POST /api/chat: thread_id={request.thread_id}")

    return StreamingResponse(
        _stream_chat(request.message, "", request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/upload")
async def chat_upload(
    message: str = Form(""),
    thread_id: str = Form(""),
    image: UploadFile = File(...),
):
    """Chat with image upload, streaming agent tokens via SSE."""
    # Save the uploaded image to a temp file so the agent can read it from disk.
    suffix = os.path.splitext(image.filename or ".jpg")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    contents = await image.read()
    tmp.write(contents)
    tmp.close()

    logger.info(
        f"POST /api/chat/upload: message={message}, "
        f"thread_id={thread_id}, image={image.filename}, "
        f"size={len(contents)} bytes"
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


@app.get("/api/history/{thread_id}")
async def get_history(thread_id: str):
    """Return the conversation history for a given session."""
    logger.info(f"GET /api/history/{thread_id}")
    try:
        messages = await asyncio.to_thread(get_messages, thread_id)
        return messages
    except Exception as e:
        logger.error(f"获取历史消息失败: {e}")
        return JSONResponse(status_code=500, content={"detail": "获取历史消息失败"})


@app.delete("/api/history/{thread_id}")
async def delete_history(thread_id: str):
    """Clear the conversation history for a given session."""
    logger.info(f"DELETE /api/history/{thread_id}")
    try:
        await asyncio.to_thread(clear_messages, thread_id)
        return {"ok": True}
    except Exception as e:
        logger.error(f"清空历史消息失败: {e}")
        return JSONResponse(status_code=500, content={"detail": "清空历史消息失败"})
