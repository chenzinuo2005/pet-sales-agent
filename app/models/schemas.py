from typing import Literal

from pydantic import BaseModel, Field


class CNNPredictResult(BaseModel):
    """CNN 品种识别结果"""
    breed_en: str
    breed_cn: str
    confidence: float
    top3: list[dict]
    status: Literal["success", "low_confidence", "failed"]


class ErrorResponse(BaseModel):
    """Unified error response format across all API endpoints."""
    error: dict  # {"code": "RATE_LIMIT_EXCEEDED", "message": "请求太频繁"}
    request_id: str | None = None
    timestamp: str  # ISO 8601


class ComponentStatus(BaseModel):
    """Status of a single infrastructure component."""
    status: Literal["healthy", "degraded", "unhealthy"]
    details: str = ""


class HealthResponse(BaseModel):
    """Deep health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str = "1.0.0"
    components: dict  # {"database": ComponentStatus, "cnn_model": ComponentStatus, "rag_vector_store": ComponentStatus, "agent_model": ComponentStatus}


class ChatRequest(BaseModel):
    """Validated chat request."""
    message: str = Field(default="", min_length=0, max_length=4000, description="用户输入文本")
    thread_id: str = Field(default="", min_length=0, max_length=64, description="会话ID，空则自动创建")


class SessionResponse(BaseModel):
    """Session creation response."""
    thread_id: str
    created_at: str | None = None


class HistoryMessage(BaseModel):
    """Single history message."""
    role: Literal["user", "assistant"]
    content: str
