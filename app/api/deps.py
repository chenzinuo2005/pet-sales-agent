"""FastAPI dependency functions — inject application resources into endpoints."""
from __future__ import annotations

from fastapi import Request

from app.common.config import AppConfig, settings
from app.common.container import AppContainer, get_container
from app.common.logger import get_logger

logger = get_logger(__name__)


def get_settings() -> AppConfig:
    """Dependency: return the application configuration singleton."""
    return settings


def get_app_container() -> AppContainer:
    """Dependency: return the application DI container singleton."""
    return get_container()


# Convenience shortcuts for commonly injected resources

def get_agent():
    """Dependency: return the LangGraph agent (lazy-loaded on first access)."""
    return get_container().get_agent()


def get_cnn_model():
    """Dependency: return the CNN model (lazy-loaded on first access)."""
    return get_container().get_cnn_model()


def get_vector_store():
    """Dependency: return the Chroma vector store (lazy-loaded on first access)."""
    return get_container().get_vector_store()


def get_checkpointer():
    """Dependency: return the SQLite checkpointer (lazy-loaded on first access)."""
    return get_container().get_checkpointer()


def get_request_id(request: Request) -> str:
    """Dependency: extract X-Request-ID from the incoming request."""
    return getattr(request.state, "request_id", "")
