"""
Centralized application configuration.

All environment-specific values (API keys, file paths, model parameters)
are read from a `.env` file and exposed through the ``settings`` singleton.

Usage::

    from app.common.config import settings
    api_key = settings.deepseek_api_key.get_secret_value()
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application-wide configuration loaded from ``.env`` and environment.

    API keys are stored as ``SecretStr`` so they are never leaked in logs
    or tracebacks.  File-system paths are resolved relative to this source
    file at import time.
    """

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ------------------------------------------------------------------
    # API keys
    # ------------------------------------------------------------------
    deepseek_api_key: SecretStr
    """API key for DeepSeek Chat / Reasoner models.  **Required** in ``.env``."""

    dashscope_api_key: SecretStr
    """API key for Alibaba DashScope (text-embedding-v4).  **Required** in ``.env``."""

    tavily_api_key: SecretStr
    """API key for Tavily web search.  **Required** in ``.env``."""

    api_key: str = ""
    """Optional shared API key for FastAPI authentication.  When empty, auth is
    disabled (development mode)."""

    # ------------------------------------------------------------------
    # DeepSeek model configuration
    # ------------------------------------------------------------------
    deepseek_base_url: str = "https://api.deepseek.com"
    """Base URL for the DeepSeek OpenAI-compatible endpoint."""

    deepseek_model: str = "deepseek-chat"
    """Model name passed to the LangChain ``init_chat_model`` helper.
    Use ``deepseek-chat`` (V3) for fast streaming chat; ``deepseek-reasoner``
    (R1) for tasks that need chain-of-thought reasoning."""

    deepseek_temperature: float = 0.7
    """Sampling temperature for the DeepSeek chat model."""

    deepseek_max_tokens: int = 4096
    """Maximum tokens per response. Prevents runaway generation."""

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = "INFO"
    """Log level for the Python ``logging`` module (DEBUG / INFO / WARNING / ERROR)."""

    log_format: Literal["text", "json"] = "text"
    """Output format for log records.  ``text`` uses a human-readable line format;
    ``json`` emits one JSON object per line for machine ingestion."""

    # ------------------------------------------------------------------
    # File-system paths  (resolved in ``__init__``)
    # ------------------------------------------------------------------
    data_dir: str = ""
    """Directory containing the RAG knowledge-base ``.txt`` files.

    Default: ``<project_root>/data``."""

    chroma_dir: str = ""
    """Directory for the Chroma persistent vector store.

    Default: ``<project_root>/resources/chroma_db``."""

    db_path: str = ""
    """SQLite database path used by LangGraph's ``SqliteSaver`` checkpointer.

    Default: ``<project_root>/resources/pet_agent.db``."""

    model_weights_path: str = ""
    """Path to the serialised CNN model weights (``pet_cnn.pth``).

    Default: ``<project_root>/resources/models/pet_cnn.pth``."""

    confusion_matrix_dir: str = ""
    """Directory where the confusion-matrix PNG is saved after evaluation.

    Default: ``<project_root>/resources/outputs``."""

    # ------------------------------------------------------------------
    # CNN hyper-parameters
    # ------------------------------------------------------------------
    cnn_num_classes: int = 37
    """Number of Oxford-IIIT Pet breeds (37).

    Must match the classification head of the CNN model."""

    # ------------------------------------------------------------------
    # Validation & path resolution
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _resolve_paths(self) -> AppConfig:
        """Convert relative-path defaults to absolute, normalised paths.

        Every path default is expressed relative to *this source file* so
        the project can be relocated without breaking configuration.
        """
        _here = os.path.dirname(os.path.abspath(__file__))
        def _resolve(rel: str) -> str:
            return os.path.normpath(os.path.join(_here, rel))

        if not self.data_dir:
            self.data_dir = _resolve("../../data")
        if not self.chroma_dir:
            self.chroma_dir = _resolve("../../resources/chroma_db")
        if not self.db_path:
            self.db_path = _resolve("../../resources/pet_agent.db")
        if not self.model_weights_path:
            self.model_weights_path = _resolve("../../resources/models/pet_cnn.pth")
        if not self.confusion_matrix_dir:
            self.confusion_matrix_dir = _resolve("../../resources/outputs")

        return self


# -----------------------------------------------------------------------
# Module-level singleton
# -----------------------------------------------------------------------

settings = AppConfig()  # type: ignore[call-arg]
"""Pre-built singleton instance.  Import this wherever config is needed."""


def get_settings() -> AppConfig:
    """Return the module-level ``AppConfig`` singleton.

    This function exists so callers do not depend on a module-level global
    directly, making the singleton pattern easier to change later.
    """
    return settings
