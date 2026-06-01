"""Dependency Injection container — lazy-loads and wires all application resources."""
from __future__ import annotations

import os
import threading

from app.common.config import AppConfig, settings
from app.common.exceptions import ModelNotAvailableException, RAGNotInitializedException
from app.common.logger import get_logger

logger = get_logger(__name__)


class AppContainer:
    """Lazy-loading DI container for all application singletons."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or settings
        self._lock = threading.RLock()
        self._model = None
        self._embeddings = None
        self._vector_store = None
        self._checkpointer = None
        self._agent = None
        self._tavily = None
        self._cnn_model = None
        self._cnn_lock = threading.Lock()

    # ------------------------------------------------------------------
    # LLM model
    # ------------------------------------------------------------------

    def get_model(self):
        """Lazy-load the DeepSeek chat model."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import httpx
                    from langchain.chat_models import init_chat_model

                    http_client = httpx.Client(trust_env=False)
                    http_async_client = httpx.AsyncClient(trust_env=False)

                    self._model = init_chat_model(
                        model=self.config.deepseek_model,
                        model_provider="openai",
                        base_url=self.config.deepseek_base_url,
                        api_key=self.config.deepseek_api_key.get_secret_value(),
                        temperature=self.config.deepseek_temperature,
                        max_tokens=self.config.deepseek_max_tokens,
                        http_client=http_client,
                        http_async_client=http_async_client,
                    )
                    logger.info(
                        "llm_model_initialized",
                        extra={"model": self.config.deepseek_model},
                    )
        return self._model

    # ------------------------------------------------------------------
    # Embeddings (DashScope → HuggingFace fallback)
    # ------------------------------------------------------------------

    def get_embeddings(self):
        """Lazy-load embeddings (DashScope → HuggingFace fallback)."""
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    try:
                        from langchain_community.embeddings import (
                            DashScopeEmbeddings,
                        )

                        self._embeddings = DashScopeEmbeddings(
                            model="text-embedding-v4",
                            dashscope_api_key=self.config.dashscope_api_key.get_secret_value(),
                        )
                        logger.info(
                            "embeddings_initialized",
                            extra={"provider": "dashscope"},
                        )
                    except Exception:
                        from langchain_community.embeddings import (
                            HuggingFaceEmbeddings,
                        )

                        self._embeddings = HuggingFaceEmbeddings(
                            model_name="shibing624/text2vec-base-chinese"
                        )
                        logger.info(
                            "embeddings_initialized",
                            extra={"provider": "huggingface"},
                        )
        return self._embeddings

    # ------------------------------------------------------------------
    # Chroma vector store
    # ------------------------------------------------------------------

    def get_vector_store(self):
        """Lazy-load Chroma vector store. Raises RAGNotInitializedException if missing."""
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    import chromadb
                    from langchain_chroma import Chroma

                    if not os.path.exists(self.config.chroma_dir):
                        raise RAGNotInitializedException(
                            message="向量数据库未初始化，请运行 python -m app.main init-rag"
                        )

                    client = chromadb.PersistentClient(
                        path=self.config.chroma_dir
                    )
                    self._vector_store = Chroma(
                        client=client,
                        embedding_function=self.get_embeddings(),
                    )
                    logger.info(
                        "vector_store_loaded",
                        extra={"path": self.config.chroma_dir},
                    )
        return self._vector_store

    # ------------------------------------------------------------------
    # SQLite checkpointer
    # ------------------------------------------------------------------

    def get_checkpointer(self):
        """Lazy-load SQLite checkpointer.

        Uses the constructor directly with a persistent sqlite3.Connection,
        because ``SqliteSaver.from_conn_string`` returns a context manager
        that closes the connection on exit (LangGraph ≥ 3.1.0).
        """
        if self._checkpointer is None:
            with self._lock:
                if self._checkpointer is None:
                    import sqlite3

                    from langgraph.checkpoint.sqlite import SqliteSaver

                    conn = sqlite3.connect(
                        self.config.db_path,
                        check_same_thread=False,
                    )
                    self._checkpointer = SqliteSaver(conn)
                    self._checkpointer.setup()
                    logger.info(
                        "checkpointer_initialized",
                        extra={"db": self.config.db_path},
                    )
        return self._checkpointer

    # ------------------------------------------------------------------
    # Tavily web search
    # ------------------------------------------------------------------

    def get_tavily(self):
        """Lazy-load TavilySearch client.

        Errors during initialization are NOT caught — they propagate to the
        caller so it knows Tavily is unavailable.
        """
        if self._tavily is None:
            with self._lock:
                if self._tavily is None:
                    from langchain_tavily import TavilySearch

                    self._tavily = TavilySearch(
                        max_results=5,
                        topic="general",
                        tavily_api_key=self.config.tavily_api_key.get_secret_value(),
                    )
                    logger.info("tavily_initialized")
        return self._tavily

    # ------------------------------------------------------------------
    # LangGraph agent
    # ------------------------------------------------------------------

    def get_agent(self):
        """Lazy-load the LangGraph agent with parameterized tools."""
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    from langchain.agents import create_agent

                    from app.agents.custom_tools import (
                        create_search_pet_knowledge,
                        create_tavily_web_search,
                    )
                    from app.agents.system_prompt import SYSTEM_PROMPT

                    tools = [
                        create_search_pet_knowledge(self),
                        create_tavily_web_search(self),
                    ]
                    self._agent = create_agent(
                        model=self.get_model(),
                        tools=tools,
                        checkpointer=self.get_checkpointer(),
                        system_prompt=SYSTEM_PROMPT,
                    )
                    logger.info("agent_initialized")
        return self._agent

    # ------------------------------------------------------------------
    # CNN model (double-checked locking)
    # ------------------------------------------------------------------

    def get_cnn_model(self):
        """Lazy-load CNN model with double-checked locking."""
        if self._cnn_model is None:
            with self._cnn_lock:
                if self._cnn_model is None:
                    import torch

                    from app.cnn.model import create_model

                    weights_path = self.config.model_weights_path
                    if not os.path.exists(weights_path):
                        raise ModelNotAvailableException(
                            message=f"CNN 模型文件未找到: {weights_path}"
                        )

                    model = create_model(
                        num_classes=self.config.cnn_num_classes
                    )
                    state_dict = torch.load(
                        weights_path, map_location="cpu", weights_only=True
                    )
                    model.load_state_dict(state_dict)
                    model.eval()
                    self._cnn_model = model
                    logger.info(
                        "cnn_model_loaded", extra={"path": weights_path}
                    )
        return self._cnn_model

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self):
        """Clean up resources (close connections, free memory)."""
        logger.info("container_shutdown_started")
        # SqliteSaver handles its own connection lifecycle via from_conn_string
        self._model = None
        self._embeddings = None
        self._vector_store = None
        self._checkpointer = None
        self._agent = None
        self._cnn_model = None
        logger.info("container_shutdown_complete")


# Module-level singleton -------------------------------------------------
_container: AppContainer | None = None
_container_lock = threading.Lock()


def get_container() -> AppContainer:
    """Return the global AppContainer singleton, creating it on first call."""
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = AppContainer()
    return _container


def reset_container() -> None:
    """Reset the container singleton (useful for testing)."""
    global _container
    with _container_lock:
        if _container is not None:
            _container.shutdown()
        _container = None
