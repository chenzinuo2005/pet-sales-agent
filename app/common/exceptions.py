"""
Custom exception hierarchy for the Pet Agent application.

Every exception carries an HTTP ``status_code`` and a machine-readable
``error_code`` so FastAPI exception handlers can produce consistent JSON
error responses without ad-hoc ``JSONResponse`` calls.
"""

from __future__ import annotations

from typing import Any


class AppException(Exception):  # noqa: N818
    """Base exception for all application-level errors.

    Attributes:
        status_code: HTTP status code to return to the client.
        error_code:  Machine-readable error string (e.g. ``"MODEL_NOT_AVAILABLE"``).
        message:     Human-readable description of what went wrong.
        details:     Optional free-form payload with extra context (e.g. validation
                     errors, stack traces in debug mode).
    """

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str | None = None,
        details: Any = None,
    ) -> None:
        self.message = message or self.__doc__ or "An unexpected error occurred."
        self.details = details
        super().__init__(self.message)


class ModelNotAvailableException(AppException):
    """The CNN model or LLM backend is not reachable."""

    status_code = 503
    error_code = "MODEL_NOT_AVAILABLE"


class RAGNotInitializedException(AppException):
    """The RAG vector store has not been built yet — run ``init-rag`` first."""

    status_code = 503
    error_code = "RAG_NOT_INITIALIZED"


class InvalidInputException(AppException):
    """The client request is malformed or contains invalid parameters."""

    status_code = 400
    error_code = "INVALID_INPUT"


class RateLimitExceededException(AppException):
    """The client has exceeded the per-minute or per-hour rate limit."""

    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"


class AuthenticationException(AppException):
    """The request is missing a valid API key or token."""

    status_code = 401
    error_code = "UNAUTHORIZED"


# Convenience re-export
__all__ = [
    "AppException",
    "AuthenticationException",
    "InvalidInputException",
    "ModelNotAvailableException",
    "RAGNotInitializedException",
    "RateLimitExceededException",
]
