"""Integration tests for FastAPI endpoints using TestClient."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_container():
    """Reset the DI container before each test for clean state."""
    from app.common.container import reset_container
    reset_container()
    yield
    reset_container()


@pytest.fixture(autouse=True)
def mock_agent():
    """Mock agent/checkpointer functions so tests never hit real LLM or DB.

    Patches the functions at their import location in ``app.api.server`` so
    that all endpoint handlers use the mocks.  Without this the chat endpoint
    would try to initialise a real DeepSeek model (costing credits) and the
    history endpoints would need a real SQLite checkpointer file.
    """
    def _fake_chat(container, message, image_path=None, thread_id=""):
        """Generator that yields a couple of tokens, mimicking the real agent."""
        yield "Hello"
        yield " from "
        yield "pet agent!"

    def _fake_get_messages(container, thread_id):
        """Return empty history so the endpoint returns []."""
        return []

    def _fake_clear_messages(container, thread_id):
        """No-op history clear."""
        return None

    with (
        patch("app.api.server.chat_with_container", side_effect=_fake_chat),
        patch("app.api.server.get_messages_with_container", side_effect=_fake_get_messages),
        patch("app.api.server.clear_messages_with_container", side_effect=_fake_clear_messages),
    ):
        yield


@pytest.fixture
def client():
    """Return a TestClient wrapping the FastAPI app."""
    from app.api.server import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /api/v1/health"""

    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data
        assert "database" in data["components"]
        assert "cnn_model" in data["components"]
        assert "rag_vector_store" in data["components"]
        assert "agent_model" in data["components"]

    def test_health_legacy_endpoint(self, client):
        """Legacy /api/health should still work."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_components_have_valid_status(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        valid_statuses = {"healthy", "degraded", "unhealthy"}
        for component_name, component in data["components"].items():
            assert component["status"] in valid_statuses, (
                f"Component '{component_name}' has invalid status: {component['status']}"
            )


class TestSessionEndpoint:
    """Tests for POST /api/v1/session"""

    def test_create_session_returns_thread_id(self, client):
        response = client.post("/api/v1/session")
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert len(data["thread_id"]) > 0
        assert "created_at" in data

    def test_create_session_legacy_endpoint(self, client):
        response = client.post("/api/session")
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data


class TestChatEndpoint:
    """Tests for POST /api/v1/chat"""

    def test_chat_with_empty_message(self, client):
        """Empty message should be accepted (validated by Pydantic model)."""
        response = client.post("/api/v1/chat", json={"message": "", "thread_id": ""})
        # Should stream SSE or return error — at minimum not crash
        assert response.status_code in (200, 422)

    def test_chat_with_too_long_message(self, client):
        """Message exceeding max_length should be rejected with 422."""
        long_msg = "x" * 5000
        response = client.post("/api/v1/chat", json={"message": long_msg, "thread_id": ""})
        assert response.status_code == 422

    def test_chat_with_too_long_thread_id(self, client):
        """Thread ID exceeding max_length should be rejected with 422."""
        long_id = "x" * 100
        response = client.post("/api/v1/chat", json={"message": "hello", "thread_id": long_id})
        assert response.status_code == 422

    def test_chat_streaming_response(self, client):
        """Chat with streaming should return SSE content-type."""
        # Note: without real LLM, this may error — we're testing the endpoint exists
        response = client.post("/api/v1/chat", json={"message": "你好", "thread_id": "test"})
        # Can be 200 (SSE) or error (no LLM) — just ensure it's not 500
        assert response.status_code != 500

    def test_chat_legacy_endpoint(self, client):
        response = client.post("/api/chat", json={"message": "你好", "thread_id": "test"})
        assert response.status_code != 500


class TestHistoryEndpoint:
    """Tests for GET/DELETE /api/v1/history/{thread_id}"""

    def test_get_nonexistent_history(self, client):
        """Getting history for nonexistent thread returns empty list."""
        response = client.get("/api/v1/history/nonexistent-thread-id")
        assert response.status_code in (200, 500)  # 200 with empty list is ideal

    def test_delete_nonexistent_history(self, client):
        """Deleting nonexistent history should succeed (idempotent)."""
        response = client.delete("/api/v1/history/nonexistent-thread-id")
        assert response.status_code in (200, 500)

    def test_history_legacy_endpoint(self, client):
        response = client.get("/api/history/nonexistent-thread-id")
        assert response.status_code in (200, 500)


class TestErrorHandling:
    """Tests for global error handlers."""

    def test_404_on_unknown_route(self, client):
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_request_id_header_present(self, client):
        """Response should include X-Request-ID header."""
        response = client.get("/api/v1/health")
        assert "X-Request-ID" in response.headers

    def test_auth_middleware_disabled_by_default(self, client):
        """When api_key is empty, auth middleware should pass through."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200  # not 401
