"""Shared test fixtures for the pet agent project."""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def project_root():
    """Return the absolute project root path."""
    return Path(__file__).parent.parent.resolve()


@pytest.fixture
def data_dir(project_root):
    """Return the data directory path."""
    return project_root / "data"


@pytest.fixture
def resources_dir(project_root):
    """Return the resources directory path."""
    return project_root / "resources"


@pytest.fixture
def mock_config():
    """Return a mock AppConfig with fake API keys (never real keys)."""
    from pydantic import SecretStr

    from app.common.config import AppConfig

    return AppConfig(
        deepseek_api_key=SecretStr("sk-test-deepseek"),
        dashscope_api_key=SecretStr("sk-test-dashscope"),
        tavily_api_key=SecretStr("tvly-test-tavily"),
        deepseek_model="deepseek-reasoner",
        deepseek_temperature=0.7,
        log_level="DEBUG",
        log_format="text",
        api_key="",
    )


@pytest.fixture
def mock_llm():
    """Return a mock LLM that returns predictable responses."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Test response")
    mock.stream.return_value = [MagicMock(content="Test"), MagicMock(content=" response")]
    return mock


@pytest.fixture
def mock_checkpointer():
    """Return a mock SqliteSaver."""
    mock = MagicMock()
    mock.get.return_value = None
    mock.setup.return_value = None
    return mock


@pytest.fixture
def mock_vector_store():
    """Return a mock Chroma vector store."""
    mock = MagicMock()
    mock.similarity_search.return_value = [
        MagicMock(
            page_content="金毛犬性格温顺，适合家庭饲养",
            metadata={"source": "breed_info.txt"},
        ),
        MagicMock(
            page_content="金毛犬价格在3000-8000元之间",
            metadata={"source": "pricing.txt"},
        ),
    ]
    return mock


@pytest.fixture
def mock_agent(mock_llm):
    """Return a mock LangGraph agent."""
    mock = MagicMock()
    # Simulate streaming: yield 3 chunks then finish
    mock.stream.return_value = [
        (MagicMock(content="你好"), {}),
        (MagicMock(content="！"), {}),
        (MagicMock(content="有什么可以帮你的？"), {}),
    ]
    return mock


@pytest.fixture
def mock_container(mock_llm, mock_checkpointer, mock_vector_store, mock_agent):
    """Return a fully mocked AppContainer with all resources wired."""
    mock = MagicMock()
    mock.get_model.return_value = mock_llm
    mock.get_checkpointer.return_value = mock_checkpointer
    mock.get_vector_store.return_value = mock_vector_store
    mock.get_agent.return_value = mock_agent
    mock.get_tavily.return_value = MagicMock()
    mock.get_cnn_model.return_value = MagicMock()
    mock.get_embeddings.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_container_with_real_paths(mock_config):
    """Return a mock AppContainer that uses real config paths for health checks."""
    with patch("app.common.container.AppContainer", autospec=True) as mock_cls:
        instance = mock_cls.return_value
        instance.config = mock_config
        yield instance


@pytest.fixture
def sample_cnn_result():
    """Return a sample CNNPredictResult for testing breed hints."""
    from app.models.schemas import CNNPredictResult

    return CNNPredictResult(
        breed_en="Persian",
        breed_cn="波斯猫",
        confidence=0.95,
        top3=[
            {"breed_en": "Persian", "breed_cn": "波斯猫", "conf": 0.95},
            {"breed_en": "British Shorthair", "breed_cn": "英国短毛猫", "conf": 0.03},
            {"breed_en": "Ragdoll", "breed_cn": "布偶猫", "conf": 0.02},
        ],
        status="success",
    )


@pytest.fixture
def temp_image_file():
    """Create a temporary JPEG file for testing image upload."""
    from PIL import Image

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)  # noqa: SIM115
    img = Image.new("RGB", (224, 224), color="white")
    img.save(tmp.name, "JPEG")
    tmp.close()
    yield Path(tmp.name)
    os.unlink(tmp.name)
