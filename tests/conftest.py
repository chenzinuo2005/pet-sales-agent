"""Shared pytest fixtures for the pet-agent test suite."""
import os
import pytest


@pytest.fixture(scope="session")
def project_root() -> str:
    """Absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def data_dir(project_root: str) -> str:
    """Path to the knowledge-base data directory."""
    return os.path.join(project_root, "data")


@pytest.fixture(scope="session")
def resources_dir(project_root: str) -> str:
    """Path to the resources directory (models, db, outputs)."""
    return os.path.join(project_root, "resources")
