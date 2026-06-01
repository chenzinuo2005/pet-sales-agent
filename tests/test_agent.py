"""Tests for Agent core, tools, and system prompt."""
import sqlite3
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSystemPrompt:
    def test_system_prompt_is_string(self):
        from app.agents.system_prompt import SYSTEM_PROMPT

        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 50

    def test_system_prompt_contains_keywords(self):
        from app.agents.system_prompt import SYSTEM_PROMPT

        assert "小宠" in SYSTEM_PROMPT
        assert "萌宠之家" in SYSTEM_PROMPT


class TestTools:
    def test_search_pet_knowledge_factory(self):
        from unittest.mock import MagicMock
        from app.agents.custom_tools import create_search_pet_knowledge
        from langchain_core.tools import BaseTool

        container = MagicMock()
        tool = create_search_pet_knowledge(container)
        assert isinstance(tool, BaseTool)

    def test_tavily_web_search_factory(self):
        from unittest.mock import MagicMock
        from app.agents.custom_tools import create_tavily_web_search
        from langchain_core.tools import BaseTool

        container = MagicMock()
        tool = create_tavily_web_search(container)
        assert isinstance(tool, BaseTool)


class TestBreedHint:
    def test_success_hint(self):
        from app.agents.pet_agent import _build_breed_hint
        from app.models.schemas import CNNPredictResult

        result = CNNPredictResult(
            breed_en="Persian", breed_cn="波斯猫",
            confidence=0.92, top3=[], status="success",
        )
        hint = _build_breed_hint(result)
        assert "波斯猫" in hint
        assert "92" in hint

    def test_low_confidence_hint(self):
        from app.agents.pet_agent import _build_breed_hint
        from app.models.schemas import CNNPredictResult

        result = CNNPredictResult(
            breed_en="Beagle", breed_cn="比格犬",
            confidence=0.65, top3=[], status="low_confidence",
        )
        hint = _build_breed_hint(result)
        assert "可能" in hint
        assert "比格犬" in hint

    def test_failed_hint(self):
        from app.agents.pet_agent import _build_breed_hint
        from app.models.schemas import CNNPredictResult

        result = CNNPredictResult(
            breed_en="", breed_cn="",
            confidence=0.0, top3=[], status="failed",
        )
        hint = _build_breed_hint(result)
        assert "置信度较低" in hint or "仅供参考" in hint


class TestCheckpointer:
    @pytest.fixture
    def temp_db(self):
        import tempfile
        from langgraph.checkpoint.sqlite import SqliteSaver

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name, check_same_thread=False)
        cp = SqliteSaver(conn)
        cp.setup()
        yield cp, tmp.name
        conn.close()
        os.unlink(tmp.name)

    def test_checkpointer_setup(self, temp_db):
        cp, _ = temp_db
        # Should not raise
        assert cp is not None

    def test_checkpointer_empty_get(self, temp_db):
        cp, _ = temp_db
        result = cp.get({"configurable": {"thread_id": "nonexistent"}})
        assert result is None or result == {}
