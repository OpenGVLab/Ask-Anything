"""Integration tests for MiniMax LLM provider.

These tests hit the real MiniMax API and are skipped when MINIMAX_API_KEY
is not set.
"""

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub old langchain imports that may not exist in newer versions
_STUBS = {}
for _mod_name in [
    "langchain.agents.initialize",
    "langchain.agents.tools",
    "langchain.chains.conversation.memory",
]:
    if _mod_name not in sys.modules:
        _stub = ModuleType(_mod_name)
        sys.modules[_mod_name] = _stub
        _STUBS[_mod_name] = _stub

if "langchain.agents.tools" in _STUBS:
    _STUBS["langchain.agents.tools"].Tool = MagicMock
if "langchain.agents.initialize" in _STUBS:
    _STUBS["langchain.agents.initialize"].initialize_agent = MagicMock
if "langchain.chains.conversation.memory" in _STUBS:
    _STUBS["langchain.chains.conversation.memory"].ConversationBufferMemory = MagicMock

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
skip_no_key = pytest.mark.skipif(
    not MINIMAX_API_KEY, reason="MINIMAX_API_KEY not set"
)


@skip_no_key
class TestMiniMaxIntegration:
    def test_create_llm_minimax(self):
        from chatbot import create_llm

        llm = create_llm("minimax", MINIMAX_API_KEY)
        assert llm is not None

    def test_minimax_chat_completion(self):
        from chatbot import create_llm

        llm = create_llm("minimax", MINIMAX_API_KEY, temperature=0.5)
        # ChatOpenAI supports predict() / invoke()
        response = llm.predict("Say hello in one word.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_minimax_m27_highspeed(self):
        from chatbot import create_llm

        llm = create_llm(
            "minimax", MINIMAX_API_KEY,
            model_name="MiniMax-M2.7-highspeed",
            temperature=0.5,
        )
        response = llm.predict("What is 2+2? Answer with just the number.")
        assert isinstance(response, str)
        assert "4" in response
