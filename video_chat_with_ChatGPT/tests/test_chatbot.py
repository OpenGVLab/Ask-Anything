"""Unit tests for chatbot.py – LLM provider support."""

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Ensure langchain sub-modules required by chatbot.py are importable
# (the project pins langchain==0.0.101 whose import paths differ from the
# version installed on CI).
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

# Provide stub symbols so `from langchain.agents.tools import Tool` works
if "langchain.agents.tools" in _STUBS:
    _STUBS["langchain.agents.tools"].Tool = MagicMock
if "langchain.agents.initialize" in _STUBS:
    _STUBS["langchain.agents.initialize"].initialize_agent = MagicMock
if "langchain.chains.conversation.memory" in _STUBS:
    _STUBS["langchain.chains.conversation.memory"].ConversationBufferMemory = MagicMock

from chatbot import LLM_PROVIDERS, ConversationBot, create_llm, cut_dialogue_history


# ---------------------------------------------------------------------------
# cut_dialogue_history
# ---------------------------------------------------------------------------


class TestCutDialogueHistory:
    def test_empty_history(self):
        assert cut_dialogue_history("") == ""

    def test_none_history(self):
        assert cut_dialogue_history(None) is None

    def test_short_history(self):
        short = "hello world"
        assert cut_dialogue_history(short, keep_last_n_words=100) == short

    def test_long_history_trimmed(self):
        history = "\n".join([f"line {i}" for i in range(100)])
        result = cut_dialogue_history(history, keep_last_n_words=10)
        assert len(result.split()) <= 20  # trimmed


# ---------------------------------------------------------------------------
# LLM_PROVIDERS registry
# ---------------------------------------------------------------------------


class TestLLMProviders:
    def test_openai_registered(self):
        assert "openai" in LLM_PROVIDERS

    def test_minimax_registered(self):
        assert "minimax" in LLM_PROVIDERS

    def test_minimax_api_base(self):
        assert LLM_PROVIDERS["minimax"]["api_base"] == "https://api.minimax.io/v1"

    def test_minimax_default_model(self):
        assert LLM_PROVIDERS["minimax"]["default_model"] == "MiniMax-M2.7"

    def test_openai_default_model(self):
        assert LLM_PROVIDERS["openai"]["default_model"] == "gpt-4"


# ---------------------------------------------------------------------------
# create_llm
# ---------------------------------------------------------------------------


class TestCreateLLM:
    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_llm("nonexistent", "key123")

    @patch("chatbot.OpenAI")
    def test_openai_provider(self, mock_openai):
        mock_openai.return_value = MagicMock()
        llm = create_llm("openai", "sk-test123")
        mock_openai.assert_called_once_with(
            temperature=0, openai_api_key="sk-test123", model_name="gpt-4"
        )

    @patch("chatbot.ChatOpenAI")
    def test_minimax_provider(self, mock_chat):
        mock_chat.return_value = MagicMock()
        llm = create_llm("minimax", "mm-key-abc")
        mock_chat.assert_called_once_with(
            model_name="MiniMax-M2.7",
            openai_api_key="mm-key-abc",
            openai_api_base="https://api.minimax.io/v1",
            temperature=0.01,
        )

    @patch("chatbot.ChatOpenAI")
    def test_minimax_temperature_clamped_above_zero(self, mock_chat):
        mock_chat.return_value = MagicMock()
        create_llm("minimax", "key", temperature=0)
        _, kwargs = mock_chat.call_args
        assert kwargs["temperature"] >= 0.01

    @patch("chatbot.ChatOpenAI")
    def test_minimax_temperature_clamped_at_max(self, mock_chat):
        mock_chat.return_value = MagicMock()
        create_llm("minimax", "key", temperature=2.0)
        _, kwargs = mock_chat.call_args
        assert kwargs["temperature"] <= 1.0

    @patch("chatbot.ChatOpenAI")
    def test_minimax_custom_model(self, mock_chat):
        mock_chat.return_value = MagicMock()
        create_llm("minimax", "key", model_name="MiniMax-M2.7-highspeed")
        _, kwargs = mock_chat.call_args
        assert kwargs["model_name"] == "MiniMax-M2.7-highspeed"

    @patch("chatbot.OpenAI")
    def test_openai_custom_model(self, mock_openai):
        mock_openai.return_value = MagicMock()
        create_llm("openai", "sk-test", model_name="gpt-3.5-turbo")
        _, kwargs = mock_openai.call_args
        assert kwargs["model_name"] == "gpt-3.5-turbo"

    def test_case_insensitive_provider(self):
        with patch("chatbot.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            create_llm("MiniMax", "key")
            mock_chat.assert_called_once()


# ---------------------------------------------------------------------------
# ConversationBot.init_agent
# ---------------------------------------------------------------------------


class TestConversationBotInitAgent:
    @patch("chatbot.initialize_agent")
    @patch("chatbot.create_llm")
    def test_init_agent_openai(self, mock_create, mock_init_agent):
        mock_create.return_value = MagicMock()
        mock_init_agent.return_value = MagicMock()
        bot = ConversationBot()
        result = bot.init_agent("sk-test", "cap", "dense", "vid", "tags", [], "openai")
        mock_create.assert_called_once_with(provider="openai", api_key="sk-test")
        assert len(result) == 4

    @patch("chatbot.initialize_agent")
    @patch("chatbot.create_llm")
    def test_init_agent_minimax(self, mock_create, mock_init_agent):
        mock_create.return_value = MagicMock()
        mock_init_agent.return_value = MagicMock()
        bot = ConversationBot()
        result = bot.init_agent("mm-key", "cap", "dense", "vid", "tags", [], "minimax")
        mock_create.assert_called_once_with(provider="minimax", api_key="mm-key")

    @patch("chatbot.initialize_agent")
    @patch("chatbot.create_llm")
    def test_init_agent_empty_key_rejected(self, mock_create, mock_init_agent):
        bot = ConversationBot()
        result = bot.init_agent("", "cap", "dense", "vid", "tags", [], "openai")
        mock_create.assert_not_called()
        assert "API key" in result[3]

    @patch("chatbot.initialize_agent")
    @patch("chatbot.create_llm")
    def test_init_agent_openai_bad_key_rejected(self, mock_create, mock_init_agent):
        bot = ConversationBot()
        result = bot.init_agent("bad-key", "cap", "dense", "vid", "tags", [], "openai")
        mock_create.assert_not_called()
        assert "sk-" in result[3]

    @patch("chatbot.initialize_agent")
    @patch("chatbot.create_llm")
    def test_init_agent_minimax_no_sk_required(self, mock_create, mock_init_agent):
        mock_create.return_value = MagicMock()
        mock_init_agent.return_value = MagicMock()
        bot = ConversationBot()
        result = bot.init_agent("mm-any-key", "c", "d", "v", "t", [], "minimax")
        mock_create.assert_called_once()

    @patch.dict(os.environ, {"LLM_PROVIDER": "minimax"})
    @patch("chatbot.initialize_agent")
    @patch("chatbot.create_llm")
    def test_init_agent_env_var_provider(self, mock_create, mock_init_agent):
        mock_create.return_value = MagicMock()
        mock_init_agent.return_value = MagicMock()
        bot = ConversationBot()
        # provider=None should fallback to env var
        result = bot.init_agent("mm-key", "c", "d", "v", "t", [], None)
        mock_create.assert_called_once_with(provider="minimax", api_key="mm-key")

    @patch("chatbot.initialize_agent")
    @patch("chatbot.create_llm")
    def test_init_agent_state_appended(self, mock_create, mock_init_agent):
        mock_create.return_value = MagicMock()
        mock_init_agent.return_value = MagicMock()
        bot = ConversationBot()
        result = bot.init_agent("sk-test", "c", "d", "v", "t", [], "openai")
        # state should contain the welcome message
        state = result[1]
        assert len(state) == 1
        assert "upload a video" in state[0][0].lower()


# ---------------------------------------------------------------------------
# ConversationBot.run_text
# ---------------------------------------------------------------------------


class TestConversationBotRunText:
    @patch("chatbot.initialize_agent")
    @patch("chatbot.create_llm")
    def test_run_text(self, mock_create, mock_init_agent):
        mock_create.return_value = MagicMock()
        mock_agent = MagicMock()
        mock_agent.return_value = {"output": "Test response"}
        mock_agent.memory = MagicMock()
        mock_agent.memory.buffer = ""
        mock_init_agent.return_value = mock_agent
        bot = ConversationBot()
        bot.init_agent("sk-test", "c", "d", "v", "t", [], "openai")
        result_state, _ = bot.run_text("hello", [])
        assert result_state[-1][1] == "Test response"
