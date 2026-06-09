import pytest
from unittest.mock import patch
from core.llm import query_ollama

def test_llm_fallback_chain_success():
    """Test that query_ollama falls back through the chain when preceding models fail."""
    # We will simulate failures for the primary model, deepseek-v4-pro:cloud, and gemma4:31b-cloud.
    # Qwen3.5:cloud will succeed.
    def mock_query_raw(model, prompt, system="", timeout=300):
        if model in ["primary-model", "deepseek-v4-pro:cloud", "gemma4:31b-cloud"]:
            return "❌ Connection timeout error"
        elif model == "qwen3.5:cloud":
            return "Qwen success response"
        return "❌ Unexpected model"

    # We mock get_ollama_models to return the models in the fallback chain to pass the check.
    mock_available = [
        "deepseek-v4-pro:cloud",
        "gemma4:31b-cloud",
        "qwen3.5:cloud",
        "mistral-nemo:latest"
    ]

    with patch("core.llm._query_ollama_raw", side_effect=mock_query_raw), \
         patch("core.llm.get_ollama_models", return_value=mock_available):
         
        res = query_ollama("primary-model", "hello test")
        assert res == "Qwen success response"

def test_llm_fallback_all_fail():
    """Test that query_ollama returns failure when all models in the fallback chain fail."""
    def mock_query_raw(model, prompt, system="", timeout=300):
        return "❌ Connection timeout error"

    mock_available = [
        "deepseek-v4-pro:cloud",
        "gemma4:31b-cloud",
        "qwen3.5:cloud",
        "mistral-nemo:latest"
    ]

    with patch("core.llm._query_ollama_raw", side_effect=mock_query_raw), \
         patch("core.llm.get_ollama_models", return_value=mock_available):
         
        res = query_ollama("primary-model", "hello test")
        assert "All models in fallback chain failed" in res
        assert "Last error: ❌ Connection timeout error" in res
