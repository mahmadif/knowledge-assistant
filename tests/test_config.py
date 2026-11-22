"""Unit tests for rag_assistant.config."""

import importlib
from dataclasses import FrozenInstanceError, fields

import pytest

import rag_assistant.config as config

ENV_VARS = [field.name.upper() for field in fields(config.Settings)]


@pytest.fixture
def reload_config(monkeypatch):
    """Reload config after controlling the environment variables it reads."""

    def _reload(**env):
        for name in ENV_VARS:
            monkeypatch.delenv(name, raising=False)

        for name, value in env.items():
            monkeypatch.setenv(name, str(value))

        return importlib.reload(config)

    # Provide the reload function to the test.
    yield _reload

    # Leave the imported module in a predictable default state for other tests.
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    importlib.reload(config)


def test_default_settings(reload_config):
    module = reload_config()

    assert module.settings.llm_provider == "ollama"
    assert module.settings.ollama_model == "qwen2.5:3b"
    assert module.settings.ollama_base_url == "http://localhost:11434"
    assert module.settings.chroma_persist_dir == "./chroma_db"
    assert module.settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert module.settings.chunk_size == 1000
    assert module.settings.chunk_overlap == 150
    assert module.settings.data_dir == "./data"
    assert module.settings.retrieval_k == 3


def test_settings_are_loaded_from_environment(reload_config):
    module = reload_config(
        LLM_PROVIDER="azure",
        OLLAMA_MODEL="custom-model",
        OLLAMA_BASE_URL="http://ollama:11434",
        CHROMA_PERSIST_DIR="/tmp/chroma",
        EMBEDDING_MODEL="custom-embedding",
        CHUNK_SIZE=500,
        CHUNK_OVERLAP=75,
        DATA_DIR="/tmp/documents",
        RETRIEVAL_K=8,
        AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com/",
        AZURE_OPENAI_API_KEY="test-key",
        AZURE_OPENAI_API_VERSION="2025-01-01",
        AZURE_OPENAI_DEPLOYMENT="test-deployment",
        CONFLUENCE_URL="https://example.atlassian.net",
        CONFLUENCE_USERNAME="test-user",
        CONFLUENCE_API_TOKEN="test-token",
        CONFLUENCE_SPACE_KEY="DOCS",
    )

    settings = module.settings

    assert settings.llm_provider == "azure"
    assert settings.ollama_model == "custom-model"
    assert settings.ollama_base_url == "http://ollama:11434"
    assert settings.chroma_persist_dir == "/tmp/chroma"
    assert settings.embedding_model == "custom-embedding"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 75
    assert settings.data_dir == "/tmp/documents"
    assert settings.retrieval_k == 8
    assert settings.azure_endpoint == "https://example.openai.azure.com/"
    assert settings.azure_api_key == "test-key"
    assert settings.azure_api_version == "2025-01-01"
    assert settings.azure_deployment == "test-deployment"
    assert settings.confluence_url == "https://example.atlassian.net"
    assert settings.confluence_username == "test-user"
    assert settings.confluence_api_token == "test-token"
    assert settings.confluence_space_key == "DOCS"


def test_valid_ollama_configuration_passes_validation(reload_config):
    module = reload_config(LLM_PROVIDER="ollama", OLLAMA_MODEL="llama3.2:3b")

    module.settings.validate_llm_configs()  # Should not raise.


def test_empty_ollama_model_fails_validation(reload_config):
    module = reload_config(LLM_PROVIDER="ollama", OLLAMA_MODEL="")

    with pytest.raises(RuntimeError, match="OLLAMA_MODEL"):
        module.settings.validate_llm_configs()


def test_valid_azure_configuration_passes_validation(reload_config):
    module = reload_config(
        LLM_PROVIDER="azure",
        AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com/",
        AZURE_OPENAI_API_KEY="test-key",
        AZURE_OPENAI_DEPLOYMENT="test-deployment",
    )

    module.settings.validate_llm_configs()  # Should not raise.


def test_incomplete_azure_configuration_reports_all_missing_values(reload_config):
    module = reload_config(LLM_PROVIDER="azure")

    with pytest.raises(RuntimeError) as exc_info:
        module.settings.validate_llm_configs()

    message = str(exc_info.value)

    expected_missing_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    assert all(var in message for var in expected_missing_vars)


def test_incomplete_azure_configuration_reports_only_missing_values(reload_config):
    module = reload_config(
        LLM_PROVIDER="azure",
        AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com/",
        AZURE_OPENAI_API_KEY="test-key",
    )

    with pytest.raises(RuntimeError) as exc_info:
        module.settings.validate_llm_configs()

    message = str(exc_info.value)

    assert "AZURE_OPENAI_DEPLOYMENT" in message
    assert "AZURE_OPENAI_ENDPOINT" not in message
    assert "AZURE_OPENAI_API_KEY" not in message


def test_settings_instance_is_immutable(reload_config):
    module = reload_config()

    with pytest.raises(FrozenInstanceError):
        module.settings.chunk_size = 2000
