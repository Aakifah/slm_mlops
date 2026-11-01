"""Ollama client and model registry tests (fully mocked, no live LLM)."""

from __future__ import annotations

import ollama
import pytest

from src.backend.services.ollama_client import (
    ALL_MODELS,
    EMBEDDING_MODEL,
    SLM_BACKBONES,
    SLM_ROLES,
    OllamaClient,
    create_ollama_client,
)
from src.config import Settings


class _FakeClient:
    """A deterministic stand-in for ``ollama.Client``."""

    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list(self) -> dict[str, object]:
        return {"models": [{"name": name} for name in self._models]}

    def generate(self, model: str, prompt: str, format: object = "") -> dict[str, object]:
        return {"response": f"{model}|{prompt}|{format}"}

    def embeddings(self, model: str, prompt: str) -> dict[str, object]:
        return {"embedding": [0.1, 0.2, 0.3]}


def test_registry_constants() -> None:
    """The registry must enumerate the three SLMs plus the embedding model."""
    assert SLM_BACKBONES == ("mistral:7b", "qwen2.5:7b", "llama3.2")
    assert EMBEDDING_MODEL == "nomic-embed-text"
    assert set(ALL_MODELS) == {*SLM_BACKBONES, EMBEDDING_MODEL}
    assert set(SLM_ROLES) == set(SLM_BACKBONES)


def test_list_models_extracts_names() -> None:
    """The wrapper must surface model names from the raw list response."""
    client = OllamaClient(_FakeClient(["mistral:7b", "llama3.2"]))
    assert client.list_models() == ["mistral:7b", "llama3.2"]


def test_is_available() -> None:
    """Availability must reflect the pulled model list."""
    client = OllamaClient(_FakeClient(["mistral:7b"]))
    assert client.is_available("mistral:7b") is True
    assert client.is_available("qwen2.5:7b") is False


def test_generate_without_format() -> None:
    """Generation without a format must omit the format argument."""
    client = OllamaClient(_FakeClient(["mistral:7b"]))
    assert client.generate("mistral:7b", "hello") == "mistral:7b|hello|"


def test_generate_with_format() -> None:
    """Generation with a format must forward it to the raw client."""
    client = OllamaClient(_FakeClient(["mistral:7b"]))
    result = client.generate("mistral:7b", "hello", format="json")
    assert result == "mistral:7b|hello|json"


def test_embed_returns_floats() -> None:
    """Embeddings must be returned as a float list."""
    client = OllamaClient(_FakeClient([]))
    assert client.embed(EMBEDDING_MODEL, "text") == [0.1, 0.2, 0.3]


def test_available_backbones() -> None:
    """The backbone report must reflect availability of every selected model."""
    client = OllamaClient(_FakeClient(["mistral:7b", "nomic-embed-text"]))
    report = client.available_backbones()
    assert report["mistral:7b"] is True
    assert report["qwen2.5:7b"] is False
    assert report["nomic-embed-text"] is True


def test_create_ollama_client_passes_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory must construct the raw client with the configured host."""
    captured: dict[str, object] = {}

    class _FakeFactory:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(ollama, "Client", _FakeFactory)
    settings = Settings(ollama_host="http://example.com")
    create_ollama_client(settings)
    assert captured["host"] == "http://example.com"


def test_create_ollama_client_passes_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory must pass basic-auth credentials when configured."""
    captured: dict[str, object] = {}

    class _FakeFactory:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(ollama, "Client", _FakeFactory)
    settings = Settings(
        ollama_host="http://example.com",
        ollama_user="user",
        ollama_pass="pass",
    )
    create_ollama_client(settings)
    assert captured["auth"] == ("user", "pass")
