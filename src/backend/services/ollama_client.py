"""Thin Ollama client for the selected Small Language Model backbones.

Milestone 2 selects three open-weight SLMs and one embedding model. This module
registers those selections and wraps the ``ollama`` package behind a narrow,
testable interface so later milestones can generate text and embeddings without
touching the transport layer directly.
"""

from __future__ import annotations

from typing import Any, Mapping

import ollama

from src.config import Settings

SLM_BACKBONES: tuple[str, ...] = ("mistral:7b", "qwen2.5:7b", "llama3.2")
EMBEDDING_MODEL: str = "nomic-embed-text"

SLM_ROLES: dict[str, str] = {
    "mistral:7b": "Baseline instruction-tuned SLM",
    "qwen2.5:7b": "High SQL-accuracy backbone",
    "llama3.2": "Low-latency backbone",
}

ALL_MODELS: tuple[str, ...] = (*SLM_BACKBONES, EMBEDDING_MODEL)


class OllamaClient:
    """A narrow wrapper around the raw Ollama client."""

    def __init__(self, client: Any) -> None:
        """Instantiate the wrapper.

        Args:
            client: The underlying Ollama client or a test double.
        """
        self._client = client

    def list_models(self) -> list[str]:
        """Return the names of every model currently pulled on the server.

        Returns:
            The list of available model names.
        """
        response: Mapping[str, Any] = self._client.list()
        return [str(model["name"]) for model in response["models"]]

    def is_available(self, model: str) -> bool:
        """Return whether a model is pulled on the server.

        Args:
            model: The model name to check.

        Returns:
            ``True`` when the model is present.
        """
        return model in self.list_models()

    def generate(
        self,
        model: str,
        prompt: str,
        format: str | dict[str, object] | None = None,
    ) -> str:
        """Generate a completion for a prompt.

        Args:
            model: The model name.
            prompt: The input prompt.
            format: Optional structured-output specification (``"json"`` or a
                JSON schema) forwarded to the server.

        Returns:
            The generated response text.
        """
        if format is None:
            response = self._client.generate(model=model, prompt=prompt)
        else:
            response = self._client.generate(model=model, prompt=prompt, format=format)
        return str(response["response"])

    def embed(self, model: str, text: str) -> list[float]:
        """Embed a text with the configured embedding model.

        Args:
            model: The embedding model name.
            text: The text to embed.

        Returns:
            The dense embedding vector.
        """
        response: Mapping[str, Any] = self._client.embeddings(model=model, prompt=text)
        return [float(value) for value in response["embedding"]]

    def available_backbones(self) -> dict[str, bool]:
        """Report availability of every selected backbone.

        Returns:
            A mapping from model name to availability.
        """
        pulled = set(self.list_models())
        return {model: model in pulled for model in ALL_MODELS}


def create_ollama_client(settings: Settings) -> OllamaClient:
    """Build an ``OllamaClient`` from validated settings.

    Args:
        settings: The validated application settings.

    Returns:
        A wrapped Ollama client.
    """
    if settings.ollama_user and settings.ollama_pass:
        client = ollama.Client(
            host=settings.ollama_host,
            auth=(settings.ollama_user, settings.ollama_pass),
        )
    else:
        client = ollama.Client(host=settings.ollama_host)
    return OllamaClient(client)
