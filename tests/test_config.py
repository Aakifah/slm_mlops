"""Configuration validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings


def test_settings_defaults() -> None:
    """Settings must fall back to documented defaults for undeclared fields."""
    settings = Settings()
    assert settings.seed == 42
    assert settings.rate_limit_per_minute == 60
    assert settings.rate_limit_per_hour == 1000
    assert settings.api_key_auth_enabled is True


def test_invalid_ollama_host_rejected() -> None:
    """A non-HTTP Ollama host must fail validation."""
    with pytest.raises(ValidationError):
        Settings(ollama_host="localhost:11434")


def test_ollama_host_strips_trailing_slash() -> None:
    """Trailing slashes on the host must be normalized away."""
    settings = Settings(ollama_host="http://example.com/")
    assert settings.ollama_host == "http://example.com"


def test_ollama_credentials_optional() -> None:
    """Ollama credentials must default to None when unset."""
    settings = Settings(ollama_host="http://example.com")
    assert settings.ollama_user is None
    assert settings.ollama_pass is None
