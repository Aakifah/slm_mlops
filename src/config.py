"""Centralized application configuration via Pydantic settings.

Loads environment variables (including a project-root ``.env`` file) and
validates them at startup so misconfigured deployments fail fast rather than at
inference time.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_host: str = Field(default="http://localhost:11434", min_length=1)
    ollama_user: str | None = None
    ollama_pass: str | None = None

    seed: int = 42

    api_key_auth_enabled: bool = True
    api_key: str | None = None
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    cors_origins: str = "http://localhost:3002"

    @field_validator("ollama_host")
    @classmethod
    def validate_ollama_host(cls, value: str) -> str:
        """Reject an Ollama host that is not an HTTP(S) URL.

        Args:
            value: The raw ``OLLAMA_HOST`` value.

        Returns:
            The host with any trailing slash stripped.

        Raises:
            ValueError: If the host is not an HTTP(S) URL.
        """
        if not value.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_HOST must be an http(s) URL")
        return value.rstrip("/")


def get_settings() -> Settings:
    """Return a freshly loaded settings instance.

    Returns:
        The validated application settings.
    """
    return Settings()
