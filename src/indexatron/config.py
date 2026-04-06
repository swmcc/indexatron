"""Environment-based configuration for Indexatron."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Indexatron configuration loaded from environment variables."""

    # Environment
    env: Literal["development", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # API connection (the-mcculloughs.org)
    api_base_url: str = "http://localhost:3000"
    api_key: str = Field(description="API key for the-mcculloughs.org")

    # Ollama
    ollama_host: str = "http://localhost:11434"
    vision_model: str = "llava:7b"
    embedding_model: str = "nomic-embed-text"

    # Processing
    batch_size: int = Field(default=10, description="Number of uploads to fetch per run")
    download_dir: Path = Field(
        default=Path("/tmp/indexatron"),
        description="Directory for temporary image downloads",
    )

    # Request timeouts (seconds)
    api_timeout: float = 30.0
    ollama_timeout: float = 120.0  # LLaVA can be slow

    model_config = SettingsConfigDict(
        env_prefix="INDEXATRON_",
        env_file=f".env.{os.getenv('INDEXATRON_ENV', 'development')}",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure download directory exists
        self.download_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance - lazy loaded
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings
