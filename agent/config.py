"""Agent configuration using pydantic-settings for env var support."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Agent configuration loaded from environment variables or .env file."""

    llm_provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    api_key: str = ""
    max_steps: int = 10
    temperature: float = 0.0
    log_level: str = "INFO"

    model_config = {
        "env_prefix": "AGENT_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
