"""Один конфіг для локального запуску й контейнера."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    index_path: Path = Path("data/web-index.json")
    embeddings_path: Path | None = None
    model_cache: Path | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=16)


@lru_cache
def get_settings() -> Settings:
    return Settings()
