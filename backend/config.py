from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "FinSight AI"
    APP_ENV: str = "development"
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
    GEMINI_API_KEY: str | None = None
    FINSIGHT_LLM_MODEL: str = "gemini-3.5-flash-lite"
    FINSIGHT_HF_REPO_ID: str = "Nandita10/finsight-ai-artifacts"
    HF_TOKEN: str | None = None
    FINSIGHT_ARTIFACT_DIR: str | None = None
    FINSIGHT_HF_REVISION: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
