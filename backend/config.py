from functools import lru_cache

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
    FINSIGHT_LLM_MODEL: str = "gemini-2.5-flash-lite"


@lru_cache
def get_settings() -> Settings:
    return Settings()
