from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseAppSettings(BaseSettings):
    """공통 애플리케이션 설정."""

    environment: Literal["development", "production", "staging", "test"] = (
        "development"
    )
    debug: bool = False
    upstage_api_key: str = ""
    llm_model: str = "solar-pro2"
    supabase_url: str = ""
    supabase_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    project_name: str = "prac"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
