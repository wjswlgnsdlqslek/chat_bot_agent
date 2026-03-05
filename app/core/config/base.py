"""
.env: api_key, 설정 변경이 잦은 변수
config.py: .env에서 읽어온 변수들을 기반으로 애플리케이션에서 사용할 설정 정의
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseAppSettings(BaseSettings):
    """공통 애플리케이션 설정."""

    environment: Literal["development", "production", "staging", "test"] = "development"
    debug: bool = False
    upstage_api_key: str = ""
    llm_model: str = "solar-pro2"
    supabase_url: str = ""
    supabase_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    project_name: str = "prac"

    use_litellm: bool = True
    litellm_num_retries: int = 3
    gemini_api_key: str = ""
    litellm_fallback_model: str = "gemini/gemini-2.5-flash"
    litellm_timeout: int = 10

    # 라우터 LLM 모델
    router_llm_model: str = "solar-mini"

    supabase_connection_string: str = ""
    discord_webhook_url: str = ""
    enable_checkpointer: bool = True
    checkpointer_type: Literal["postgres", "memory"] = "postgres"
    enable_cost_tracking: bool = True
    daily_cost_limit: float = 1.0
    max_context_tokens: int = 4096

    enable_langfuse: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_flush_interval: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
