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

    # TODO 1: LiteLLM 사용 여부
    use_litellm: bool = True

    # TODO 2: LiteLLM 재시도 횟수 설정 추가
    litellm_num_retries: int = 3

    # TODO 3 : Gemini API 키 (Fallback용)
    gemini_api_key: str = ""

    # TODO 4: LiteLLM Fallback 모델 설정 추가(gemini-2.5-flash)
    litellm_fallback_model: str = "gemini/gemini-2.5-flash"

    # TODO 5: LiteLLM 타임아웃 설정 10초로 추가
    litellm_timeout: int = 10

    # 라우터 LLM 모델
    router_llm_model: str = "solar-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
