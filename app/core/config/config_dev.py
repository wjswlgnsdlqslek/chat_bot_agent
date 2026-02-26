from pydantic_settings import SettingsConfigDict

from .base import BaseAppSettings


class DevSettings(BaseAppSettings):
    """개발 환경 설정."""

    debug: bool = True

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.dev", ".env.development"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
