from pydantic_settings import SettingsConfigDict

from .base import BaseAppSettings


class ProdSettings(BaseAppSettings):
    """운영 환경 설정."""

    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=(".env.prod", ".env.production"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
