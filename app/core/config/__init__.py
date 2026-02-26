import os
from functools import lru_cache

from .base import BaseAppSettings
from .config_dev import DevSettings
from .config_prod import ProdSettings

_ENV_TO_SETTINGS_CLASS = {
    "development": DevSettings,
    "production": ProdSettings,
}


def _resolve_settings_class() -> type[BaseAppSettings]:
    environment = os.getenv("ENVIRONMENT", "development").strip().lower()
    return _ENV_TO_SETTINGS_CLASS.get(environment, BaseAppSettings)


@lru_cache
def get_settings() -> BaseAppSettings:
    settings_class = _resolve_settings_class()
    return settings_class()


settings = get_settings()

__all__ = [
    "BaseAppSettings",
    "DevSettings",
    "ProdSettings",
    "get_settings",
    "settings",
]
