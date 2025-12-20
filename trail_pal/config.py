"""Configuration settings for Trail Pal."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql://localhost:5432/trail_pal"

    # OpenRouteService API
    ors_api_key: str = ""
    ors_base_url: str = "https://api.openrouteservice.org"
    ors_requests_per_minute: int = 40
    ors_requests_per_day: int = 2000

    # Overpass API (OpenStreetMap)
    overpass_api_url: str = "https://overpass-api.de/api/interpreter"

    # Hiking constraints
    min_daily_distance_km: float = 10.0
    max_daily_distance_km: float = 20.0
    default_trip_days: int = 3

    # Graph building
    # Hiking routes are typically 1.2-1.5x straight-line distance
    # For 10-20km hikes, filter to ~12km straight-line max
    max_straight_line_distance_km: float = 12.0

    # API Configuration
    # Comma-separated list of valid API keys for third-party access
    api_keys: str = ""

    @property
    def api_keys_list(self) -> list[str]:
        """Parse API keys from comma-separated string."""
        if not self.api_keys:
            return []
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()

