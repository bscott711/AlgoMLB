from typing import Self
from pydantic import Field, PostgresDsn, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class DatabaseConfig(BaseSettings):
    """Database connection and ORM configuration."""

    url: PostgresDsn | None = Field(
        default=None, description="PostgreSQL connection string"
    )
    echo: bool = Field(default=False, description="Enable SQLAlchemy query logging")
    pool_size: int = Field(default=5, ge=1, description="Connection pool size")

    @model_validator(mode="after")
    def validate_required_settings(self) -> Self:
        """Enforce required fields that are otherwise optional to allow for 0-arg instantiation."""
        if self.url is None:
            raise ValueError(
                "Database connection URL is required. Set DB__URL env variable."
            )
        return self


class APIConfig(BaseSettings):
    """External API credentials and endpoints."""

    odds_api_key: SecretStr | None = Field(
        default=None, description="API key for The-Odds-API"
    )
    mlb_stats_url: str = Field(
        default="https://statsapi.mlb.com/api/v1",
        description="Base URL for MLB Stats API",
    )

    @model_validator(mode="after")
    def validate_required_settings(self) -> Self:
        if self.odds_api_key is None:
            raise ValueError(
                "The Odds API key is required. Set API__ODDS_API_KEY env variable."
            )
        return self


class MLConfig(BaseSettings):
    """Machine learning and strategy thresholds."""

    min_edge_percent: float = Field(
        default=2.5, ge=0.0, description="Minimum EV percentage to execute a bet"
    )
    max_risk_units: float = Field(
        default=3.0, gt=0.0, description="Maximum units to risk on a single bet"
    )


class Settings(BaseSettings):
    """Root application settings orchestrator."""

    environment: str = Field(
        default="development", description="Current execution environment"
    )

    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        yaml_file="config.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define the priority of configuration sources.
        Env vars take highest priority, followed by the .env file, then the yaml file."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Retrieve the validated application settings. Acts as a singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
