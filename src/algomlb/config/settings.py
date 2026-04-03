from typing import Self
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PostgresDsn, SecretStr, model_validator
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
    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    user: str = Field(default="postgres", description="PostgreSQL user")
    password: SecretStr | None = Field(default=None, description="PostgreSQL password")
    name: str = Field(default="algomlb", description="PostgreSQL database name")

    echo: bool = Field(default=False, description="Enable SQLAlchemy query logging")
    pool_size: int = Field(default=5, ge=1, description="Connection pool size")

    @model_validator(mode="after")
    def validate_and_sync_settings(self) -> Self:
        """
        Enforce required settings and populate individual components from the URL if provided.
        """
        if self.url:
            # Sync individual fields from the URL for convenience
            hosts = self.url.hosts()
            if hosts:
                self.host = hosts[0].get("host") or self.host
                self.port = hosts[0].get("port") or self.port
                self.user = hosts[0].get("username") or self.user
            if self.url.path:
                self.name = self.url.path.lstrip("/")
        # If no URL, we rely on the host, user, name which have defaults but
        # we can still enforce some logic here if needed.
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

    # --- Quant Layer (Epic 2.2) ---
    quant_baseline_window: int = Field(
        default=14, ge=1, description="Days to look back for player baseline averages"
    )

    # --- Silver Layer (Epic 2.3) ---
    quant_pitcher_shrinkage_k: int = Field(
        default=75, ge=0, description="Shrinkage factor (k) for pitcher metrics"
    )
    quant_batter_shrinkage_k: int = Field(
        default=250, ge=0, description="Shrinkage factor (k) for batter metrics"
    )


class DBHealthConfig(BaseModel):
    """Configuration for database health and introspection."""

    allow_null_columns: dict[str, list[str]] = Field(
        default_factory=dict, description="Table-to-column mapping of allowed NULLs"
    )
    known_placeholders: list[str] = Field(
        default_factory=list, description="Tables that are expected to be empty"
    )
    table_naming_pattern: str = Field(
        default="^[a-z][a-z0-9_]*$", description="Regex for table naming convention"
    )


class Settings(BaseSettings):
    """Root application settings orchestrator."""

    environment: str = Field(
        default="development", description="Current execution environment"
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    db_health: DBHealthConfig = Field(default_factory=DBHealthConfig)

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
        load_dotenv()
        _settings = Settings()
    return _settings
