from collections.abc import Generator

import pytest
from pydantic import ValidationError

import algomlb.config.settings as settings_module
from algomlb.config.settings import (
    APIConfig,
    DatabaseConfig,
    MLConfig,
    Settings,
    get_settings,
)


@pytest.fixture(autouse=True)
def reset_singleton() -> Generator[None, None, None]:
    """Ensure the settings singleton is reset before and after every test."""
    settings_module._settings = None
    yield
    settings_module._settings = None


def test_database_config_validation() -> None:
    """Test database configuration constraints and URL parsing."""
    valid_config = DatabaseConfig(url="postgresql+psycopg2://user:pass@localhost/db")  # type: ignore
    assert valid_config.pool_size == 5
    assert valid_config.echo is False

    with pytest.raises(ValidationError, match="url"):
        DatabaseConfig(url="invalid-database-url")  # type: ignore

    with pytest.raises(ValidationError, match="pool_size"):
        DatabaseConfig(url="postgresql://user:pass@localhost/db", pool_size=0)  # type: ignore


def test_api_config_missing_key() -> None:
    """Test that APIConfig fails if odds_api_key is None."""
    # APIConfig requires odds_api_key (model_validator ensures it)
    with pytest.raises(ValidationError, match="Odds API key"):
        APIConfig()


def test_api_config_secondary_key_only() -> None:
    """Test that APIConfig succeeds if only odds_api_key_secondary is provided."""
    config = APIConfig(odds_api_key_secondary="secondary_test_key")  # type: ignore
    assert config.odds_api_key_secondary is not None
    assert config.odds_api_key_secondary.get_secret_value() == "secondary_test_key"
    assert config.odds_api_key is None


def test_ml_config_validation() -> None:
    """Test machine learning threshold constraints."""
    valid_config = MLConfig(min_edge_percent=1.5, max_risk_units=1.0)
    assert valid_config.min_edge_percent == 1.5

    with pytest.raises(ValidationError, match="min_edge_percent"):
        MLConfig(min_edge_percent=-0.1)

    with pytest.raises(ValidationError, match="max_risk_units"):
        MLConfig(max_risk_units=0.0)


def test_settings_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_settings() returns the exact same instance in memory."""
    # Inject required variables so instantiation does not fail
    monkeypatch.setenv("DATABASE__URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("API__ODDS_API_KEY", "test_secret_key")

    instance_one = get_settings()
    instance_two = get_settings()

    assert instance_one is instance_two


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variables properly override defaults via nested delimiters."""
    monkeypatch.setenv("DATABASE__URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("API__ODDS_API_KEY", "test_secret_key")
    monkeypatch.setenv("ML__MIN_EDGE_PERCENT", "5.5")
    monkeypatch.setenv("DATABASE__POOL_SIZE", "20")

    settings = Settings()

    # Verify Pydantic PostgresDsn serialization
    assert str(settings.database.url) == "postgresql://user:pass@localhost/db"

    # Verify SecretStr requires explicit unwrapping
    assert settings.api.odds_api_key is not None
    assert not isinstance(settings.api.odds_api_key, str)
    assert settings.api.odds_api_key.get_secret_value() == "test_secret_key"

    # Verify type coercion from env var strings to integers/floats
    assert settings.ml.min_edge_percent == 5.5
    assert settings.database.pool_size == 20
