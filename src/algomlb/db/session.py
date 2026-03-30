from typing import Any
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from algomlb.config import get_settings
from algomlb.config.settings import DatabaseConfig


class Base(DeclarativeBase):
    """Base class for modern SQLAlchemy 2.0 declarative models."""

    pass


# Singletons
_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def create_db_engine(
    url: str | DatabaseConfig | None = None, echo: bool | None = None
) -> Engine:
    """
    Initialize the SQLAlchemy Engine using the connection URL from settings.
    Allows passing a connection string or a DatabaseConfig object.
    """
    settings = get_settings()

    if isinstance(url, DatabaseConfig):
        db_url = str(url.url)
        db_echo = echo if echo is not None else url.echo
    else:
        db_url = url or str(settings.database.url)
        db_echo = echo if echo is not None else settings.database.echo

    engine_kwargs: dict[str, Any] = {
        "echo": db_echo,
    }

    # Only include pooling args for databases that support them (e.g. Postgres)
    if not db_url.startswith("sqlite"):
        engine_kwargs["pool_size"] = settings.database.pool_size
        engine_kwargs["max_overflow"] = 10

    return create_engine(db_url, **engine_kwargs)


def get_engine() -> Engine:
    """Retrieve the singleton database engine."""
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def get_session_factory() -> sessionmaker:
    """Retrieve the singleton session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _session_factory
