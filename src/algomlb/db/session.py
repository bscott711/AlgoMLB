from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from algomlb.config import get_settings


class Base(DeclarativeBase):
    """Base class for modern SQLAlchemy 2.0 declarative models."""

    pass


def create_db_engine(url: str | None = None, echo: bool | None = None):
    """
    Initialize the SQLAlchemy Engine using the connection URL from settings.
    Allows overriding for testing (e.g., sqlite:///:memory:).
    """
    settings = get_settings()
    db_url = url or str(settings.db.url)
    db_echo = echo if echo is not None else settings.db.echo
    return create_engine(db_url, echo=db_echo)


def get_session_factory(engine):
    """Create a configured session factory."""
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)
