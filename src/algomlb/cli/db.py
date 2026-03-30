import typer
from alembic import command
from alembic.config import Config
from algomlb.core.logger import logger

app = typer.Typer(
    help="Database initialization, migrations, and status.", no_args_is_help=True
)


def _get_alembic_config() -> Config:
    """Load Alembic configuration."""
    # Note: Target the alembic.ini in the project root
    return Config("alembic.ini")


@app.command()
def init(ctx: typer.Context) -> None:
    """Apply all pending migrations to initialize the database."""
    logger.info("Initializing database schema via Alembic...")
    try:
        alembic_cfg = _get_alembic_config()
        # Set the pythonpath to src so models can be imported in env.py
        import os
        import sys

        sys.path.append(os.path.join(os.getcwd(), "src"))

        command.upgrade(alembic_cfg, "head")
        logger.success("Database migrations applied successfully!")
    except Exception as e:
        logger.exception(f"Failed to initialize database: {e}")
        raise typer.Exit(code=1)


@app.command()
def status(ctx: typer.Context) -> None:
    """Show current database migration status."""
    logger.info("Checking database migration status...")
    try:
        alembic_cfg = _get_alembic_config()
        command.current(alembic_cfg)
    except Exception as e:
        logger.exception(f"Failed to check database status: {e}")
        raise typer.Exit(code=1)
