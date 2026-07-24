import typer
from alembic import command
from alembic.config import Config
from algomlb.core.logger import logger

app = typer.Typer(
    help="Database initialization, migrations, and status.", no_args_is_help=True
)

# Standard sysexits-style codes so `db health` can be used as a monitoring
# probe: 0 = healthy, 1 = stale data, 2 = pipeline itself isn't running.
_EXIT_STALE_DATA = 1
_EXIT_PIPELINE_DOWN = 2


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


@app.command()
def health(ctx: typer.Context) -> None:
    """
    Check data freshness AND that the daily sync pipeline is actually
    running. Exits non-zero on failure so this can be wired into a monitor:
    0 = healthy, 1 = data is stale, 2 = the pipeline itself looks stopped.
    """
    from algomlb.db.introspection import check_freshness
    from algomlb.db.session import get_engine

    checks = check_freshness(get_engine())

    heartbeat_failed = False
    data_failed = False

    logger.info("📡 AlgoMLB Freshness Check")
    for c in checks:
        icon = "✅" if c.ok else "❌"
        logger.info(f"  {icon} {c.name}: {c.detail}")
        if not c.ok:
            if "heartbeat" in c.name:
                heartbeat_failed = True
            else:
                data_failed = True

    if heartbeat_failed:
        logger.error(
            "🚨 sync_daily heartbeat is stale or failing — the pipeline may "
            "not be running at all, not just producing stale data."
        )
        raise typer.Exit(code=_EXIT_PIPELINE_DOWN)
    if data_failed:
        logger.warning("⚠️ One or more tables are stale.")
        raise typer.Exit(code=_EXIT_STALE_DATA)

    logger.success("✅ All freshness checks passed.")
