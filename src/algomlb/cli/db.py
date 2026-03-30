import typer
from algomlb.core.logger import logger
from algomlb.db.session import Base, create_db_engine

app = typer.Typer(help="Database initialization, migrations, and status.", no_args_is_help=True)


@app.command()
def init(ctx: typer.Context) -> None:
    """Initialize the database schema."""
    logger.info("Initializing database schema...")
    try:
        engine = create_db_engine()
        # Import all ORM models to ensure they're registered with Base before create_all
        import algomlb.db.models  # noqa: F401
        
        Base.metadata.create_all(engine)
        logger.success("Database tables created successfully!")
    except Exception as e:
        logger.exception(f"Failed to initialize database: {e}")
        raise typer.Exit(code=1)


@app.command()
def status(ctx: typer.Context) -> None:
    """Show current database migration status."""
    logger.info("TODO: implement status check")
