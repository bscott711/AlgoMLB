import typer
from algomlb.core.agent_io import AgentResult, emit_agent_result
from algomlb.core.logger import logger
from algomlb.db.repository import DatabaseRepository
from algomlb.db.session import get_session_factory
from algomlb.ingestion import (
    IngestionOrchestrator,
    MLBStatsAPIClient,
    OddsAPIClient,
)
from algomlb.ingestion.historical import HistoricalDataLoader

app = typer.Typer(help="Ingest data from external APIs.", no_args_is_help=True)


@app.command()
def odds(ctx: typer.Context):
    """Fetch latest live odds data from The-Odds-API and persist to DB."""
    agent_mode = ctx.obj.get("agent_mode", False)

    # Setup Infrastructure
    session_factory = get_session_factory()

    with session_factory() as session:
        repo = DatabaseRepository(session)
        odds_client = OddsAPIClient()
        stats_client = MLBStatsAPIClient()
        historical_loader = HistoricalDataLoader(repo)
        orchestrator = IngestionOrchestrator(
            repo, odds_client, stats_client, historical_loader
        )

        logger.info("Starting live odds ingestion...")
        records_inserted = orchestrator.run_odds_ingestion()
        logger.success(f"Successfully ingested {records_inserted} odds records.")

        if agent_mode:
            emit_agent_result(
                AgentResult(
                    status="success",
                    command="ingest.odds",
                    data={"records_inserted": records_inserted},
                )
            )


@app.command()
def schedule(ctx: typer.Context):
    """Fetch latest MLB game schedule from Stats API and persist to DB."""
    agent_mode = ctx.obj.get("agent_mode", False)

    # Setup Infrastructure
    session_factory = get_session_factory()

    with session_factory() as session:
        repo = DatabaseRepository(session)
        odds_client = OddsAPIClient()
        stats_client = MLBStatsAPIClient()
        historical_loader = HistoricalDataLoader(repo)
        orchestrator = IngestionOrchestrator(
            repo, odds_client, stats_client, historical_loader
        )

        logger.info("Starting daily schedule ingestion...")
        records_inserted = orchestrator.run_schedule_ingestion()
        logger.success(f"Successfully ingested {records_inserted} schedule records.")

        if agent_mode:
            emit_agent_result(
                AgentResult(
                    status="success",
                    command="ingest.schedule",
                    data={"records_inserted": records_inserted},
                )
            )


@app.command()
def historical(
    ctx: typer.Context,
    start_year: int = typer.Option(
        None, "--start-year", help="Start year for historical stats"
    ),
    end_year: int = typer.Option(
        None, "--end-year", help="End year for historical stats"
    ),
    start: str = typer.Option(
        None, "--start", help="Start date (YYYY-MM-DD) for Statcast data"
    ),
    end: str = typer.Option(
        None, "--end", help="End date (YYYY-MM-DD) for Statcast data"
    ),
    statcast: bool = typer.Option(
        False, "--statcast", help="Fetch Statcast data if enabled"
    ),
):
    """Fetch historical MLB stats (pitching/batting/statcast) and persist to DB."""
    agent_mode = ctx.obj.get("agent_mode", False)

    # Setup Infrastructure
    session_factory = get_session_factory()

    with session_factory() as session:
        repo = DatabaseRepository(session)
        odds_client = OddsAPIClient()
        stats_client = MLBStatsAPIClient()
        historical_loader = HistoricalDataLoader(repo)
        orchestrator = IngestionOrchestrator(
            repo, odds_client, stats_client, historical_loader
        )

        records_processed = 0

        # Handle Yearly Aggregate Stats
        if start_year and end_year:
            logger.info(f"Starting historical ingestion for {start_year}-{end_year}...")
            records_processed += orchestrator.run_historical_ingestion(
                start_year, end_year
            )

        # Handle Pitch-Level Statcast Data
        if start and end:
            logger.info(f"Fetching pitch-level Statcast data from {start} to {end}...")
            s_df = historical_loader.fetch_statcast(start, end)
            records_processed += len(s_df)
        elif statcast and start_year:
            # Fallback backward compatibility for statcast month pull
            start_date = f"{start_year}-04-01"
            end_date = f"{start_year}-04-30"
            logger.info(
                f"Fetching Statcast data for {start_year} (defaults to April)..."
            )
            s_df = historical_loader.fetch_statcast(start_date, end_date)
            records_processed += len(s_df)

        logger.success(
            f"Successfully processed {records_processed} historical records."
        )

        if agent_mode:
            emit_agent_result(
                AgentResult(
                    status="success",
                    command="ingest.historical",
                    data={"records_processed": records_processed},
                )
            )
