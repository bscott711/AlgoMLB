import typer
from algomlb.core.agent_io import AgentResult, emit_agent_result
from algomlb.core.logger import logger
from algomlb.db.repository import DatabaseRepository
from algomlb.db.session import create_db_engine, get_session_factory
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
    engine = create_db_engine()
    session_factory = get_session_factory(engine)

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
    engine = create_db_engine()
    session_factory = get_session_factory(engine)

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
    start_year: int = typer.Option(2023, help="Start year for historical data"),
    end_year: int = typer.Option(2023, help="End year for historical data"),
    statcast: bool = typer.Option(False, help="In addition, fetch Statcast pitch data"),
):
    """Fetch historical MLB stats (pitching/batting) and persist to DB."""
    agent_mode = ctx.obj.get("agent_mode", False)

    # Setup Infrastructure
    engine = create_db_engine()
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        repo = DatabaseRepository(session)
        odds_client = OddsAPIClient()
        stats_client = MLBStatsAPIClient()
        historical_loader = HistoricalDataLoader(repo)
        orchestrator = IngestionOrchestrator(
            repo, odds_client, stats_client, historical_loader
        )

        logger.info(f"Starting historical ingestion for {start_year}-{end_year}...")
        records_processed = orchestrator.run_historical_ingestion(start_year, end_year)

        if statcast:
            logger.info(f"Fetching Statcast data for {start_year}...")
            # For brevity, fetch opening month of the start year
            s_df = historical_loader.fetch_statcast(
                f"{start_year}-04-01", f"{start_year}-04-30"
            )
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
