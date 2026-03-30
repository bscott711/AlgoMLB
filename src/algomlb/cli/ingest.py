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
        orchestrator = IngestionOrchestrator(repo, odds_client, stats_client)

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
        orchestrator = IngestionOrchestrator(repo, odds_client, stats_client)

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
