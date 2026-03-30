import typer
from typing import Optional
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
from algomlb.ingestion.ballpark_ingester import BallparkIngester
from algomlb.ingestion.historical_odds import HistoricalOddsIngester
from algomlb.ingestion.umpire_ingester import UmpireScorecardIngester
from algomlb.ingestion.retrosheet_ingester import RetrosheetIngester

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
def schedule(
    ctx: typer.Context,
    start: str = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
):
    """Fetch MLB game schedule/results and persist to DB."""
    agent_mode = ctx.obj.get("agent_mode", False)
    import datetime

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

        s_date = datetime.datetime.strptime(start, "%Y-%m-%d").date() if start else None
        e_date = datetime.datetime.strptime(end, "%Y-%m-%d").date() if end else None

        logger.info(
            f"Starting schedule ingestion for {start or 'today'} to {end or 'today'}..."
        )
        records_inserted = orchestrator.run_schedule_ingestion(
            start_date=s_date, end_date=e_date
        )
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


@app.command()
def ballparks(
    ctx: typer.Context,
    csv_path: str = typer.Option(
        None, "--csv", help="Path to ballpark CSV (from Kaggle)"
    ),
):
    """Ingest MLB ballpark structural data from CSV."""
    session_factory = get_session_factory()
    with session_factory() as session:
        ingester = BallparkIngester(session)
        # Default to the cached kaggle path if not specified
        path = (
            csv_path
            or "/home/opc/.cache/kagglehub/datasets/paulrjohnson/mlb-ballparks/versions/4/ballparks.csv"
        )
        ingester.ingest_from_csv(path)


@app.command()
def historical_odds(
    ctx: typer.Context,
    date: str = typer.Option(..., "--date", help="Date to ingest (YYYY-MM-DD)"),
):
    """Ingest opening and closing odds snapshots for a specific date."""
    import datetime

    session_factory = get_session_factory()
    with session_factory() as session:
        repo = DatabaseRepository(session)
        ingester = HistoricalOddsIngester(repo)

        day = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        ingester.ingest_day_snapshots(day)


@app.command()
def umpire_scorecards(
    ctx: typer.Context,
    csv_path: Optional[str] = typer.Option(
        None, "--csv", help="Path to umpire scorecard CSV"
    ),
    url: Optional[str] = typer.Option(None, "--url", help="Direct URL to umpire CSV"),
):
    """Ingest umpire accuracy and bias data (local file, URL, or Kaggle)."""
    session_factory = get_session_factory()
    with session_factory() as session:
        ingester = UmpireScorecardIngester(session)
        if csv_path:
            ingester.ingest_from_csv(csv_path)
        elif url:
            ingester.ingest_from_url(url)
        else:
            ingester.ingest_from_kaggle()


@app.command()
def retrosheet(
    ctx: typer.Context,
    csv_path: Optional[str] = typer.Option(
        None, "--csv", help="Path to parsed Retrosheet CSV"
    ),
    url: Optional[str] = typer.Option(
        None, "--url", help="Direct URL to Retrosheet ZIP"
    ),
):
    """Ingest Retrosheet play-by-play events (local file or URL)."""
    session_factory = get_session_factory()
    with session_factory() as session:
        ingester = RetrosheetIngester(session)
        if csv_path:
            ingester.ingest_from_csv(csv_path)
        else:
            # Default to the full historical parsed data if no URL/CSV is provided
            target_url = url or "https://www.retrosheet.org/downloads/plays/plays.zip"
            ingester.ingest_from_url(target_url)
