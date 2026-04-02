import typer
import datetime
from typing import Optional
from algomlb.core.agent_io import AgentResult, emit_agent_result
from algomlb.core.logger import logger
from algomlb.db.repository import DatabaseRepository
from algomlb.db.session import get_session_factory
from algomlb.ingestion import (
    IngestionOrchestrator,
    MLBStatsAPIClient,
    OddsAPIClient,
    OpenMeteoIngester,
    PlayerTransactionsIngester,
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
        transactions_ingester = PlayerTransactionsIngester(repo)
        openmeteo_ingester = OpenMeteoIngester(session)
        orchestrator = IngestionOrchestrator(
            repo,
            odds_client,
            stats_client,
            historical_loader,
            transactions_ingester,
            openmeteo_ingester,
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
        transactions_ingester = PlayerTransactionsIngester(repo)
        openmeteo_ingester = OpenMeteoIngester(session)
        orchestrator = IngestionOrchestrator(
            repo,
            odds_client,
            stats_client,
            historical_loader,
            transactions_ingester,
            openmeteo_ingester,
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
        transactions_ingester = PlayerTransactionsIngester(repo)
        openmeteo_ingester = OpenMeteoIngester(session)
        orchestrator = IngestionOrchestrator(
            repo,
            odds_client,
            stats_client,
            historical_loader,
            transactions_ingester,
            openmeteo_ingester,
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
    date: str = typer.Option(None, "--date", help="Single date to ingest (YYYY-MM-DD)"),
    start: str = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    reverse: bool = typer.Option(True, "--reverse", help="Ingest in reverse order"),
):
    """Ingest historical odds snapshots for a date or range."""
    import datetime

    session_factory = get_session_factory()
    with session_factory() as session:
        repo = DatabaseRepository(session)
        ingester = HistoricalOddsIngester(repo)

        if date:
            day = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            ingester.ingest_day_snapshots(day)
        elif start and end:
            s_date = datetime.datetime.strptime(start, "%Y-%m-%d").date()
            e_date = datetime.datetime.strptime(end, "%Y-%m-%d").date()
            ingester.run_backfill(s_date, e_date, reverse=reverse)
        else:
            logger.error("Must provide either --date or both --start and --end.")


@app.command()
def umpire_scorecards(
    ctx: typer.Context,
    since: int = typer.Option(2019, "--since", help="Starting year for ingestion"),
    scrape: bool = typer.Option(
        False, "--scrape", help="Scrape from umpscorecards.us API"
    ),
):
    """Ingest umpire accuracy and bias data from umpscorecards.us API."""
    session_factory = get_session_factory()
    with session_factory() as session:
        ingester = UmpireScorecardIngester(session, since_year=since)
        if scrape:
            count = ingester.ingest_from_api()
            logger.success(f"Scraped {count} scorecards from umpscorecards.us API.")
        else:
            logger.warning(
                "No ingestion source specified. Use --scrape to fetch from the API."
            )


@app.command()
def retrosheet(
    ctx: typer.Context,
    csv_path: Optional[str] = typer.Option(
        None, "--csv", help="Path to parsed Retrosheet CSV"
    ),
    url: Optional[str] = typer.Option(
        None, "--url", help="Direct URL to Retrosheet ZIP"
    ),
    since: int = typer.Option(2019, "--since", help="Starting year for ingestion"),
):
    """Ingest Retrosheet play-by-play events (local file or URL)."""
    session_factory = get_session_factory()
    with session_factory() as session:
        ingester = RetrosheetIngester(session, since_year=since)
        if csv_path:
            ingester.ingest_from_csv(csv_path)
        elif url:
            ingester.ingest_from_url(url)
        else:
            # Default to seasonal downloads to stay within disk quotas
            current_year = datetime.datetime.now().year
            logger.info(
                f"Ingesting Retrosheet seasons from {since} to {current_year}..."
            )
            for year in range(since, current_year + 1):
                seasonal_url = (
                    f"https://www.retrosheet.org/downloads/plays/{year}plays.zip"
                )
                try:
                    ingester.ingest_from_url(seasonal_url)
                except Exception as e:
                    logger.warning(f"Could not ingest Retrosheet for {year}: {e}")


@app.command()
def transactions(
    ctx: typer.Context,
    start: str = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
):
    """Fetch MLB player transactions and IL stints."""
    agent_mode = ctx.obj.get("agent_mode", False)
    import datetime

    session_factory = get_session_factory()
    with session_factory() as session:
        repo = DatabaseRepository(session)
        odds_client = OddsAPIClient()
        stats_client = MLBStatsAPIClient()
        historical_loader = HistoricalDataLoader(repo)
        transactions_ingester = PlayerTransactionsIngester(repo)
        openmeteo_ingester = OpenMeteoIngester(session)
        orchestrator = IngestionOrchestrator(
            repo,
            odds_client,
            stats_client,
            historical_loader,
            transactions_ingester,
            openmeteo_ingester,
        )

        s_date = datetime.datetime.strptime(start, "%Y-%m-%d").date() if start else None
        e_date = datetime.datetime.strptime(end, "%Y-%m-%d").date() if end else None

        logger.info(
            f"Starting transaction ingestion for {start or 'trailing 7d'} to {end or 'trailing 7d'}..."
        )
        records_inserted = orchestrator.run_transaction_ingestion(
            start_date=s_date, end_date=e_date
        )
        logger.success(f"Successfully ingested {records_inserted} transaction records.")

        if agent_mode:
            emit_agent_result(
                AgentResult(
                    status="success",
                    command="ingest.transactions",
                    data={"records_inserted": records_inserted},
                )
            )


@app.command()
def weather(
    ctx: typer.Context,
    start: str = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
):
    """Fetch Open-Meteo weather progression and market deltas."""
    # Setup Infrastructure
    session_factory = get_session_factory()

    with session_factory() as session:
        repo = DatabaseRepository(session)
        odds_client = OddsAPIClient()
        stats_client = MLBStatsAPIClient()
        historical_loader = HistoricalDataLoader(repo)
        transactions_ingester = PlayerTransactionsIngester(repo)
        openmeteo_ingester = OpenMeteoIngester(session_factory)
        orchestrator = IngestionOrchestrator(
            repo,
            odds_client,
            stats_client,
            historical_loader,
            transactions_ingester,
            openmeteo_ingester,
        )

        s_date = datetime.datetime.strptime(start, "%Y-%m-%d").date() if start else None
        e_date = datetime.datetime.strptime(end, "%Y-%m-%d").date() if end else None

        logger.info(
            f"Starting weather ingestion for {start or 'trailing 7d'} to {end or 'trailing 7d'}..."
        )
        orchestrator.run_weather_ingestion(
            start_date=s_date, end_date=e_date
        )
        logger.success("Successfully completed weather ingestion.")
