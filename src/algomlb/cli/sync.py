from datetime import date, timedelta
import typer
from algomlb.core.logger import logger
from algomlb.db.repository import DatabaseRepository
from algomlb.db.session import get_session_factory
from algomlb.ingestion import (
    IngestionOrchestrator,
    MLBStatsAPIClient,
    OddsAPIClient,
    OpenMeteoIngester,
    PlayerTransactionsIngester,
    StatcastIngester,
    UmpireScorecardIngester,
    HistoricalDataLoader,
    GumboIngester,
    LineupIngester,
)

app = typer.Typer(help="Synchronize all data layers (Ingest + Process).")


@app.command("daily")
def daily(
    target_date: str = typer.Option(
        None, "--date", help="Target date for sync (YYYY-MM-DD). Defaults to yesterday."
    ),
    days_back: int = typer.Option(
        3, "--days-back", help="Number of trailing days to sync for schedules/weather."
    ),
):
    """
    Perform a complete daily synchronization:
    1. Ingest Schedules (results/scores)
    2. Ingest Statcast (raw pitch data)
    3. Ingest Weather (Open-Meteo)
    4. Ingest Transactions
    5. Ingest Transactions
    6. Ingest Umpires (Scrape)
    7. Ingest Starting Lineups
    8. Ingest Live Odds
    9. Process Silver Layer (Incremental)
    10. Process Gold Layer (Rolling Features)
    """
    # 1. Setup Dates
    if target_date:
        today = date.fromisoformat(target_date)
    else:
        today = date.today()

    yesterday = today - timedelta(days=1)
    start_trailing = today - timedelta(days=days_back)

    logger.info(f"🔄 Starting Daily Sync (Target: {yesterday})")

    session_factory = get_session_factory()
    with session_factory() as session:
        repo = DatabaseRepository(session)

        # Ingestion Setup
        orchestrator = IngestionOrchestrator(
            repo=repo,
            odds_client=OddsAPIClient(),
            stats_client=MLBStatsAPIClient(),
            historical_loader=HistoricalDataLoader(repo),
            transactions_ingester=PlayerTransactionsIngester(repo),
            openmeteo_ingester=OpenMeteoIngester(session_factory),
            statcast_ingester=StatcastIngester(repo=repo),
            umpire_ingester=UmpireScorecardIngester(session),
            lineup_ingester=LineupIngester(session),
            gumbo_ingester=GumboIngester(session),
        )

        # A. Ingest Schedules (results for trailing window)
        logger.info(f"📅 Syncing Schedules: {start_trailing} to {today}")
        orchestrator.run_schedule_ingestion(start_date=start_trailing, end_date=today)

        # C. Ingest Statcast (yesterday)
        logger.info(f"⚾ Syncing Statcast: {yesterday}")
        orchestrator.run_statcast_ingestion(start_date=yesterday, end_date=yesterday)

        # D. Ingest GUMBO (trailing window)
        logger.info(f"🍲 Syncing GUMBO Pitch Clocks: {start_trailing} to {today}")
        orchestrator.run_gumbo_ingestion(start_date=start_trailing, end_date=today)

        # E. Ingest Weather (trailing window)
        logger.info(f"⛅ Syncing Weather: {start_trailing} to {today}")
        orchestrator.run_weather_ingestion(start_date=start_trailing, end_date=today)

        # F. Ingest Transactions (trailing 7 days)
        logger.info("💸 Syncing Transactions (Trailing 7d)")
        orchestrator.run_transaction_ingestion()

        # G. Ingest Umpires (Current + Previous Season for transition)
        logger.info("⚖️ Syncing Umpires (Current Season)")
        cur_year = today.year
        orchestrator.run_umpire_ingestion(seasons=[cur_year - 1, cur_year])

        # H. Ingest Lineups (trailing window)
        logger.info(f"⚾ Syncing Lineups: {start_trailing} to {today}")
        orchestrator.run_lineup_ingestion(start_date=start_trailing, end_date=today)

        # I. Ingest Live Odds
        logger.info("💸 Syncing Live Odds")
        orchestrator.run_odds_ingestion()

        session.commit()

    # 2. Processing (Silver/Gold)
    logger.info("🛠️ Running Processing Pipelines")

    # Silver Layer (Incremental)
    from algomlb.ml.silver_processor import process_silver_incremental

    process_silver_incremental(batch_size=50000)

    # Gold Layer (Rolling Features for yesterday)
    from algomlb.ml.rolling_service import RollingService
    from algomlb.ml.rolling_processor import RollingProcessor
    from algomlb.config.settings import get_settings

    settings = get_settings()
    with session_factory() as session:
        db = DatabaseRepository(session)
        processor = RollingProcessor(settings.ml)
        service = RollingService(db, processor)
        service.process_date_range(yesterday, yesterday)

    # 3. Paper Trading (Betting + Settlement)
    logger.info("💰 Running Paper Trading Engine")
    from algomlb.strategy.betting_service import BettingService
    with session_factory() as session:
        bet_service = BettingService(session)
        settled = bet_service.settle_bets()
        placed = bet_service.place_daily_bets(today)
        logger.success(f"✅ Paper Trading: Settled {settled} bets, Placed {placed} new bets.")

    logger.success(f"✅ Daily Sync Complete for {yesterday}")


if __name__ == "__main__":
    app()
