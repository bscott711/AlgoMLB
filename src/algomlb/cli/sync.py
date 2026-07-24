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
    9. Process Silver Layer (Incremental + trailing-window guarantee)
    10. Process Gold Layer (Rolling Features)

    Each stage below is isolated: a failure in one data source is logged and
    skipped rather than rolling back every other stage, so one dead API can
    never freeze the whole pipeline (see 2026-06-30 incident).
    """
    # 1. Setup Dates
    if target_date:
        today = date.fromisoformat(target_date)
    else:
        today = date.today()

    yesterday = today - timedelta(days=1)
    start_trailing = today - timedelta(days=days_back)

    logger.info(f"🔄 Starting Daily Sync (Target: {yesterday})")

    stage_results: dict[str, bool] = {}

    def run_stage(name: str, fn, commit_session=None) -> None:
        try:
            fn()
            if commit_session is not None:
                commit_session.commit()
            stage_results[name] = True
            logger.success(f"✅ Stage OK: {name}")
        except Exception as e:
            if commit_session is not None:
                commit_session.rollback()
            stage_results[name] = False
            logger.error(f"❌ Stage FAILED: {name} — {e}")

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

        run_stage(
            "schedules",
            lambda: orchestrator.run_schedule_ingestion(
                start_date=start_trailing, end_date=today
            ),
            session,
        )
        run_stage(
            "statcast",
            lambda: orchestrator.run_statcast_ingestion(
                start_date=start_trailing, end_date=yesterday
            ),
            session,
        )
        run_stage(
            "gumbo",
            lambda: orchestrator.run_gumbo_ingestion(
                start_date=start_trailing, end_date=today
            ),
            session,
        )
        run_stage(
            "weather",
            lambda: orchestrator.run_weather_ingestion(
                start_date=start_trailing, end_date=today
            ),
            session,
        )
        run_stage("transactions", orchestrator.run_transaction_ingestion, session)
        run_stage(
            "umpires",
            lambda: orchestrator.run_umpire_ingestion(
                seasons=[today.year - 1, today.year]
            ),
            session,
        )
        run_stage(
            "lineups",
            lambda: orchestrator.run_lineup_ingestion(
                start_date=start_trailing, end_date=today
            ),
            session,
        )
        run_stage("live_odds", orchestrator.run_odds_ingestion, session)

    # 2. Processing (Silver/Gold) — each step isolated so one failure doesn't
    # skip the rest (e.g. Elo/Sabermetrics/Gold/Betting must still run even if
    # Silver has nothing new to process).
    logger.info("🛠️ Running Processing Pipelines")

    from algomlb.ml.silver_processor import (
        ensure_silver_coverage,
        process_silver_incremental,
    )

    run_stage("silver_incremental", lambda: process_silver_incremental(batch_size=50000))
    run_stage(
        "silver_trailing_window",
        lambda: ensure_silver_coverage(start_trailing, yesterday),
    )

    from algomlb.ml.elo import backfill_team_elo_history
    from algomlb.ml.sabermetrics import backfill_team_sabermetrics_history

    run_stage(
        "elo", lambda: backfill_team_elo_history(engine=session_factory.kw["bind"])
    )
    run_stage(
        "sabermetrics",
        lambda: backfill_team_sabermetrics_history(engine_in=session_factory.kw["bind"]),
    )

    # Gold Layer (Rolling Features for trailing window to yesterday)
    from algomlb.ml.rolling_service import RollingService
    from algomlb.ml.rolling_processor import RollingProcessor
    from algomlb.config.settings import get_settings

    settings = get_settings()

    def _gold():
        with session_factory() as gold_session:
            db = DatabaseRepository(gold_session)
            processor = RollingProcessor(settings.ml)
            service = RollingService(db, processor)
            service.process_date_range(start_trailing, yesterday)

    run_stage("gold_rolling_features", _gold)

    # 3. Paper Trading (Betting + Settlement)
    logger.info("💰 Running Paper Trading Engine")
    from algomlb.strategy.betting_service import BettingService

    def _paper_trade():
        with session_factory() as bet_session:
            bet_service = BettingService(bet_session)
            settled = bet_service.settle_bets()
            placed = bet_service.place_daily_bets(today)
            logger.success(
                f"✅ Paper Trading: Settled {settled} bets, Placed {placed} new bets."
            )

    run_stage("paper_trading", _paper_trade)

    # 4. Closing Line Value — best-effort over the trailing window; games
    # without a closing snapshot yet are simply skipped and picked up on a
    # later run once live_odds has a pre-first-pitch snapshot for them.
    from algomlb.ml.clv import compute_clv_for_range

    run_stage("clv", lambda: compute_clv_for_range(start_trailing, yesterday))

    failed = [name for name, ok in stage_results.items() if not ok]
    if failed:
        logger.warning(
            f"⚠️ Daily Sync completed with {len(failed)} failed stage(s) for "
            f"{yesterday}: {failed}"
        )
    else:
        logger.success(
            f"✅ Daily Sync Complete for {yesterday} — all {len(stage_results)} stages OK"
        )

    # Heartbeat: recorded even on partial failure, so `algomlb db health` can
    # tell "the job ran but some stages failed" apart from "the job never ran
    # at all" — the exact blind spot behind the 2026-06-30 incident.
    _record_heartbeat("sync_daily", stage_results)


def _record_heartbeat(job_name: str, stage_results: dict[str, bool]) -> None:
    import datetime as _dt

    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from algomlb.db.models import PipelineHeartbeatORM

    succeeded = [n for n, ok in stage_results.items() if ok]
    failed = [n for n, ok in stage_results.items() if not ok]
    if not stage_results:
        status = "FAILED"
    elif not failed:
        status = "OK"
    elif succeeded:
        status = "PARTIAL"
    else:
        status = "FAILED"

    now = _dt.datetime.now(_dt.timezone.utc)
    values = {
        "job_name": job_name,
        "last_run_at": now,
        "last_success_at": now if succeeded else None,
        "last_status": status,
        "details": f"failed={failed}" if failed else None,
    }

    engine = get_session_factory().kw["bind"]
    with engine.begin() as conn:
        stmt = pg_insert(PipelineHeartbeatORM).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["job_name"],
            set_={
                "last_run_at": stmt.excluded.last_run_at,
                # Only overwrite last_success_at when this run actually
                # succeeded at something — a fully-failed run shouldn't erase
                # the last known-good timestamp.
                "last_success_at": (
                    stmt.excluded.last_success_at
                    if succeeded
                    else PipelineHeartbeatORM.last_success_at
                ),
                "last_status": stmt.excluded.last_status,
                "details": stmt.excluded.details,
            },
        )
        conn.execute(stmt)


if __name__ == "__main__":
    app()
