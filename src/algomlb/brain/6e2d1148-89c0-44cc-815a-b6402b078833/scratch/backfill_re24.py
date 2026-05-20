import logging
from datetime import date
from sqlalchemy import select, text
from algomlb.db.session import get_engine
from algomlb.db.models import StatcastRawORM
from algomlb.ml.silver_processor import summarize_to_silver, _upsert_silver
from algomlb.ml.rolling_service import RollingService
from algomlb.ml.rolling_processor import RollingProcessor
from algomlb.config.settings import get_settings
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backfill_re24():
    engine = get_engine()
    settings = get_settings()

    # 1. Backfill Silver Layer re24
    logger.info("Starting Silver Layer RE24 backfill...")
    with engine.connect() as conn:
        # Find games that still have NULL re24 in the summary table
        missing_stmt = text("""
            SELECT DISTINCT game_pk 
            FROM statcast_player_game_logs 
            WHERE re24 IS NULL
        """)
        missing_game_pks = conn.execute(missing_stmt).scalars().all()

    if not missing_game_pks:
        logger.info(
            "All Silver Layer logs already have RE24 populated. Skipping phase 1."
        )
    else:
        total = len(missing_game_pks)
        logger.info(f"Processing {total} games with missing RE24...")
        for i, game_pk in enumerate(missing_game_pks):
            if i % 100 == 0:
                logger.info(f"Processing Silver game {i}/{total} ({game_pk})")

            with engine.connect() as conn:
                query = select(StatcastRawORM).where(StatcastRawORM.game_pk == game_pk)
                df = pd.read_sql(query, conn)

            if df.empty:
                continue

            silver_df = summarize_to_silver(df)
            if not silver_df.empty:
                _upsert_silver(silver_df)

    logger.info("Silver Layer RE24 cleanup complete.")

    # 2. Backfill Gold Layer roll_re24
    logger.info("Starting Gold Layer roll_re24 backfill (2024-2026)...")
    processor = RollingProcessor(settings.ml)
    from algomlb.db.repository import DatabaseRepository
    from algomlb.db.session import get_session_factory

    factory = get_session_factory()
    with factory() as session:
        repo = DatabaseRepository(session)
        service = RollingService(repo, processor)

        start_date = date(2019, 3, 20)  # Start of earliest Statcast era in DB
        end_date = date.today()

        service.process_date_range(start_date, end_date)
        logger.info("Gold Layer roll_re24 backfill complete.")


if __name__ == "__main__":
    backfill_re24()
