import datetime
from sqlalchemy import update
from loguru import logger
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM
from algomlb.ingestion.mlb_stats import MLBStatsAPIClient


def fetch_and_repair_times(years=[2019, 2020, 2021, 2022, 2023, 2024, 2025]):
    client = MLBStatsAPIClient()
    session_factory = get_session_factory()

    for year in years:
        start_date = datetime.date(year, 3, 20)
        end_date = datetime.date(year, 11, 15)

        logger.info(f"📡 Fetching times for {year} ({start_date} to {end_date})...")
        try:
            games = client.fetch_daily_schedule(
                start_date=start_date, end_date=end_date
            )
            if not games:
                logger.warning(f"No games for {year}")
                continue

            with session_factory() as session:
                repaired = 0
                for g in games:
                    stmt = (
                        update(GameResultORM)
                        .where(GameResultORM.game_id == g.game_id)
                        .where(GameResultORM.game_datetime.is_(None))
                        .values(game_datetime=g.game_datetime)
                    )
                    result = session.execute(stmt)
                    if result.rowcount > 0:
                        repaired += 1
                session.commit()
                logger.success(f"✅ Updated {repaired} games for {year}.")
        except Exception as e:
            logger.error(f"Error for {year}: {e}")


if __name__ == "__main__":
    fetch_and_repair_times()
