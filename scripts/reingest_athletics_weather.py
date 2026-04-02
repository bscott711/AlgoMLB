import datetime
from sqlalchemy import select, delete
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM, OpenMeteoWeatherProgressionORM
from algomlb.ingestion.openmeteo_ingester import OpenMeteoIngester
from loguru import logger


def reingest_athletics_weather():
    session_factory = get_session_factory()
    ingester = OpenMeteoIngester(session_factory)

    # Athletics Ballpark IDs
    A_BALLPARK_IDS = [20, 33, 34]

    with session_factory() as session:
        # 1. Identity all games played at Athletics venues
        stmt = select(GameResultORM).where(
            GameResultORM.ballpark_id.in_(A_BALLPARK_IDS)
        )
        games = session.execute(stmt).scalars().all()

        if not games:
            logger.warning("No Athletics games found to re-ingest.")
            return

        game_ids = [g.game_id for g in games]
        logger.info(
            f"Found {len(game_ids)} games at A's venues. Clearing old weather data..."
        )

        # 2. Clear old weather data to force fresh ingestion with new coordinates logic
        del_stmt = delete(OpenMeteoWeatherProgressionORM).where(
            OpenMeteoWeatherProgressionORM.game_id.in_(game_ids)
        )
        session.execute(del_stmt)
        session.commit()

        # 3. Group by year for efficient batching
        years = sorted(list(set(g.game_date.year for g in games)))

    for year in years:
        logger.info(f"🚀 Re-ingesting A's weather for {year}...")
        # We can use a wide range for the A's specifically
        ingester.ingest_range(datetime.date(year, 1, 1), datetime.date(year, 12, 31))

    logger.success("Athletics weather re-ingestion complete.")


if __name__ == "__main__":
    reingest_athletics_weather()
