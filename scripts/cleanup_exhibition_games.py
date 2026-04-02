from sqlalchemy import text
from algomlb.db.session import get_engine
from loguru import logger


def cleanup_exhibition_games():
    engine = get_engine()
    with engine.connect() as conn:
        target_criteria = "game_type IN ('A', 'E', 'S') OR game_type IS NULL"

        logger.warning(
            "Purging non-regular season data including historical forecasts..."
        )

        # 1. Purge Pitch Events
        conn.execute(
            text(
                f"DELETE FROM pitch_events WHERE game_id IN (SELECT game_id FROM game_results WHERE {target_criteria})"
            )
        )

        # 2. Purge Weather Progression Data
        conn.execute(
            text(
                f"DELETE FROM openmeteo_weather_progression WHERE game_id IN (SELECT game_id FROM game_results WHERE {target_criteria})"
            )
        )

        # 3. Purge Daily Forecasts (Foreign Key link)
        conn.execute(
            text(
                f"DELETE FROM openmeteo_daily_forecasts WHERE game_id IN (SELECT game_id FROM game_results WHERE {target_criteria})"
            )
        )

        # 4. Purge Game Results
        p = conn.execute(text(f"DELETE FROM game_results WHERE {target_criteria}"))

        conn.commit()
        logger.success(f"Cleanup complete. Total games removed: {p.rowcount}")


if __name__ == "__main__":
    cleanup_exhibition_games()
