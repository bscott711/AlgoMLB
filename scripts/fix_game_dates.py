import pytz
from sqlalchemy import select, update
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM, StatcastRawORM, PitchEventORM
from algomlb.core.logger import logger


def fix_game_dates():
    session = get_session_factory()()
    tz_et = pytz.timezone("America/New_York")

    logger.info("Starting historical game date correction (UTC Rollover Fix)...")

    # 1. Identify misdated games in game_results
    # Heuristic: games starting between 00:00 and 08:00 UTC are potentially misdated
    query = select(GameResultORM).where(GameResultORM.game_datetime.isnot(None))
    games = session.execute(query).scalars().all()

    corrections = []
    for g in games:
        # Localize UTC datetime to Eastern Time
        dt_utc = g.game_datetime
        if dt_utc.tzinfo is None:
            dt_utc = pytz.utc.localize(dt_utc)

        dt_et = dt_utc.astimezone(tz_et)
        official_date = dt_et.date()

        if g.game_date != official_date:
            corrections.append(
                {
                    "game_id": g.game_id,
                    "old_date": g.game_date,
                    "new_date": official_date,
                }
            )

    logger.info(f"Found {len(corrections)} games requiring date correction.")

    if not corrections:
        logger.info("No corrections needed.")
        return

    # 2. Apply corrections in batches to avoid locking or unique constraint issues
    for corr in corrections:
        gid = corr["game_id"]
        old_d = corr["old_date"]
        new_d = corr["new_date"]

        logger.debug(f"Correcting game {gid}: {old_d} -> {new_d}")

        # Update game_results
        session.execute(
            update(GameResultORM)
            .where(GameResultORM.game_id == gid)
            .values(game_date=new_d)
        )

        # Update pitch_events
        session.execute(
            update(PitchEventORM)
            .where(PitchEventORM.game_id == gid)
            .values(game_date=new_d)
        )

        # Update statcast_raw
        session.execute(
            update(StatcastRawORM)
            .where(StatcastRawORM.game_pk == int(gid))
            .values(game_date=new_d)
        )

    logger.info(
        "Game-level dates corrected. Feature tables (Usage/Rolling) require localized re-backfill."
    )

    try:
        session.commit()
        logger.info("Successfully committed all date corrections.")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to commit corrections: {e}")
        raise


if __name__ == "__main__":
    fix_game_dates()
