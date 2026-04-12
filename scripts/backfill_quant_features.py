from datetime import date
from typing import List

from sqlalchemy import text
from loguru import logger

from algomlb.db.session import get_engine
from algomlb.ml.quant_processor import process_quant_for_date


def get_unique_dates(engine) -> List[date]:
    """Fetch all distinct game dates from statcast_raw, ordered."""
    query = text("SELECT DISTINCT game_date FROM statcast_raw ORDER BY game_date")
    with engine.connect() as conn:
        res = conn.execute(query).scalars().all()
        return [d if isinstance(d, date) else date.fromisoformat(str(d)) for d in res]


def backfill_quant_features(start_date: date = None, end_date: date = None):
    """Iterate through dates and trigger quant feature processing."""
    engine = get_engine()
    all_dates = get_unique_dates(engine)

    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]

    logger.info(f"🚀 Starting Quant Feature Backfill for {len(all_dates)} dates...")

    total_rows = 0
    for gd in all_dates:
        try:
            count = process_quant_for_date(gd, engine=engine)
            total_rows += count
            if count > 0:
                logger.info(
                    f"  Processed {gd}: {count} rows. (Running Total: {total_rows})"
                )
        except Exception as e:
            logger.error(f"  ❌ Failed to process {gd}: {e}")

    logger.success(
        f"🏁 Backfill complete! Total quant features populated: {total_rows}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill Statcast Quant Features")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    s_dt = date.fromisoformat(args.start) if args.start else None
    e_dt = date.fromisoformat(args.end) if args.end else None

    backfill_quant_features(start_date=s_dt, end_date=e_dt)
