import sys
from algomlb.db.session import get_engine
from algomlb.ml.sabermetrics import backfill_team_sabermetrics_history
from algomlb.ml.elo import backfill_team_elo_history
from loguru import logger

logger.info("Starting Emergency Sabermetrics + Elo Backfill for 2026...")

engine = get_engine()

try:
    logger.info("Running Elo Backfill...")
    backfill_team_elo_history(engine=engine)
    logger.success("Elo Backfill Complete.")

    logger.info("Running Sabermetrics Backfill...")
    backfill_team_sabermetrics_history(engine_in=engine)
    logger.success("Sabermetrics Backfill Complete.")

except Exception as e:
    logger.exception(f"Backfill failed: {e}")
    sys.exit(1)
