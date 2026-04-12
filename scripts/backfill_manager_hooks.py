from algomlb.db.session import get_engine
from algomlb.ml.hooks import backfill_hook_events
from algomlb.core.logger import logger


def run_backfill():
    """Trigger the manager hook backfill pipeline."""
    engine = get_engine()

    logger.info("🚀 Starting Manager Hook Backfill (Retrosheet -> Hooks)...")
    try:
        # Defaults to 2019-2025 (excluding 2020) as per hooks.py implementation
        backfill_hook_events(engine)
        logger.success("🏁 Manager Hook Backfill complete!")
    except Exception as e:
        logger.error(f"❌ Backfill failed: {e}")
        raise


if __name__ == "__main__":
    run_backfill()
