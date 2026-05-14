import datetime
import os
import subprocess
import sys
from loguru import logger
from algomlb.db.session import get_session_factory
from algomlb.strategy.betting_service import BettingService

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<g>{time:HH:mm:ss}</g> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

def run_command(command: list[str], description: str) -> bool:
    """Helper to run shell commands and log status."""
    logger.info(f"🚀 {description}...")
    try:
        # We use uv run to ensure the virtualenv is used correctly
        full_command = ["uv", "run"] + command
        result = subprocess.run(full_command, check=True, capture_output=True, text=True)
        logger.success(f"✅ {description} complete.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed!")
        logger.error(f"Error: {e.stderr}")
        return False

def place_today_bets():
    """Finds +EV bets and populates the bankroll_ledger based on Uranium sims."""
    today = datetime.date.today()
    logger.info(f"📈 Analyzing +EV edges for {today}...")
    
    session = get_session_factory()()
    try:
        service = BettingService(session)
        # This will compare the fresh Uranium sims with the latest odds pulled in sync-daily
        placed = service.place_daily_bets(today)
        if placed > 0:
            logger.success(f"✅ Strategy: Locked in {placed} +EV bets as PENDING in the ledger.")
        else:
            logger.warning("Strategy: No edges found above the threshold today.")
        return placed
    except Exception as e:
        logger.error(f"❌ Failed to place bets: {e}")
        return 0
    finally:
        session.close()

def main():
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    logger.info(f"🏟️ --- AlgoMLB MASTER PIPELINE START [{today_str}] --- 🏟️")

    # 1. Sync Everything (Lineups, Statcast, Weather, Odds, Settlement)
    # This gets the board ready for simulation.
    if not run_command(["algomlb", "sync", "daily"], "Syncing data & settling yesterday"):
        logger.warning("Pipeline continuing despite minor sync issues...")

    # 2. Run Uranium Monte Carlo simulations
    # This generates the "True Probabilities" we need for strategy.
    if not run_command(["python", "-m", "src.algomlb.cli.main", "ml", "sim-day", today_str], "Running Uranium MC Simulations"):
        logger.error("🛑 CRITICAL: Simulations failed. Strategy layer cannot proceed.")
        return

    # 3. Strategy Layer: Find +EV Edges and lock into Ledger
    # This bridges the model output to the bankroll.
    placed_count = place_today_bets()

    # 4. Social Layer: FadeGoblin Timeline Post
    if placed_count > 0:
        logger.info(f"👺 Found {placed_count} targets. Alerting the Goblin...")
        # uv run bot --mode sniper will pick up the PENDING bets we just created
        if run_command(["bot", "--mode", "sniper"], "FadeGoblin Sniper Post"):
            logger.success("🚀 Social post is LIVE on Bluesky!")
        else:
            logger.error("❌ Social post failed.")
    else:
        logger.info("💤 No +EV targets found. Goblin is staying in the cave.")

    logger.info(f"🏁 --- AlgoMLB DAILY PIPELINE COMPLETE --- 🏁")

if __name__ == "__main__":
    main()
