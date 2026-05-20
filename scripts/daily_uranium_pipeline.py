import datetime
import os
import subprocess
import sys
from dotenv import load_dotenv
from loguru import logger
from algomlb.db.session import get_session_factory
from algomlb.strategy.betting_service import BettingService
from algomlb.db.models import BankrollLedgerORM
from algomlb.domain import TransactionStatus

# Load project .env at the start
load_dotenv()

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<g>{time:HH:mm:ss}</g> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
)


def run_command(command: list[str], description: str, background: bool = False) -> bool:
    """Helper to run shell commands and log status with environment passthrough."""
    logger.info(f"🚀 {description}...")
    try:
        # Explicitly pass current environment (including loaded .env)
        env = os.environ.copy()

        if (
            command[0] == "python"
            or command[0].endswith("python")
            or "xvfb-run" in command[0]
        ):
            full_command = command
        else:
            full_command = ["uv", "run"] + command

        if background:
            subprocess.Popen(
                full_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            logger.info(f"🛰️ {description} sent to background.")
            return True
        else:
            result = subprocess.run(
                full_command, check=True, capture_output=True, text=True, env=env
            )
            # Log the inner output for debugging if it's the bot
            if "main.py" in command[1]:
                logger.debug(f"Bot Output:\n{result.stdout}")
            logger.success(f"✅ {description} complete.")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed!")
        logger.error(f"Error: {e.stderr}")
        return False


def count_pending_bets():
    """Counts how many bets are currently PENDING in the ledger."""
    session = get_session_factory()()
    try:
        count = (
            session.query(BankrollLedgerORM)
            .filter(BankrollLedgerORM.status == TransactionStatus.PENDING)
            .count()
        )
        return count
    finally:
        session.close()


def place_today_bets():
    """Finds +EV bets using FAST TOP-DOWN predictions and populates the ledger."""
    today = datetime.date.today()
    logger.info(f"📈 Analyzing +EV edges for {today} [FAST TOP-DOWN]...")

    session = get_session_factory()()
    try:
        service = BettingService(session)
        placed = service.place_daily_bets(today)
        if placed > 0:
            logger.success(
                f"✅ Strategy: Locked in {placed} NEW +EV bets as PENDING in the ledger."
            )
        else:
            logger.info("Strategy: No new edges found (or already placed).")
        return placed
    except Exception as e:
        logger.error(f"❌ Failed to place bets: {e}")
        return 0
    finally:
        session.close()


def main():
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    logger.info(f"🏟️ --- AlgoMLB FAST PIPELINE START [{today_str}] --- 🏟️")

    # 1. Sync Daily Data & Settle (Lineups, Weather, Odds)
    if not run_command(
        ["algomlb", "sync", "daily"], "Syncing data & settling yesterday"
    ):
        logger.warning("Pipeline continuing despite minor sync issues...")

    # 2. Process Silver Layer (Statcast → Player Game Logs)
    #    Rolling features depend on this upstream data.
    if not run_command(
        ["algomlb", "process", "silver", "--incremental"],
        "Processing Silver Layer (incremental)",
    ):
        logger.warning("⚠️ Silver processing had issues. Rolling features may be stale.")

    # 3. Refresh Rolling Features (CRITICAL for model accuracy)
    #    Without this, pitcher/batter features go stale and the model predicts on zeros.
    if not run_command(
        ["algomlb", "process", "rolling", "--date", today_str],
        "Computing rolling features for today",
    ):
        logger.error("⚠️ Rolling features failed! Model predictions may be degraded.")

    # 4. Strategy Layer: Find +EV Edges using Fast Top-Down Models
    place_today_bets()

    # 4. Check for ANY Pending Bets (New or Existing)
    pending_count = count_pending_bets()

    # 5. Social Layer: FadeGoblin Timeline Post
    if pending_count > 0:
        logger.info(
            f"👺 Found {pending_count} pending targets. Triggering the Goblin..."
        )
        # Point to the local main.py file and set PYTHONPATH explicitly in the env
        os.environ["PYTHONPATH"] = "/home/opc/AlgoMLB/fadegoblin_playwright/src"
        local_python = "/home/opc/AlgoMLB/.venv/bin/python"
        bot_script = "/home/opc/AlgoMLB/fadegoblin_playwright/src/fadegoblin/main.py"

        # We wrap ExecStart in xvfb-run to provide a virtual frame buffer for browser automation
        bot_cmd = [
            "/usr/bin/xvfb-run",
            "--auto-servernum",
            "--server-args=-screen 0 1280x800x24",
            local_python,
            bot_script,
            "--mode",
            "sniper",
        ]

        if run_command(bot_cmd, "FadeGoblin Sniper Post"):
            logger.success("🚀 Social post is LIVE on Bluesky and Twitter!")
        else:
            logger.error("❌ Social post failed.")
    else:
        logger.info("💤 No +EV targets found in ledger. Goblin is staying in the cave.")

    logger.info("🏁 --- AlgoMLB FAST PIPELINE COMPLETE --- 🏁")


if __name__ == "__main__":
    main()
