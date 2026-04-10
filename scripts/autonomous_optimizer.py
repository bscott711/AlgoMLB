"""
Autonomous Model Optimization Runner for AlgoMLB.
Sequentially tunes, backtests, and explains component models.

Usage:
  uv run scripts/autonomous_optimizer.py --trials 100 --range "2021,2022,2023,2024,2025"
"""

import subprocess
import os
import json
import argparse
from datetime import datetime
from algomlb.core.logger import logger

# Prioritized list of Uranium-level targets
REGISTRY = [
    {"target": "home_win", "version": "v1.0", "description": "Moneyline Model"},
    {"target": "total_runs_actual", "version": "v1.0", "description": "O/U Model"},
    {
        "target": "pitcher_strikeouts",
        "version": "v1.0",
        "description": "Pitcher Prop Head",
    },
    {"target": "batter_hits", "version": "v1.0", "description": "Batter Prop Head"},
    {
        "target": "pitcher_outs_recorded",
        "version": "v1.0",
        "description": "Pitcher Prop Head",
    },
]

LOG_FILE = "models/optimization_history.json"


def run_command(cmd_list):
    """Executes a command and returns completion status."""
    logger.info(f"Executing: {' '.join(cmd_list)}")
    try:
        result = subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        # Log to a persistent error file
        with open("models/error_traceback.log", "a") as f:
            f.write(f"\n--- FAILED: {' '.join(cmd_list)} ---\n")
            f.write(e.stderr)
            f.write("\n-----------------------------------\n")
        return False, e.stderr


def log_progress(target, step, status, details=None):
    """Persists progress to a JSON log to survive session resets."""
    history = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    history.append(
        {
            "timestamp": datetime.now().isoformat(),
            "target": target,
            "step": step,
            "status": status,
            "details": details,
        }
    )

    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--range", type=str, default="2021,2022,2023,2024,2025")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("📡 Starting Autonomous Optimization Run...")

    if not os.path.exists("models"):
        os.makedirs("models")

    for job in REGISTRY:
        target = job["target"]
        version = job["version"]

        logger.info(f"🚀 Processing Target: {target} ({job['description']})")

        if args.dry_run:
            logger.info(f"[Dry Run] would optimize {target}")
            continue

        # 1. TUNE
        success, output = run_command(
            [
                "uv",
                "run",
                "algomlb",
                "ml",
                "tune",
                "--target",
                target,
                "--trials",
                str(args.trials),
                "--version",
                version,
            ]
        )
        if not success:
            log_progress(target, "tune", "failed", output)
            continue
        log_progress(target, "tune", "success")

        # 2. BACKTEST
        success, output = run_command(
            [
                "uv",
                "run",
                "algomlb",
                "ml",
                "backtest",
                "--target",
                target,
                "--version",
                version,
            ]
        )
        if not success:
            log_progress(target, "backtest", "failed", output)
            continue
        log_progress(target, "backtest", "success")

        # 3. EXPLAIN
        success, output = run_command(
            [
                "uv",
                "run",
                "algomlb",
                "ml",
                "explain",
                "--target",
                target,
                "--version",
                version,
            ]
        )
        if not success:
            log_progress(target, "explain", "failed", output)
            continue
        log_progress(target, "explain", "success")

        # 4. COMMIT PROGRESS
        run_command(["git", "add", "models/optuna_history.db", LOG_FILE])
        run_command(
            [
                "git",
                "commit",
                "-m",
                f"feat(ml): optimized {target} with {args.trials} trials",
            ]
        )

        logger.success(f"✅ Successfully optimized and committed {target}")

    logger.success("🏆 All Uranium-level targets optimized!")


if __name__ == "__main__":
    main()
