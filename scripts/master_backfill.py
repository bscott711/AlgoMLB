import subprocess
import sys
from loguru import logger
import time

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

def run(cmd):
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"Command failed with code {result.returncode}")
    return result.returncode

def main():
    logger.info("Starting Master Backfill (Bronze -> Silver -> Gold -> Uranium)")
    start_time = time.time()

    # 1. Bronze (Sync 2026 to ensure fresh start)
    # logger.info("PHASE 1: Refreshing recent Bronze data...")
    # run(["uv", "run", "algomlb", "ingest", "statcast", "--start", "2026-03-01", "--end", "2026-04-12"])

    # 2. Silver (Enrich with Specialty Metrics like fb_speed)
    # logger.info("PHASE 2: Backfilling Silver Layer (Game Logs)...")
    # for yr in YEARS:
    #    logger.info(f"--- Processing Silver for {yr} ---")
    #    run(["uv", "run", "algomlb", "process", "silver", "--year", str(yr)])

    # 3. Gold (Feature Engineering loop)
    # Rolling needs the full history in order to compute EMAs and deltas correctly.
    # logger.info("PHASE 3: Backfilling Gold Layer (Rolling Features)...")
    # run(["uv", "run", "algomlb", "process", "rolling", "--start", "2019-01-01", "--end", "2026-12-31"])

    # Quant features (At-bat level)
    # logger.info("PHASE 3b: Backfilling Gold Layer (Quant Features)...")
    # for yr in YEARS:
    #     logger.info(f"--- Processing Quant for {yr} ---")
    #     run(["uv", "run", "algomlb", "process", "quant", "--start-date", f"{yr}-03-01", "--end-date", f"{yr}-11-15"])

    # 4. Uranium (Training state and Backtesting)
    logger.info("PHASE 4: Backfilling Uranium Layer (ML Models)...")
    run(["uv", "run", "algomlb", "ml", "elo-backfill"])
    
    targets = ["home_win", "total_runs_actual", "pa_outcome", "is_strike"]
    version = "v1.0"

    # Step A: Tune (To generate param files)
    for target in targets:
        logger.info(f"--- Tuning target: {target} ---")
        run(["uv", "run", "algomlb", "ml", "tune", "--target", target, "--version", version, "--trials", "20"])

    # Step B: Backtest
    for target in targets:
        logger.info(f"--- Backtesting target: {target} ---")
        run(["uv", "run", "algomlb", "ml", "backtest", "--target", target, "--version", version])

        # Step C: Train
        logger.info(f"--- Training Production Model: {target} ---")
        run(["uv", "run", "algomlb", "ml", "train", "--target", target, "--version", version])

    duration = (time.time() - start_time) / 3600
    logger.success(f"MASTER BACKFILL COMPLETE in {duration:.2f} hours")

if __name__ == "__main__":
    main()
