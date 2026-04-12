import subprocess
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
import pandas as pd
from algomlb.db.session import get_engine


def run_simulation(game_pk, trials, version):
    """Worker function to run a single game simulation."""
    cmd = [
        "uv",
        "run",
        "algomlb",
        "ml",
        "simulate-game",
        "--game-pk",
        str(game_pk),
        "--trials",
        str(trials),
        "--version",
        version,
    ]
    logger.info(f"Starting simulation for game_pk: {game_pk}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.success(f"Successfully simulated {game_pk}")
    else:
        logger.error(f"Failed to simulate {game_pk}: {result.stderr}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Parallel batch simulation for AlgoMLB."
    )
    parser.add_argument("--year", type=int, default=2025, help="Season to simulate.")
    parser.add_argument(
        "--trials", type=int, default=1000, help="Number of trials per game."
    )
    parser.add_argument("--version", type=str, default="v1.0", help="Model version.")
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count() - 1,
        help="Number of parallel workers.",
    )
    args = parser.parse_args()

    engine = get_engine()

    # Query games for the year that don't have props yet
    query = f"""
        SELECT gr.game_id as game_pk 
        FROM game_results gr
        LEFT JOIN (
            SELECT DISTINCT game_pk FROM uranium_simulated_player_props
        ) upp ON CAST(gr.game_id AS BIGINT) = upp.game_pk
        WHERE EXTRACT(YEAR FROM gr.game_date) = {args.year}
        AND gr.status = 'COMPLETED'
        AND upp.game_pk IS NULL
        ORDER BY gr.game_date DESC
    """

    games_df = pd.read_sql(query, engine)
    game_pks = games_df["game_pk"].tolist()

    logger.info(f"Found {len(game_pks)} games needing simulation for {args.year}.")

    if not game_pks:
        logger.info("Nothing to simulate.")
        return

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_simulation, pk, args.trials, args.version): pk
            for pk in game_pks
        }

        for future in as_completed(futures):
            pk = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Simulation task for {pk} raised an exception: {e}")


if __name__ == "__main__":
    main()
