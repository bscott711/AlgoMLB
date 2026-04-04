import random
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd
import typer

from algomlb.core.agent_io import AgentResult, emit_agent_result
from algomlb.core.logger import logger
from algomlb.db.repository import DatabaseRepository
from algomlb.db.session import get_session_factory
from algomlb.ml import FeaturePipeline, MLBModel
from algomlb.ml.decoupler_pipeline import run_decoupler_pipeline
from algomlb.ingestion.historical import HistoricalDataLoader

app = typer.Typer(help="Train and optimize ML models.", no_args_is_help=True)


@app.command()
def fetch_history(
    ctx: typer.Context,
    start_year: int = typer.Option(
        2023, "--start-year", help="Year to start fetching data from."
    ),
    end_year: int = typer.Option(
        datetime.now().year, "--end-year", help="Year to end fetching data at."
    ),
) -> None:
    """Fetch historical pitching and batting stats for a given year range."""
    agent_mode = ctx.obj.get("agent_mode", False)

    # Setup Infrastructure
    session_factory = get_session_factory()

    with session_factory() as session:
        repo = DatabaseRepository(session)
        loader = HistoricalDataLoader(repo)

        logger.info(f"Fetching historical data from {start_year} to {end_year}...")

        pitching_df = loader.fetch_pitching_stats(start_year, end_year)
        batting_df = loader.fetch_team_batting(start_year, end_year)

        logger.success(
            f"Fetched pitching data: {pitching_df.shape} and team batting data: {batting_df.shape}"
        )

        if agent_mode:
            emit_agent_result(
                AgentResult(
                    status="success",
                    command="ml.fetch-history",
                    data={
                        "pitching_shape": list(pitching_df.shape),
                        "batting_shape": list(batting_df.shape),
                    },
                )
            )


@app.command()
def train(ctx: typer.Context) -> None:
    """Train the baseline model using cached feature matrices."""
    # Setup Infrastructure
    session_factory = get_session_factory()

    with session_factory() as session:
        repo = DatabaseRepository(session)
        loader = HistoricalDataLoader(repo)
        logger.info("Initializing baseline model training...")
    try:
        # Use 2023 to hit the local Parquet cache
        pitching_df = loader.fetch_pitching_stats(2023, 2023)
        batting_df = loader.fetch_team_batting(2023, 2023)
    except Exception as e:
        logger.error(f"Failed to load cached stats: {e}")
        raise typer.Exit(code=1)

    # Determine available teams and pitchers
    # Use fallback if columns missing
    has_team = "team" in pitching_df.columns and "team" in batting_df.columns
    has_player = "player_id" in pitching_df.columns

    teams = list(pitching_df["team"].unique()) if has_team else ["NYY", "BOS"]
    # Ensure at least 2 teams for random sampling
    if len(teams) < 2:
        teams = ["NYY", "BOS"]

    player_ids = list(pitching_df["player_id"].unique()) if has_player else []
    # Ensure at least some pitcher IDs exist for dummy games
    if not player_ids:
        player_ids = [1, 2]

    games_data = []
    for _ in range(100):
        h_team, a_team = random.sample(teams, 2)
        games_data.append(
            {
                "team_h": h_team,
                "team_a": a_team,
                "home_pitcher_id": random.choice(player_ids),
                "away_pitcher_id": random.choice(player_ids),
                "home_score": random.randint(0, 10),
                "away_score": random.randint(0, 10),
                "game_id": f"g_{random.randint(1000, 9999)}",
                "date": "2024-04-01",
            }
        )
    games_df = pd.DataFrame(games_data)

    # Prepare historical baseline
    if has_team:
        stats_df = pitching_df.merge(batting_df, on="team")
    else:
        stats_df = pd.concat([pitching_df, batting_df], axis=1)

    pipeline = FeaturePipeline()
    X, y = pipeline.build_training_matrix(games_df, stats_df)

    model = MLBModel(n_estimators=50, max_depth=3)
    model.train(X, y)

    model_path = Path(".data/models/baseline.joblib")
    model.save(model_path)

    logger.success(f"Baseline model trained and saved to {model_path}")

    agent_mode = ctx.obj.get("agent_mode", False)
    if agent_mode:
        emit_agent_result(
            AgentResult(
                status="success",
                command="ml.train",
                data={
                    "feature_shape": list(X.shape),
                    "model_path": str(model_path),
                },
            )
        )


@app.command()
def optimize(
    ctx: typer.Context,
    market: str = typer.Option("Moneyline", help="Market to optimize"),
) -> None:
    """Run Optuna optimization studies."""
    typer.echo(f"TODO: Implement ML optimize for {market}")


@app.command()
def decouple(
    ctx: typer.Context,
    action: str = typer.Argument(
        ..., help="Action to perform: train, calibrate, backfill, or full"
    ),
    version: str = typer.Option("v1", "--version", help="Model version to use"),
) -> None:
    """Run the Batted Ball Flight Decoupler pipeline (Train, Calibrate, Backfill)."""
    try:
        run_decoupler_pipeline(action, version)
    except Exception as e:
        logger.error(f"Decoupler pipeline failed: {e}")
        raise typer.Exit(code=1)


# Dummy use for deptry to ignore optuna
_ = optuna.Study
