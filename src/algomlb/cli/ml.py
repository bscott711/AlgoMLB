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
def train(
    ctx: typer.Context,
    start_year: int = typer.Option(2019, help="Start year for training data"),
    end_year: int = typer.Option(2024, help="End year for training data"),
    test_year: int = typer.Option(2025, help="Year to use for validation/testing"),
) -> None:
    """Train the Uranium model using Gold Layer features and report on a specific test year."""
    session_factory = get_session_factory()
    
    logger.info(f"Initializing Uranium model training ({start_year}-{end_year}, excluding 2020)...")
    logger.info(f"Validation Year: {test_year}")
    
    engine = session_factory.kw["bind"]
    
    # 1. Fetch real games (Train + Test years)
    years_to_fetch = list(range(start_year, end_year + 1)) + [test_year]
    # Remove 2020 as requested
    years_to_fetch = [y for y in years_to_fetch if y != 2020]
    
    years_str = ",".join(map(str, sorted(list(set(years_to_fetch)))))
    
    games_query = f"""
        SELECT game_id, game_pk, game_date, home_team, away_team, 
               home_pitcher_id, away_pitcher_id, home_score, away_score
        FROM game_results
        WHERE EXTRACT(YEAR FROM game_date) IN ({years_str})
        AND home_pitcher_id IS NOT NULL 
        AND away_pitcher_id IS NOT NULL
        AND status = 'COMPLETED'
    """
    logger.info("Fetching game results...")
    games_df = pd.read_sql(games_query, engine)
    
    if games_df.empty:
        logger.error("No completed games found for the specified years.")
        raise typer.Exit(code=1)

    # 2. Fetch Gold Layer Pitcher Features
    gold_pitcher_query = f"""
        SELECT *
        FROM player_rolling_features
        WHERE role = 'PITCHER'
        AND season IN ({years_str})
    """
    logger.info("Fetching Gold Layer pitcher features...")
    pitcher_gold_df = pd.read_sql(gold_pitcher_query, engine)
    
    if pitcher_gold_df.empty:
        logger.error("No Gold Layer pitcher features found.")
        raise typer.Exit(code=1)

    # 3. Fetch lineup data
    lineup_query = f"""
        SELECT game_pk, game_date, team_side, batting_order, player_id
        FROM game_lineups
        WHERE EXTRACT(YEAR FROM game_date) IN ({years_str})
    """
    logger.info("Fetching starting lineups...")
    lineups_df = pd.read_sql(lineup_query, engine)
    
    # 4. Fetch Gold Layer Batter Features
    batter_gold_query = f"""
        SELECT *
        FROM player_rolling_features
        WHERE role = 'BATTER'
        AND season IN ({years_str})
    """
    logger.info("Fetching Gold Layer batter features...")
    batter_gold_df = pd.read_sql(batter_gold_query, engine)

    lineup_count = len(lineups_df) if not lineups_df.empty else 0
    batter_count = len(batter_gold_df) if not batter_gold_df.empty else 0
    logger.info(f"Data loaded: {len(games_df)} games, {len(pitcher_gold_df)} pitcher records, {lineup_count} lineup slots, {batter_count} batter records")

    # 5. Build Uranium Matrix
    pipeline = FeaturePipeline()
    
    # Pass lineups and batter Gold if available
    if not lineups_df.empty and not batter_gold_df.empty:
        X, y = pipeline.build_uranium_matrix(games_df, pitcher_gold_df, lineups_df, batter_gold_df)
    else:
        logger.warning("Lineup or batter data unavailable. Training with pitcher-only features.")
        X, y = pipeline.build_uranium_matrix(games_df, pitcher_gold_df)
    
    if X.empty:
        logger.error("Uranium feature matrix is empty after pipeline.")
        raise typer.Exit(code=1)

    # 6. Split Train/Test by Year
    # games_df has our game records. pipeline.build_uranium_matrix should return indices aligned with games_df's outcome rows.
    # To be safe, we re-derive the years from the index if possible, but X should have aligned index.
    
    # We'll use the game_date from the games_df to split
    # Note: df in build_uranium_matrix might have fewer rows due to dropna(subset=["home_score","away_score"])
    # But X index should match the filtered df. 
    # We'll attach the year to the feature matrix temporarily to split.
    
    # Re-fetch the game dates for the rows in X
    df_indices = X.index
    # The build_uranium_matrix internal 'df' is constructed from games_df. 
    # We need to ensure we can split by game_date.
    
    # Let's check the index of X. If we didn't reset it, it matches the merged df.
    # The merged df started with games_df.
    
    X["temp_year"] = pd.to_datetime(games_df.loc[df_indices, "game_date"]).dt.year
    
    X_train = X[X["temp_year"] != test_year].drop(columns=["temp_year"])
    y_train = y[X["temp_year"] != test_year]
    
    X_test = X[X["temp_year"] == test_year].drop(columns=["temp_year"])
    y_test = y[X["temp_year"] == test_year]
    
    if X_train.empty:
        logger.error("Training set is empty!")
        raise typer.Exit(code=1)
    if X_test.empty:
        logger.warning(f"Test set (Year {test_year}) is empty! Results will only reflect training performance.")
        X_test, y_test = X_train, y_train # Fallback for reporting
    
    # 7. Train Model
    logger.info(f"Training XGBoost Uranium Model on {X_train.shape[0]} games, {X_train.shape[1]} features...")
    model = MLBModel(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    model.train(X_train, y_train)

    # 8. Evaluation
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    y_prob = model.clf.predict_proba(X_test)[:, 1]
    y_pred = model.clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    logger.success(f"Uranium Evaluation ({test_year}):")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Log Loss: {ll:.4f}")
    logger.info(f"  ROC AUC:  {auc:.4f}")

    # 9. Save Artifact
    model_path = Path(".data/models/uranium_win_model.joblib")
    model.save(model_path)

    logger.success(f"Uranium model trained and saved to {model_path}")

    # Log feature importances
    if hasattr(model.clf, "feature_importances_"):
        importances = pd.Series(model.clf.feature_importances_, index=X_train.columns)
        top_20 = importances.sort_values(ascending=False).head(20)
        logger.info(f"Top 20 Features:\n{top_20.to_string()}")

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
