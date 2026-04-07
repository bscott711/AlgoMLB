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

    # 5. Fetch Team Elo History
    elo_query = """
        SELECT game_pk, game_date, team_id, is_home, elo_pre, elo_post
        FROM team_elo_history
    """
    logger.info("Fetching team Elo history...")
    try:
        elo_df = pd.read_sql(elo_query, engine)
    except Exception:
        logger.warning("team_elo_history table not found. Run 'algomlb ml elo-backfill' first.")
        elo_df = pd.DataFrame()

    # 5b. Compute Pythagorean Expectation (from games_df — no extra DB query)
    from algomlb.ml.sabermetrics import compute_pythagorean_features, compute_re24_per_pa, compute_rolling_re24
    logger.info("Computing Pythagorean expectation features...")
    pythag_df = compute_pythagorean_features(games_df)

    # 5c. Compute RE24 from Retrosheet events
    re24_df = pd.DataFrame()
    try:
        retro_query = f"""
            SELECT game_id, date, inning, top_bot, outs_pre, outs_post,
                   br1_pre, br2_pre, br3_pre, br1_post, br2_post, br3_post,
                   runs, pa_flag, batter_id, pitcher_id, bat_team, pit_team
            FROM retrosheet_events
            WHERE EXTRACT(YEAR FROM date) IN ({years_str})
            AND pa_flag = 1
        """
        logger.info("Fetching Retrosheet events for RE24...")
        retro_df = pd.read_sql(retro_query, engine)
        if not retro_df.empty:
            re24_pa = compute_re24_per_pa(retro_df)
            re24_df = compute_rolling_re24(re24_pa, window=20)
            logger.info(f"RE24 computed: {len(re24_df)} rolling player-game rows.")
        else:
            logger.warning("No Retrosheet events found for RE24.")
    except Exception as e:
        logger.warning(f"RE24 computation skipped: {e}")

    lineup_count = len(lineups_df) if not lineups_df.empty else 0
    batter_count = len(batter_gold_df) if not batter_gold_df.empty else 0
    elo_count = len(elo_df) if not elo_df.empty else 0
    re24_count = len(re24_df) if not re24_df.empty else 0
    logger.info(f"Data loaded: {len(games_df)} games, {len(pitcher_gold_df)} pitcher, {lineup_count} lineups, {batter_count} batter, {elo_count} Elo, {re24_count} RE24")

    # 6. Build Uranium Matrix
    pipeline = FeaturePipeline()
    
    # Pass all available feature layers
    if not lineups_df.empty and not batter_gold_df.empty:
        X, y = pipeline.build_uranium_matrix(games_df, pitcher_gold_df, lineups_df, batter_gold_df, elo_df=elo_df, pythag_df=pythag_df, re24_df=re24_df)
    else:
        logger.warning("Lineup or batter data unavailable. Training with pitcher-only features.")
        X, y = pipeline.build_uranium_matrix(games_df, pitcher_gold_df, elo_df=elo_df, pythag_df=pythag_df, re24_df=re24_df)
    
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
    
    # 8. Train Model
    logger.info(f"Training XGBoost Uranium Model on {X_train.shape[0]} games, {X_train.shape[1]} features...")
    model = MLBModel(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.train(X_train, y_train, calibrate=True)

    # 9. Evaluation — use calibrated probabilities
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
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


@app.command(name="elo-backfill")
def elo_backfill(
    ctx: typer.Context,
) -> None:
    """Compute and persist Elo ratings from all completed game results."""
    from algomlb.ml.elo import backfill_team_elo_history

    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]

    logger.info("Starting Elo backfill from game_results...")
    backfill_team_elo_history(engine=engine)
    logger.success("Elo backfill complete.")


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


@app.command(name="walk-forward")
def walk_forward(
    ctx: typer.Context,
    start_year: int = typer.Option(2019, help="First training year"),
    end_year: int = typer.Option(2025, help="Last test year"),
    skip_2020: bool = typer.Option(True, help="Exclude 2020 COVID season"),
) -> None:
    """
    Walk-forward validation: iteratively expand training window and test on next year.

    Example with defaults:
        Fold 1: Train 2019        → Test 2021
        Fold 2: Train 2019,2021   → Test 2022
        Fold 3: Train 2019,2021-22 → Test 2023
        ...
    """
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from algomlb.ml.sabermetrics import compute_pythagorean_features, compute_re24_per_pa, compute_rolling_re24

    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]

    # Build list of valid seasons
    all_years = list(range(start_year, end_year + 1))
    if skip_2020:
        all_years = [y for y in all_years if y != 2020]

    if len(all_years) < 2:
        logger.error("Need at least 2 years for walk-forward validation.")
        raise typer.Exit(code=1)

    # ── Prefetch all data once ────────────────────────────────────────
    years_str = ",".join(map(str, all_years))

    games_query = f"""
        SELECT game_id, game_pk, game_date, home_team, away_team,
               home_pitcher_id, away_pitcher_id, home_score, away_score
        FROM game_results
        WHERE EXTRACT(YEAR FROM game_date) IN ({years_str})
        AND home_pitcher_id IS NOT NULL
        AND away_pitcher_id IS NOT NULL
        AND status = 'COMPLETED'
    """
    games_df = pd.read_sql(games_query, engine)
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])
    games_df["year"] = games_df["game_date"].dt.year

    pitcher_gold_df = pd.read_sql(
        f"SELECT * FROM player_rolling_features WHERE role = 'PITCHER' AND season IN ({years_str})",
        engine,
    )
    batter_gold_df = pd.read_sql(
        f"SELECT * FROM player_rolling_features WHERE role = 'BATTER' AND season IN ({years_str})",
        engine,
    )
    lineups_df = pd.read_sql(
        f"SELECT game_pk, game_date, team_side, batting_order, player_id FROM game_lineups WHERE EXTRACT(YEAR FROM game_date) IN ({years_str})",
        engine,
    )
    try:
        elo_df = pd.read_sql("SELECT game_pk, game_date, team_id, is_home, elo_pre, elo_post FROM team_elo_history", engine)
    except Exception:
        elo_df = pd.DataFrame()

    logger.info(f"Prefetched: {len(games_df)} games, {len(pitcher_gold_df)} pitcher, {len(batter_gold_df)} batter, {len(lineups_df)} lineups, {len(elo_df)} elo")

    # ── Pythagorean features (computed from game scores) ──────────────
    pythag_df = compute_pythagorean_features(games_df)

    # ── RE24 features (computed from Retrosheet) ──────────────────────
    re24_df = pd.DataFrame()
    try:
        retro_query = f"""
            SELECT game_id, date, inning, top_bot, outs_pre, outs_post,
                   br1_pre, br2_pre, br3_pre, br1_post, br2_post, br3_post,
                   runs, pa_flag, batter_id, pitcher_id, bat_team, pit_team
            FROM retrosheet_events
            WHERE EXTRACT(YEAR FROM date) IN ({years_str})
            AND pa_flag = 1
        """
        logger.info("Fetching Retrosheet events for RE24...")
        retro_df = pd.read_sql(retro_query, engine)
        if not retro_df.empty:
            re24_pa = compute_re24_per_pa(retro_df)
            re24_df = compute_rolling_re24(re24_pa, window=20)
            logger.info(f"RE24 computed: {len(re24_df)} rolling player-game rows.")
        else:
            logger.warning("No Retrosheet events found for RE24.")
    except Exception as e:
        logger.warning(f"RE24 computation skipped: {e}")

    # ── Walk-forward folds ────────────────────────────────────────────
    results_table: list[dict] = []

    for test_idx in range(1, len(all_years)):
        train_years = all_years[:test_idx]
        test_year = all_years[test_idx]

        logger.info(f"── Fold {test_idx}: Train {train_years} → Test {test_year} ──")

        # Filter data for this fold
        train_games = games_df[games_df["year"].isin(train_years)].copy()
        test_games = games_df[games_df["year"] == test_year].copy()
        fold_games = pd.concat([train_games, test_games], ignore_index=True)

        train_seasons = set(train_years)
        test_season = {test_year}
        fold_seasons = train_seasons | test_season

        fold_pitcher = pitcher_gold_df[pitcher_gold_df["season"].isin(fold_seasons)].copy()
        fold_batter = batter_gold_df[batter_gold_df["season"].isin(fold_seasons)].copy()

        fold_game_pks = set(fold_games["game_pk"].dropna().astype(int).tolist()) if "game_pk" in fold_games.columns else set()
        fold_lineups = lineups_df[lineups_df["game_pk"].isin(fold_game_pks)].copy() if not lineups_df.empty else pd.DataFrame()
        fold_elo = elo_df[elo_df["game_pk"].isin(fold_game_pks)].copy() if not elo_df.empty else pd.DataFrame()
        fold_pythag = pythag_df[pythag_df["game_pk"].isin(fold_game_pks)].copy() if not pythag_df.empty else pd.DataFrame()
        fold_re24 = re24_df  # RE24 is player-level, not game_pk filtered — rolling window handles temporal boundaries

        # Build matrix
        pipeline = FeaturePipeline()
        if not fold_lineups.empty and not fold_batter.empty:
            X, y = pipeline.build_uranium_matrix(
                fold_games, fold_pitcher, fold_lineups, fold_batter,
                elo_df=fold_elo, pythag_df=fold_pythag, re24_df=fold_re24,
            )
        else:
            X, y = pipeline.build_uranium_matrix(
                fold_games, fold_pitcher,
                elo_df=fold_elo, pythag_df=fold_pythag, re24_df=fold_re24,
            )

        if X.empty:
            logger.warning(f"Fold {test_idx}: Empty matrix, skipping.")
            continue

        # Split by year
        X["_year"] = pd.to_datetime(fold_games.loc[X.index, "game_date"]).dt.year
        train_mask = X["_year"].isin(train_years)
        test_mask = X["_year"] == test_year

        X_train = X[train_mask].drop(columns=["_year"])
        y_train = y[train_mask]
        X_test = X[test_mask].drop(columns=["_year"])
        y_test = y[test_mask]
        X = X.drop(columns=["_year"])

        if X_train.empty or X_test.empty:
            logger.warning(f"Fold {test_idx}: Train or test empty, skipping.")
            continue

        # Train
        model = MLBModel(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
        )
        model.train(X_train, y_train, calibrate=True)

        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        results_table.append({
            "fold": test_idx,
            "train_years": str(train_years),
            "test_year": test_year,
            "train_games": len(X_train),
            "test_games": len(X_test),
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "roc_auc": round(auc, 4),
        })

        logger.success(f"Fold {test_idx} ({test_year}): Acc={acc:.4f}  LL={ll:.4f}  AUC={auc:.4f}")

    # ── Summary Table ─────────────────────────────────────────────────
    if results_table:
        summary = pd.DataFrame(results_table)
        logger.success("Walk-Forward Validation Summary:")
        logger.info(f"\n{summary.to_string(index=False)}")

        avg_acc = summary["accuracy"].mean()
        avg_ll = summary["log_loss"].mean()
        avg_auc = summary["roc_auc"].mean()
        logger.success(f"Average: Acc={avg_acc:.4f}  LL={avg_ll:.4f}  AUC={avg_auc:.4f}")
    else:
        logger.error("No folds completed successfully.")


# Dummy use for deptry to ignore optuna
_ = optuna.Study
