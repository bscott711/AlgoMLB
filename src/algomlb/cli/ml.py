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


def _load_ml_data(engine, years_str: str) -> dict[str, pd.DataFrame]:
    """Prefetch all data layers required for Uranium training/optimization."""
    logger.info("Fetching Uranium data layers...")
    games_df = pd.read_sql(
        f"""
        SELECT game_id as game_pk, game_date, home_team, away_team, 
               home_pitcher_id, away_pitcher_id, home_score, away_score
        FROM game_results
        WHERE EXTRACT(YEAR FROM game_date) IN ({years_str})
        AND home_pitcher_id IS NOT NULL AND away_pitcher_id IS NOT NULL AND status = 'COMPLETED'
    """,
        engine,
    )

    games_df["game_date"] = pd.to_datetime(games_df["game_date"])
    games_df["year"] = games_df["game_date"].dt.year

    pitcher_gold = pd.read_sql(
        f"SELECT * FROM player_rolling_features WHERE role = 'PITCHER' AND season IN ({years_str})",
        engine,
    )
    lineups = pd.read_sql(
        f"SELECT game_pk, game_date, team_side, batting_order, player_id FROM game_lineups WHERE EXTRACT(YEAR FROM game_date) IN ({years_str})",
        engine,
    )
    batter_gold = pd.read_sql(
        f"SELECT * FROM player_rolling_features WHERE role = 'BATTER' AND season IN ({years_str})",
        engine,
    )

    try:
        elo_df = pd.read_sql(
            "SELECT game_pk, game_date, team_id, is_home, elo_pre, elo_post FROM team_elo_history",
            engine,
        )
    except Exception:
        logger.warning("team_elo_history not found.")
        elo_df = pd.DataFrame()

    from algomlb.ml.sabermetrics import (
        compute_pythagorean_features,
        compute_re24_per_pa,
        compute_rolling_re24,
    )

    pythag_df = compute_pythagorean_features(games_df)

    re24_df = pd.DataFrame()
    try:
        retro_df = pd.read_sql(
            f"""
            SELECT game_id, date, inning, top_bot, outs_pre, outs_post,
                   br1_pre, br2_pre, br3_pre, br1_post, br2_post, br3_post,
                   runs, pa_flag, batter_id, pitcher_id, bat_team, pit_team
            FROM retrosheet_events
            WHERE EXTRACT(YEAR FROM date) IN ({years_str}) AND pa_flag = 1
        """,
            engine,
        )
        if not retro_df.empty:
            re24_df = compute_rolling_re24(compute_re24_per_pa(retro_df), window=20)
    except Exception as e:
        logger.warning(f"RE24 computation skipped: {e}")

    return {
        "games": games_df,
        "pitcher_gold": pitcher_gold,
        "lineups": lineups,
        "batter_gold": batter_gold,
        "elo": elo_df,
        "pythag": pythag_df,
        "re24": re24_df,
    }


def _evaluate_and_report(
    model, X_test, y_test, test_year, engine, model_version, start_year, end_year
):
    """Run metrics, importance, SHAP, and persistence."""
    from algomlb.ml.eval import (
        compute_fold_metrics,
        compute_calibration_bins,
        persist_eval_results,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_fold_metrics(y_test, y_prob)

    logger.success(
        f"Uranium Eval ({test_year}): Acc={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f} LL={metrics['log_loss']:.4f}"
    )

    # Feature Importance
    impl_df = (
        model.get_feature_importance()
        .sort_values(by="importance", ascending=False)
        .head(10)
    )
    for _, r in impl_df.iterrows():
        logger.info(f"  {r['feature']:<30} : {r['importance']:.4f}")

    # Persistence
    try:
        persist_eval_results(
            engine=engine,
            model_version=model_version,
            test_year=test_year,
            train_start=start_year,
            train_end=end_year,
            n_games=len(X_test),
            metrics=metrics,
            cal_bins=compute_calibration_bins(y_test, y_prob, n_bins=20),
        )
    except Exception as e:
        logger.warning(f"Eval persistence failed: {e}")

    return metrics


@app.command()
def train(
    ctx: typer.Context,
    start_year: int = typer.Option(2019, help="Start year for training data"),
    end_year: int = typer.Option(2024, help="End year for training data"),
    test_year: int = typer.Option(2025, help="Year to use for validation/testing"),
    model_version: str = typer.Option("v0.1", help="Model version label"),
) -> None:
    """Train the Uranium model using Gold Layer features."""
    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]
    years_to_fetch = [y for y in range(start_year, end_year + 1) if y != 2020] + [
        test_year
    ]
    years_str = ",".join(map(str, sorted(list(set(years_to_fetch)))))

    data = _load_ml_data(engine, years_str)
    if data["games"].empty:
        logger.error("No games found.")
        raise typer.Exit(1)

    # Build Matrix
    pipeline = FeaturePipeline()
    X, y = pipeline.build_uranium_matrix(
        data["games"],
        data["pitcher_gold"],
        data["lineups"] if not data["lineups"].empty else None,
        data["batter_gold"] if not data["batter_gold"].empty else None,
        elo_df=data["elo"],
        pythag_df=data["pythag"],
        re24_df=data["re24"],
    )

    if X.empty:
        logger.error("Empty matrix.")
        raise typer.Exit(1)

    # Split
    X["_yr"] = pd.to_datetime(data["games"].loc[X.index, "game_date"]).dt.year
    X_train, y_train = (
        X[X["_yr"] != test_year].drop(columns=["_yr"]),
        y[X["_yr"] != test_year],
    )
    X_test, y_test = (
        X[X["_yr"] == test_year].drop(columns=["_yr"]),
        y[X["_yr"] == test_year],
    )

    # Train
    from algomlb.ml.hyperopt import load_optimized_params

    model = MLBModel(**load_optimized_params(model_version))
    model.train(X_train, y_train, calibrate=True)

    # Evaluate
    _evaluate_and_report(
        model,
        X_test if not X_test.empty else X_train,
        y_test if not y_test.empty else y_train,
        test_year,
        engine,
        model_version,
        start_year,
        end_year,
    )

    # Save
    model_path = Path(".data/models/uranium_win_model.joblib")
    model.save(model_path)
    logger.success(f"Model saved to {model_path}")

    if ctx.obj.get("agent_mode", False):
        emit_agent_result(
            AgentResult(
                status="success",
                command="ml.train",
                data={"model_path": str(model_path)},
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


@app.command(name="build-registry")
def build_registry(
    ctx: typer.Context,
    start_year: int = typer.Option(2019, help="First season to process"),
    end_year: int = typer.Option(2026, help="Last season to process"),
) -> None:
    """Map retrosheet IDs to game_pk and resolve manager tenure."""
    from algomlb.ml.registry import build_manager_registry
    from algomlb.db.session import get_session_factory

    session_factory = get_session_factory()
    with session_factory() as session:
        build_manager_registry(session, start_year=start_year, end_year=end_year)


@app.command(name="hook-backfill")
def hook_backfill(
    ctx: typer.Context,
    start_year: int = typer.Option(2019, help="First season to process"),
    end_year: int = typer.Option(2025, help="Last season to process"),
) -> None:
    """Extract manager hook events from retrosheet and compute hook profiles."""
    from algomlb.ml.hooks import backfill_hook_events

    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]

    backfill_hook_events(engine, start_year=start_year, end_year=end_year)


@app.command()
def optimize(
    ctx: typer.Context,
    start_year: int = typer.Option(2019, help="First training year"),
    end_year: int = typer.Option(2025, help="Last test year"),
    skip_2020: bool = typer.Option(True, help="Exclude 2020 COVID season"),
    n_trials: int = typer.Option(50, help="Number of Optuna trials"),
    model_version: str = typer.Option("v0.1", help="Model version label"),
) -> None:
    """Run Optuna hyperparameter optimization using walk-forward validation."""
    import json
    from algomlb.ml.hyperopt import build_fold_data, optimize_model

    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]

    # ── Build year list ───────────────────────────────────────────────
    all_years = [
        y for y in range(start_year, end_year + 1) if not (skip_2020 and y == 2020)
    ]
    if len(all_years) < 2:
        logger.error("Need at least 2 years for walk-forward optimization.")
        raise typer.Exit(code=1)

    # ── Prefetch all data ─────────────────────────────────────────────
    data = _load_ml_data(engine, ",".join(map(str, all_years)))

    # ── Pre-build fold matrices ───────────────────────────────────────
    logger.info("Pre-building walk-forward fold matrices...")
    fold_data = build_fold_data(
        all_years,
        data["games"],
        data["pitcher_gold"],
        data["batter_gold"],
        data["lineups"],
        data["elo"],
        data["pythag"],
        data["re24"],
    )

    if not fold_data:
        logger.error("No valid folds could be built. Exiting.")
        raise typer.Exit(code=1)

    logger.info(f"Built {len(fold_data)} folds. Launching Optuna study...")

    # ── Run Optuna ────────────────────────────────────────────────────
    best_params, study = optimize_model(
        fold_data, n_trials=n_trials, study_name=f"uranium_{model_version}"
    )

    # ── Persist best params ───────────────────────────────────────────
    params_path = Path(f".data/models/optuna_best_params_{model_version}.json")
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.success(f"Best params saved to {params_path}")


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
    model_version: str = typer.Option("v0.1", help="Model version label"),
) -> None:
    """Walk-forward validation: iteratively expand training window and test on next year."""
    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]

    all_years = [
        y for y in range(start_year, end_year + 1) if not (skip_2020 and y == 2020)
    ]
    if len(all_years) < 2:
        logger.error("Need at least 2 years for walk-forward validation.")
        raise typer.Exit(1)

    data = _load_ml_data(engine, ",".join(map(str, all_years)))
    from algomlb.ml.hyperopt import load_optimized_params

    xgb_params = load_optimized_params(model_version)
    results_table: list[dict] = []

    for test_idx in range(1, len(all_years)):
        train_years, test_year = all_years[:test_idx], all_years[test_idx]
        logger.info(f"── Fold {test_idx}: Train {train_years} → Test {test_year} ──")

        fold_games = pd.concat(
            [
                data["games"][data["games"]["year"].isin(train_years)],
                data["games"][data["games"]["year"] == test_year],
            ],
            ignore_index=True,
        )

        # Build matrix
        pipeline = FeaturePipeline()
        f_lineups = data["lineups"][
            data["lineups"]["game_pk"].isin(fold_games["game_pk"])
        ]
        f_seasons = set(train_years) | {test_year}
        f_pitcher = data["pitcher_gold"][data["pitcher_gold"]["season"].isin(f_seasons)]
        f_batter = data["batter_gold"][data["batter_gold"]["season"].isin(f_seasons)]

        X, y = pipeline.build_uranium_matrix(
            fold_games,
            f_pitcher,
            f_lineups if not f_lineups.empty else None,
            f_batter if not f_batter.empty else None,
            elo_df=data["elo"],
            pythag_df=data["pythag"],
            re24_df=data["re24"],
        )

        if X.empty:
            continue
        X["_yr"] = pd.to_datetime(fold_games.loc[X.index, "game_date"]).dt.year
        X_tr, y_tr = (
            X[X["_yr"].isin(train_years)].drop(columns=["_yr"]),
            y[X["_yr"].isin(train_years)],
        )
        X_te, y_te = (
            X[X["_yr"] == test_year].drop(columns=["_yr"]),
            y[X["_yr"] == test_year],
        )

        if X_tr.empty or X_te.empty:
            continue

        model = MLBModel(**xgb_params)
        model.train(X_tr, y_tr, calibrate=True)
        metrics = _evaluate_and_report(
            model,
            X_te,
            y_te,
            test_year,
            engine,
            model_version,
            min(train_years),
            max(train_years),
        )

        results_table.append(
            {
                "fold": test_idx,
                "train": str(train_years),
                "test": test_year,
                "acc": round(metrics["accuracy"], 4),
                "ll": round(metrics["log_loss"], 4),
                "auc": round(metrics["auc"], 4),
            }
        )

    if results_table:
        summary = pd.DataFrame(results_table)
        logger.success(f"Walk-Forward Summary:\n{summary.to_string(index=False)}")


# Dummy use for deptry to ignore optuna
_ = optuna.Study
