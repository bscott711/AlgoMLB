"""
AlgoMLB Machine Learning Command Line Interface.
Standardized interface for model tuning, backtesting, and diagnostics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import datetime
import pandas as pd
import numpy as np
import typer
import json
from pathlib import Path
import sqlalchemy as sa
from loguru import logger
from pydantic import BaseModel

from algomlb.db.session import get_session_factory
from algomlb.ml.features import FeaturePipeline
from algomlb.ml.model import MLBModel
from algomlb.ml.monte_carlo.loader import MatchupLoader
from algomlb.ml.monte_carlo.engine import SimulationEngine
from algomlb.ml.monte_carlo.aggregator import SimulationAggregator
from algomlb.db.models import UraniumSimulatedPlayerPropsORM

# Alignment: Top-level imports for Pyright
from algomlb.ml.eval import (
    compute_fold_metrics,
    compute_calibration_bins,
    fetch_eval_history,
    persist_eval_results,
)
from algomlb.ml.shap_analysis import (
    compute_global_shap,
    persist_global_shap,
)
from algomlb.ml.sabermetrics import (
    compute_pythagorean_features,
    compute_re24_per_pa,
    compute_rolling_re24,
)
from algomlb.ml.training.backtester import (
    OOFAccumulator,
    TimeSeriesSplitter,
    TimeSplitConfig,
)
from algomlb.ml.training.optuna_tuner import run_optuna_study
from algomlb.ml.hyperopt import load_optimized_params

app = typer.Typer()


class AgentResult(BaseModel):
    """Standardized reporting schema for autonomous agents."""

    status: str
    command: str
    duration_ms: int = 0
    data: Dict[str, Any] = {}
    error: Optional[str] = None


def emit_agent_result(result: AgentResult) -> None:
    """Print the result as JSON for the agent to consume."""
    print(f"AGENT_RESULT: {result.model_dump_json()}")


# ── Compatibility Layer for Unit Tests ────────────────────────────────────


def run_decoupler_pipeline(*args, **kwargs) -> Dict[str, Any]:
    """Legacy compatibility stub for testing."""
    logger.info("Decoupler pipeline stub invoked.")
    return {"status": "success", "results": {}}


# ─────────────────────────────────────────────────────────────────────────


def _load_ml_data(engine: Any, years_str: str) -> Dict[str, pd.DataFrame]:
    """Load the gold layer features for the specified training window."""
    logger.info(f"Fetching features for years: {years_str}...")

    games_df = pd.read_sql(
        f"SELECT * FROM game_results WHERE EXTRACT(YEAR FROM game_date) IN ({years_str})",
        engine,
    )
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])

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

    try:
        years_list = [int(y.strip()) for y in years_str.split(",")]
        retro_df = pd.read_sql(
            f"""
            SELECT game_id, date AS game_date, inning, top_bot, outs_pre, outs_post,
                   br1_pre, br2_pre, br3_pre, br1_post, br2_post, br3_post,
                   runs, pa_flag, batter_id, pitcher_id, bat_team, pit_team,
                   balls, strikes,
                   walk, k, hbp, single, double_flag, triple, hr
            FROM retrosheet_events
            WHERE EXTRACT(YEAR FROM date) IN ({years_str}) AND pa_flag = 1
            """,
            engine,
        )
        retro_df = _label_pa_outcomes(retro_df)
    except Exception as e:
        logger.warning(f"retrosheet_events loading failed: {e}")
        retro_df = pd.DataFrame()

    pythag_df = compute_pythagorean_features(games_df)

    # RE24: Now pre-computed in player_rolling_features (loaded into pitcher_gold/batter_gold)
    # The build_uranium_matrix method will handle joining these from the gold dataframes.
    re24_df = pd.concat(
        [pitcher_gold.dropna(axis=1, how="all"), batter_gold.dropna(axis=1, how="all")],
        ignore_index=True,
    )

    return {
        "games": games_df,
        "pitcher_gold": pitcher_gold,
        "lineups": lineups,
        "batter_gold": batter_gold,
        "elo": elo_df,
        "pythag": pythag_df,
        "re24": re24_df,
        "pas": retro_df,
    }


def _label_pa_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Map Retrosheet flags to canonical simulation outcomes."""
    if df.empty:
        return df
    
    # Priority order for mapping:
    df["pa_outcome"] = "out_in_play"
    df.loc[df["k"] == 1, "pa_outcome"] = "strikeout"
    df.loc[df["walk"] == 1, "pa_outcome"] = "walk"
    df.loc[df["hbp"] == 1, "pa_outcome"] = "hbp"
    df.loc[df["single"] == 1, "pa_outcome"] = "single"
    df.loc[df["double_flag"] == 1, "pa_outcome"] = "double"
    df.loc[df["triple"] == 1, "pa_outcome"] = "triple"
    df.loc[df["hr"] == 1, "pa_outcome"] = "home_run"
    
    return df


def _evaluate_and_report(
    model: MLBModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target: str,
    fold_date: datetime.date,
    engine: Any,
    model_version: str,
    start_year: int,
    end_year: int,
) -> dict:
    """Helper used by tests and CLI to run metrics, calibration, and persistence."""
    y_prob = model.predict_proba(X_test)

    # Hardening: Handles multiclass (pa_outcome) vs binary (props)
    is_multiclass = y_prob.ndim > 1 and y_prob.shape[1] > 2

    # Alignment: Strict numpy primitives for reportArgumentType
    y_true_arr = y_test.to_numpy()

    # Use Brier score for binary, mlogloss for multiclass
    if is_multiclass:
        from sklearn.metrics import log_loss, accuracy_score

        y_pred = np.argmax(y_prob, axis=1)
        metrics = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred)),
            "log_loss": float(log_loss(y_true_arr, y_prob)),
            "brier": 0.0,  # Not used for multiclass
            "auc": 0.0,  # ROC-AUC is complex for multiclass, placeholder
        }
    else:
        # P(class=1) for binary metrics
        p_pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        metrics = compute_fold_metrics(y_true_arr, p_pos)

    cal_bins = compute_calibration_bins(
        y_true_arr, y_prob[:, 1] if y_prob.ndim > 1 else y_prob
    )

    persist_eval_results(
        engine=engine,
        model_target=target,
        model_version=model_version,
        fold_date=fold_date,
        train_start=start_year,
        train_end=end_year,
        n_samples=len(y_test),
        metrics=metrics,
        cal_bins=cal_bins,
    )
    return metrics


@app.command()
def tune(
    ctx: typer.Context,
    target: str = typer.Option(..., "--target", help="The target column to predict."),
    trials: int = typer.Option(100, "--trials", help="Number of Optuna trials."),
    version: str = typer.Option("v1.3", "--version", help="Model version."),
) -> None:
    """Optimize Uranium model hyperparameters via Optuna."""
    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]
    years_str = "2021,2022,2023,2024,2025"

    data = _load_ml_data(engine, years_str)
    if data["games"].empty:
        logger.error("No games found.")
        raise typer.Exit(1)

    pipeline = FeaturePipeline()
    if target == "pa_outcome":
        X, y = pipeline.build_pa_matrix(
            data["pas"],
            data["pitcher_gold"],
            data["batter_gold"],
            lineups_df=data["lineups"] if not data["lineups"].empty else None,
            elo_df=data["elo"],
        )
    else:
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

    study_name = f"{target}_{version}"
    y_frame = y.to_frame(name=target)
    combined_df = cast(pd.DataFrame, pd.concat([X, y_frame], axis=1))
    combined_df = combined_df.reset_index()

    study = run_optuna_study(
        combined_df, X.columns.tolist(), target, n_trials=trials, study_name=study_name
    )
    best_params = study.best_params

    logger.success(f"Tuning complete for {target}. Best Params: {best_params}")

    # Persistence: Save best params to JSON for backtester/explain
    out_dir = Path(".data/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"optuna_best_params_{target}_{version}.json"
    with out_path.open("w") as f:
        json.dump(best_params, f, indent=4)
    logger.success(f"Optimized parameters persisted to {out_path}")

    if ctx.obj and ctx.obj.get("agent_mode", False):
        emit_agent_result(
            AgentResult(
                status="success",
                command="ml.tune",
                data={
                    "target": target,
                    "best_params": best_params,
                    "study_name": study_name,
                    "game_date": [
                        str(d) for d in combined_df.index.get_level_values("game_date")
                    ]
                    if hasattr(combined_df.index, "names")
                    and "game_date" in combined_df.index.names
                    else (
                        combined_df["game_date"].dt.strftime("%Y-%m-%d").tolist()
                        if "game_date" in combined_df.columns
                        else []
                    ),
                },
            )
        )


@app.command()
def backtest(
    ctx: typer.Context,
    target: str = typer.Option(..., "--target", help="The target column to predict."),
    version: str = typer.Option("v1.3", "--version", help="Model version."),
) -> None:
    """Run walk-forward backtesting with fixed hyperparameters."""
    import datetime

    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]
    years_str = "2021,2022,2023,2024,2025"

    data = _load_ml_data(engine, years_str)
    if data["games"].empty:
        logger.error("No games found.")
        raise typer.Exit(1)

    pipeline = FeaturePipeline()
    if target == "pa_outcome":
        X, y = pipeline.build_pa_matrix(
            data["pas"],
            data["pitcher_gold"],
            data["batter_gold"],
            lineups_df=data["lineups"] if not data["lineups"].empty else None,
            elo_df=data["elo"],
        )
    else:
        X, y = pipeline.build_uranium_matrix(
            data["games"],
            data["pitcher_gold"],
            data["lineups"] if not data["lineups"].empty else None,
            data["batter_gold"] if not data["batter_gold"].empty else None,
            elo_df=data["elo"],
            pythag_df=data["pythag"],
            re24_df=data["re24"],
        )

    # Combined DF for backtester
    combined_df = cast(pd.DataFrame, pd.concat([X, y.to_frame(name=target)], axis=1))
    combined_df = combined_df.reset_index()

    params = load_optimized_params(target, version)
    accumulator = OOFAccumulator(
        model_class=MLBModel, features=X.columns.tolist(), target=target
    )
    splitter = TimeSeriesSplitter(
        config=TimeSplitConfig(train_window_days=730, test_window_days=30)
    )
    oof_df = accumulator.run_backtest(combined_df, splitter, **params)

    # Diagnostic metrics on full OOF
    y_true = oof_df[target].to_numpy()

    # Robustly handle p_model being either a scalar or a vector (binary/multiclass)
    if isinstance(oof_df["p_model"].iloc[0], (list, np.ndarray)):
        prob_matrix = np.stack(oof_df["p_model"].tolist())
        if prob_matrix.shape[1] > 2:
            # Multiclass handling
            from sklearn.metrics import log_loss, accuracy_score

            y_pred = np.argmax(prob_matrix, axis=1)
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "log_loss": float(log_loss(y_true, prob_matrix)),
                "brier": 0.0,
                "auc": 0.0,
            }
            p_pos = prob_matrix  # Calibration bins will likely need adjustment for multiclass
        else:
            # Binary probabilities [p_neg, p_pos]
            p_pos = prob_matrix[:, 1]
            metrics = compute_fold_metrics(y_true, p_pos)
    else:
        # Scalar probabilities
        p_pos = oof_df["p_model"].to_numpy()
        metrics = compute_fold_metrics(y_true, p_pos)

    cal_bins = compute_calibration_bins(y_true, p_pos)

    persist_eval_results(
        engine=engine,
        model_target=target,
        model_version=version,
        fold_date=datetime.date.today(),
        train_start=2021,
        train_end=2025,
        n_samples=len(y_true),
        metrics=metrics,
        cal_bins=cal_bins,
    )

    logger.success(f"Backtest complete for {target}. Metrics: {metrics}")

    if ctx.obj and ctx.obj.get("agent_mode", False):
        emit_agent_result(
            AgentResult(
                status="success",
                command="ml.backtest",
                data={
                    "target": target,
                    "metrics": metrics,
                    "game_date": oof_df["game_date"].dt.strftime("%Y-%m-%d").tolist()
                    if "game_date" in oof_df.columns
                    else [],
                },
            )
        )


@app.command()
def explain(
    ctx: typer.Context,
    target: str = typer.Option(..., "--target", help="The target column to explain."),
    version: str = typer.Option("v1.3", "--version", help="Model version."),
) -> None:
    """Generate and persist SHAP explanations for the model."""
    import datetime

    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]
    years_str = "2024,2025"

    data = _load_ml_data(engine, years_str)
    pipeline = FeaturePipeline()
    if target == "pa_outcome":
        X, y = pipeline.build_pa_matrix(
            data["pas"],
            data["pitcher_gold"],
            data["batter_gold"],
            lineups_df=data["lineups"] if not data["lineups"].empty else None,
            elo_df=data["elo"],
        )
    else:
        X, y = pipeline.build_uranium_matrix(
            data["games"],
            data["pitcher_gold"],
            data["lineups"] if not data["lineups"].empty else None,
            data["batter_gold"] if not data["batter_gold"].empty else None,
            elo_df=data["elo"],
            pythag_df=data["pythag"],
            re24_df=data["re24"],
        )

    params = load_optimized_params(target, version)
    model = MLBModel(**params)
    model.fit(X, y)

    shap_df = compute_global_shap(model, X)
    shap_df = cast(pd.DataFrame, shap_df)

    persist_global_shap(
        engine=engine,
        model_target=target,
        model_version=version,
        fold_date=datetime.date.today(),
        shap_df=shap_df,
    )

    logger.success(f"Explanation complete for {target}.")

    if ctx.obj and ctx.obj.get("agent_mode", False):
        emit_agent_result(
            AgentResult(
                status="success",
                command="ml.explain",
                data={
                    "target": target,
                    "version": version,
                    "importance_top_10": shap_df.head(10).to_dict(),
                },
            )
        )


@app.command()
def train(
    ctx: typer.Context,
    target: str = typer.Option(..., "--target", help="The target column to predict."),
    version: str = typer.Option("v1.3", "--version", help="Model version."),
) -> None:
    """Train and persist a production Uranium model using best params."""
    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]
    years_str = "2021,2022,2023,2024,2025"

    data = _load_ml_data(engine, years_str)
    pipeline = FeaturePipeline()
    if target == "pa_outcome":
        X, y = pipeline.build_pa_matrix(
            data["pas"],
            data["pitcher_gold"],
            data["batter_gold"],
            lineups_df=data["lineups"] if not data["lineups"].empty else None,
            elo_df=data["elo"],
            re24_df=data["re24"],
        )
    else:
        X, y = pipeline.build_uranium_matrix(
            data["games"],
            data["pitcher_gold"],
            data["lineups"] if not data["lineups"].empty else None,
            data["batter_gold"] if not data["batter_gold"].empty else None,
            elo_df=data["elo"],
            pythag_df=data["pythag"],
            re24_df=data["re24"],
        )

    params = load_optimized_params(target, version)
    model = MLBModel(**params)

    logger.info(f"Fitting production model for {target}...")
    model.fit(X, y, calibrate=True)

    out_dir = Path(".data/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target}_{version}.joblib"
    model.save(out_path)

    logger.success(f"Production model saved to {out_path}")


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


@app.command(name="fetch-history")
def fetch_history(
    ctx: typer.Context,
    target: str = typer.Option(..., "--target", help="Target column"),
    version: str = typer.Option("v1.2", "--version", help="Model version"),
) -> None:
    """Fetch past evaluation history for visual reporting."""
    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]

    history = fetch_eval_history(target, version, engine)
    logger.info(f"Fetched {len(history)} history records.")

    if ctx.obj and ctx.obj.get("agent_mode", False):
        emit_agent_result(
            AgentResult(
                status="success",
                command="ml.fetch-history",
                data={"history_count": len(history)},
            )
        )


@app.command()
def decouple(
    ctx: typer.Context,
) -> None:
    """Compatibility command for BAT flight decoupling."""
    run_decoupler_pipeline()
    if ctx.obj and ctx.obj.get("agent_mode", False):
        emit_agent_result(AgentResult(status="success", command="ml.decouple"))


@app.command(name="simulate-game")
def simulate_game(
    ctx: typer.Context,
    game_pk: int = typer.Option(..., "--game-pk", help="Game PK to simulate."),
    trials: int = typer.Option(10000, "--trials", help="Number of Monte Carlo trials."),
    version: str = typer.Option(
        "v1.3", "--version", help="Model version for pa_outcome."
    ),
    use_hook_model: bool = typer.Option(
        True,
        "--use-hook-model/--no-hook-model",
        help="Use the trained ML hook model for pitcher substitutions (default: True).",
    ),
    hook_model_path: Optional[str] = typer.Option(
        None,
        "--hook-model-path",
        help="Path to hook model joblib. Defaults to .data/models/hook_model_v1.0.joblib.",
    ),
) -> None:
    """Run a high-fidelity Monte Carlo simulation for all player props in a single game."""
    session_factory = get_session_factory()
    session = session_factory()

    try:
        # 1. Load Data
        loader = MatchupLoader(session)
        context = loader.load_matchup(game_pk)
        if not context:
            raise typer.Exit(1)

        # 2. Load PA Outcome Model
        model_path = Path(f".data/models/pa_outcome_{version}.joblib")
        if not model_path.exists():
            logger.error(
                f"PA Outcome model not found at {model_path}. Please train it first."
            )
            raise typer.Exit(1)
        model = MLBModel.load(model_path)

        # 3. Optionally load Hook Model
        from algomlb.ml.hook_model import HookModel

        hook_model = None
        if use_hook_model:
            _hook_path = Path(
                hook_model_path or ".data/models/hook_model_v1.0.joblib"
            )
            if _hook_path.exists():
                hook_model = HookModel.load(_hook_path)
                logger.info(f"🎯 Loaded hook model from {_hook_path}")
            else:
                logger.warning(
                    f"⚠️ Hook model not found at {_hook_path}; "
                    "using heuristic pitcher substitutions."
                )

        # 4. Load Bullpen Params from SQL
        bp_params = loader.get_simulation_config("bullpen_v1.0")
        if bp_params:
            logger.info(f"⚡ Applying tuned bullpen parameters from SQL: {bp_params}")

        # 5. Simulate
        engine = SimulationEngine(pa_model=model, hook_model=hook_model)
        trial_results = engine.run_trials(context, trials=trials, bullpen_params=bp_params)

        # 4. Aggregate
        aggregator = SimulationAggregator()
        results_df = aggregator.aggregate_results(
            game_pk, context.game_date.year, trial_results, context
        )

        # 5. Persist
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        records = results_df.to_dict(orient="records")
        for rec in records:
            stmt = pg_insert(UraniumSimulatedPlayerPropsORM).values([rec])
            upsert = stmt.on_conflict_do_update(
                index_elements=["game_pk", "player_id", "stat_type"],
                set_={
                    "season": stmt.excluded.season,
                    "mean": stmt.excluded.mean,
                    "median": stmt.excluded.median,
                    "prob_over_0_5": stmt.excluded.prob_over_0_5,
                    "prob_over_1_5": stmt.excluded.prob_over_1_5,
                    "prob_over_2_5": stmt.excluded.prob_over_2_5,
                    "prob_over_3_5": stmt.excluded.prob_over_3_5,
                    "prob_over_4_5": stmt.excluded.prob_over_4_5,
                    "p10": stmt.excluded.p10,
                    "p90": stmt.excluded.p90,
                    "trials": stmt.excluded.trials,
                    "simulated_at": sa.func.now(),
                },
            )
            session.execute(upsert)

        session.commit()

        logger.success(
            f"Simulation result persisted to uranium_simulated_player_props for {len(results_df)} prop markets."
        )

        if ctx.obj and ctx.obj.get("agent_mode", False):
            emit_agent_result(
                AgentResult(
                    status="success",
                    command="ml.simulate-game",
                    data={"game_pk": game_pk, "prop_markets": len(results_df)},
                )
            )

    except Exception as e:
        logger.exception(f"Simulation failed: {e}")
        raise typer.Exit(1)
    finally:
        session.close()


@app.command(name="train-hook-model")
def train_hook_model(
    ctx: typer.Context,
    version: str = typer.Option("v1.0", "--version", help="Model version tag."),
    output: str = typer.Option(
        ".data/models", "--output", help="Output directory for the joblib bundle."
    ),
    no_calibrate: bool = typer.Option(
        False,
        "--no-calibrate",
        help="Skip isotonic calibration (faster, less accurate probabilities).",
    ),
) -> None:
    """Train and persist the manager hook decision model from manager_hook_events."""
    import sys as _sys

    from algomlb.ml.hook_model import HookModel, compute_leverage_index  # noqa: F401

    session_factory = get_session_factory()
    engine_db = session_factory.kw["bind"]

    import pandas as _pd

    query = """
        SELECT game_pk, team_id, game_date, season,
               inning, outs_at_hook, pitches_thrown, tto_at_hook,
               score_diff_at_hook, base_state_at_hook,
               runs_allowed, hits_allowed, walks_allowed, strikeouts,
               is_starter
        FROM manager_hook_events
        ORDER BY game_date, game_pk, team_id, inning
    """
    df = _pd.read_sql(query, engine_db)
    if df.empty:
        logger.error("No hook events found. Run backfill-manager-hooks first.")
        raise typer.Exit(1)

    # Derive labels (last pitcher per game-team = completed, others = hooked)
    df["_rank"] = df.groupby(["game_pk", "team_id"])["inning"].rank(
        method="first", ascending=False
    )
    df["was_hooked"] = (df["_rank"] > 1).astype(int)
    df = df.drop(columns=["_rank"])

    # Recompute LI
    df["leverage_index_at_hook"] = [
        compute_leverage_index(
            inning=int(r["inning"]),
            outs=int(r["outs_at_hook"]),
            base_state=int(r["base_state_at_hook"]),
            score_diff=int(r["score_diff_at_hook"]),
        )
        for _, r in df.iterrows()
    ]
    df["is_starter"] = df["is_starter"].astype(int)

    X = df[HookModel.FEATURE_NAMES].astype(float)
    y = df["was_hooked"].astype(int)

    if X.empty or y.nunique() < 2:
        logger.error("Insufficient or single-class training data.")
        raise typer.Exit(1)

    import datetime as _dt

    cutoff = _pd.Timestamp(_dt.date(2024, 4, 1))
    dates = _pd.to_datetime(df["game_date"])
    train_mask = dates < cutoff
    X_train, y_train = X[train_mask], y[train_mask]

    model = HookModel()
    logger.info(f"Training HookModel on {train_mask.sum():,} events...")
    model.fit(X_train, y_train, calibrate=not no_calibrate)

    from pathlib import Path as _Path

    out_path = _Path(output) / f"hook_model_{version}.joblib"
    _Path(output).mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    logger.success(f"Hook model saved → {out_path}")

    if ctx.obj and ctx.obj.get("agent_mode", False):
        emit_agent_result(
            AgentResult(
                status="success",
                command="ml.train-hook-model",
                data={"version": version, "output": str(out_path)},
            )
        )


@app.command(name="backfill-manager-hooks")
def backfill_manager_hooks(
    ctx: typer.Context,
    start_year: int = typer.Option(
        2019, "--start-year", help="First season to backfill (default: 2019)."
    ),
    end_year: int = typer.Option(
        2025, "--end-year", help="Last season to backfill (default: 2025)."
    ),
) -> None:
    """Extract hook events from Retrosheet, persist to DB, and compute manager profiles."""
    from algomlb.ml.hooks import backfill_hook_events

    session_factory = get_session_factory()
    engine_db = session_factory.kw["bind"]

    logger.info(
        f"🚀 Starting Manager Hook Backfill (Retrosheet → Hooks) for {start_year}–{end_year}..."
    )
    try:
        backfill_hook_events(engine_db, start_year=start_year, end_year=end_year)
        logger.success("🏁 Manager Hook Backfill complete!")
    except Exception as e:
        logger.exception(f"❌ Backfill failed: {e}")
        raise typer.Exit(1)

    if ctx.obj and ctx.obj.get("agent_mode", False):
        emit_agent_result(
            AgentResult(
                status="success",
                command="ml.backfill-manager-hooks",
                data={"start_year": start_year, "end_year": end_year},
            )
        )
