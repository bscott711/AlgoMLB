"""Optuna walk-forward hyperparameter optimization for Uranium XGBoost models."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from algomlb.core.logger import logger
from algomlb.ml import FeaturePipeline, MLBModel


# ── Single-fold evaluator ────────────────────────────────────────────────


def _run_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
) -> float:
    """Train an XGBoost model with *params* and return test log-loss."""
    model = MLBModel(**params)
    model.train(
        X_train, y_train, calibrate=False
    )  # no calibration during optuna search — faster
    y_prob = model.predict_proba(X_test)[:, 1]
    return float(log_loss(y_test, y_prob))


# ── Optuna objective ─────────────────────────────────────────────────────


def walk_forward_objective(
    trial: optuna.Trial,
    fold_data: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]],
) -> float:
    """
    Optuna objective that minimises average log-loss across pre-built
    walk-forward folds.

    Parameters
    ----------
    trial : optuna.Trial
    fold_data : list of (X_train, y_train, X_test, y_test) tuples
        Pre-built from the Uranium feature pipeline so the expensive
        matrix construction only happens once.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    fold_scores: list[float] = []
    for fold_idx, (X_tr, y_tr, X_te, y_te) in enumerate(fold_data):
        score = _run_fold(X_tr, y_tr, X_te, y_te, params)
        fold_scores.append(score)
        # Pruning: if first folds already terrible, stop early
        trial.report(score, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    avg = float(np.mean(fold_scores))
    return avg


# ── Pre-build fold data ──────────────────────────────────────────────────


def build_fold_data(
    all_years: list[int],
    games_df: pd.DataFrame,
    pitcher_gold_df: pd.DataFrame,
    batter_gold_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    elo_df: pd.DataFrame,
    pythag_df: pd.DataFrame,
    re24_df: pd.DataFrame,
) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """
    Pre-build the Uranium feature matrices for each walk-forward fold.
    This mirrors the exact logic in ``cli.ml.walk_forward``.
    """
    fold_data: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]] = []

    for test_idx in range(1, len(all_years)):
        train_years = all_years[:test_idx]
        test_year = all_years[test_idx]

        logger.info(
            f"  Building fold {test_idx}: Train {train_years} → Test {test_year}"
        )

        train_games = games_df[games_df["year"].isin(train_years)].copy()
        test_games = games_df[games_df["year"] == test_year].copy()
        fold_games = pd.concat([train_games, test_games], ignore_index=True)

        fold_seasons = set(train_years) | {test_year}
        fold_pitcher = pitcher_gold_df[
            pitcher_gold_df["season"].isin(fold_seasons)
        ].copy()
        fold_batter = batter_gold_df[batter_gold_df["season"].isin(fold_seasons)].copy()

        fold_game_pks = (
            set(fold_games["game_pk"].dropna().astype(int).tolist())
            if "game_pk" in fold_games.columns
            else set()
        )
        fold_lineups = (
            lineups_df[lineups_df["game_pk"].isin(fold_game_pks)].copy()
            if not lineups_df.empty
            else pd.DataFrame()
        )
        fold_elo = (
            elo_df[elo_df["game_pk"].isin(fold_game_pks)].copy()
            if not elo_df.empty
            else pd.DataFrame()
        )
        fold_pythag = (
            pythag_df[pythag_df["game_pk"].isin(fold_game_pks)].copy()
            if not pythag_df.empty
            else pd.DataFrame()
        )
        fold_re24 = re24_df  # player-level, rolling window handles temporal boundaries

        # Build Uranium matrix
        pipeline = FeaturePipeline()
        if not fold_lineups.empty and not fold_batter.empty:
            X, y = pipeline.build_uranium_matrix(
                fold_games,
                fold_pitcher,
                fold_lineups,
                fold_batter,
                elo_df=fold_elo,
                pythag_df=fold_pythag,
                re24_df=fold_re24,
            )
        else:
            X, y = pipeline.build_uranium_matrix(
                fold_games,
                fold_pitcher,
                elo_df=fold_elo,
                pythag_df=fold_pythag,
                re24_df=fold_re24,
            )

        if X.empty:
            logger.warning(f"  Fold {test_idx}: Empty matrix, skipping.")
            continue

        # Split by year
        X["_year"] = pd.to_datetime(fold_games.loc[X.index, "game_date"]).dt.year
        train_mask = X["_year"].isin(train_years)
        test_mask = X["_year"] == test_year

        X_train = X[train_mask].drop(columns=["_year"])
        y_train = y[train_mask]
        X_test = X[test_mask].drop(columns=["_year"])
        y_test = y[test_mask]

        if X_train.empty or X_test.empty:
            logger.warning(f"  Fold {test_idx}: Train or test empty, skipping.")
            continue

        fold_data.append((X_train, y_train, X_test, y_test))
        logger.info(
            f"  Fold {test_idx}: {len(X_train)} train, {len(X_test)} test, {X_train.shape[1]} features"
        )

    return fold_data


# ── Top-level runner ─────────────────────────────────────────────────────


def optimize_model(
    fold_data: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]],
    n_trials: int = 50,
    study_name: str = "uranium_walk_forward",
) -> tuple[dict[str, Any], optuna.Study]:
    """
    Run an Optuna study over pre-built walk-forward folds.

    Returns
    -------
    best_params : dict
    study : optuna.Study
    """
    if not fold_data:
        raise ValueError("No fold data provided — cannot optimise.")

    logger.info(
        f"Starting Optuna study '{study_name}' with {n_trials} trials "
        f"across {len(fold_data)} folds..."
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda trial: walk_forward_objective(trial, fold_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    logger.success(f"Optimisation complete. Best avg log-loss: {study.best_value:.4f}")
    logger.info("Best parameters:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    return study.best_params, study


# ── Param loader ─────────────────────────────────────────────────────────


def load_optimized_params(model_version: str = "v0.1") -> dict[str, Any]:
    """
    Load Optuna-optimised hyperparameters from the JSON artifact.

    Returns the parameters as a dict suitable for unpacking into ``MLBModel(**params)``.
    Falls back to conservative defaults if no artifact exists.
    """
    import json
    from pathlib import Path

    params_path = Path(f".data/models/optuna_best_params_{model_version}.json")

    if params_path.exists():
        logger.info(f"Loading optimised parameters from {params_path}")
        with params_path.open("r") as f:
            return json.load(f)

    logger.warning(
        f"No optimised params found for {model_version} at {params_path}. "
        "Using baseline defaults."
    )
    return {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
