"""Uranium evaluation harness — offline metrics, calibration, and persistence."""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from algomlb.core.logger import logger
from algomlb.db.session import get_engine


# ── Metric helpers ────────────────────────────────────────────────────────


def compute_fold_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    labels: Optional[List[Any]] = None,
) -> dict:
    """Return accuracy, AUC, log loss, and Brier score for a single fold."""
    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        roc_auc_score,
    )
    from sklearn.preprocessing import LabelEncoder

    y_true_raw = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    # Robust encoding for Accuracy/AUC if y_true is strings
    if not np.issubdtype(y_true_raw.dtype, np.number):
        le = LabelEncoder()
        if labels is not None:
            le.classes_ = np.array(labels)
            # Filter y_true to only include known labels to prevent transform errors
            y_true = le.transform(y_true_raw)
        else:
            y_true = le.fit_transform(y_true_raw)
    else:
        y_true = y_true_raw.astype(int)

    # Multiclass metrics
    if y_prob.ndim == 2 and y_prob.shape[1] > 2:
        num_classes = y_prob.shape[1]
        y_pred = np.argmax(y_prob, axis=1)
        # For ECE/Calibration on multiclass, we use Confidence Calibration (Top-1)
        p_conf = np.max(y_prob, axis=1)
        y_correct = (y_pred == y_true).astype(int)
        ece = calculate_ece(y_correct, p_conf)

        # Multiclass Brier score: Mean square error of probability vectors
        # Brier = mean(sum((y_i - p_i)^2))
        y_true_onehot = np.zeros_like(y_prob)
        y_true_indices = y_true.astype(int)
        # Handle cases where y_true might have indices outside the y_prob range
        valid_mask = y_true_indices < num_classes
        y_true_onehot[np.arange(len(y_true))[valid_mask], y_true_indices[valid_mask]] = 1
        brier = np.mean(np.sum((y_true_onehot - y_prob) ** 2, axis=1))

        # Multiclass AUC (OvR)
        try:
            # We must specify labels to ensure AUC matches the prob matrix dimensions
            auc_labels = labels if labels is not None else np.unique(y_true)
            auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=auc_labels))
        except Exception as e:
            logger.warning(f"AUC calculation failed: {e}")
            auc = 0.5

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, y_prob)),
            "ece": float(ece),
            "auc": auc,
            "brier": float(brier),
        }

    # Safety: handle 2D binary inputs passed to 1D metric functions
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    elif y_prob.ndim != 1:
        raise ValueError(f"compute_fold_metrics expects 1D array, got {y_prob.shape}")

    from sklearn.metrics import brier_score_loss

    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(calculate_ece(y_true, y_prob)),
    }


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Calculates Expected Calibration Error using a weighted average of bin errors."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if np.any(mask):
            pred_mean = y_prob[mask].mean()
            obs_rate = y_true[mask].mean()
            weight = mask.sum() / len(y_true)
            ece += weight * np.abs(pred_mean - obs_rate)
    return ece


# ── Calibration bins ─────────────────────────────────────────────────────


def compute_calibration_bins(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Bins predictions and calculates observed rates. Supports multiclass matrices."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # For multiclass, we reduce to Confidence Calibration (Top-1)
    if y_prob.ndim == 2 and y_prob.shape[1] > 2:
        y_pred = np.argmax(y_prob, axis=1)
        y_prob = np.max(y_prob, axis=1)  # p_confidence
        y_true = (y_pred == y_true).astype(int)  # matches truth
    elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    results = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())

        if n > 0:
            pred_mean = float(y_prob[mask].mean())
            obs_rate = float(y_true[mask].mean())
            results.append(
                {
                    "bin_index": i,
                    "bin_start": lo,
                    "bin_end": hi,
                    "predicted_prob_mean": pred_mean,
                    "actual_prob_mean": obs_rate,
                    "sample_count": n,
                }
            )
        else:
            results.append(
                {
                    "bin_index": i,
                    "bin_start": lo,
                    "bin_end": hi,
                    "predicted_prob_mean": None,
                    "actual_prob_mean": None,
                    "sample_count": 0,
                }
            )

    return pd.DataFrame(results)


# ── Per-game evaluation ──────────────────────────────────────────────────


def compute_per_game_eval(
    games_df: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """
    Build a per-game evaluation DataFrame with confidence tiers.
    Expects *games_df* to be aligned with y_true / y_prob by index.
    """
    out = pd.DataFrame(index=y_true.index)

    for col in ["game_pk", "game_date", "home_team", "away_team"]:
        if col in games_df.columns:
            out[col] = games_df.loc[y_true.index, col].values

    out["p_model"] = y_prob
    out["home_win"] = y_true.values
    out["edge"] = np.abs(y_prob - 0.5)

    # Confidence tiers
    out["confidence_tier"] = pd.cut(
        out["edge"],
        bins=[0, 0.05, 0.10, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )

    y_pred = (y_prob >= 0.5).astype(int)
    out["correct"] = (y_pred == y_true.values).astype(int)
    out["is_high_conf_miss"] = (out["confidence_tier"] == "high") & (
        out["correct"] == 0
    )

    return out


# ── DB persistence ───────────────────────────────────────────────────────


def persist_eval_results(
    engine: Engine | None,
    model_target: str,
    model_version: str,
    fold_date: datetime.date,
    train_start: int,
    train_end: int,
    n_samples: int,
    metrics: dict,
    cal_bins: pd.DataFrame,
) -> None:
    """Upsert eval metrics and calibration bins into PostgreSQL."""
    eng = engine or get_engine()
    from algomlb.db.models import (
        UraniumEvalHistoryORM,
        UraniumCalibrationBinsORM,
    )

    # ── Eval history row ──────────────────────────────────────────────
    eval_row = {
        "model_target": model_target,
        "model_version": model_version,
        "fold_date": fold_date,
        "train_start_year": train_start,
        "train_end_year": train_end,
        "n_samples": n_samples,
        "accuracy": metrics.get("accuracy"),
        "auc": metrics.get("auc"),
        "log_loss_val": metrics.get("log_loss"),
        "brier": metrics.get("brier"),
        "ece": metrics.get("ece"),
    }

    with eng.begin() as conn:
        stmt = pg_insert(UraniumEvalHistoryORM).values([eval_row])
        upsert = stmt.on_conflict_do_update(
            index_elements=["model_target", "model_version", "fold_date"],
            set_={
                "train_start_year": stmt.excluded.train_start_year,
                "train_end_year": stmt.excluded.train_end_year,
                "n_samples": stmt.excluded.n_samples,
                "accuracy": stmt.excluded.accuracy,
                "auc": stmt.excluded.auc,
                "log_loss_val": stmt.excluded.log_loss_val,
                "brier": stmt.excluded.brier,
                "ece": stmt.excluded.ece,
            },
        )
        conn.execute(upsert)

    # ── Calibration bins ──────────────────────────────────────────────
    bin_records = cal_bins.copy()
    bin_records["model_target"] = model_target
    bin_records["model_version"] = model_version
    bin_records["fold_date"] = fold_date
    records = bin_records.to_dict(orient="records")

    with eng.begin() as conn:
        for rec in records:
            stmt = pg_insert(UraniumCalibrationBinsORM).values([rec])
            upsert = stmt.on_conflict_do_update(
                index_elements=[
                    "model_target",
                    "model_version",
                    "fold_date",
                    "bin_index",
                ],
                set_={
                    "bin_start": stmt.excluded.bin_start,
                    "bin_end": stmt.excluded.bin_end,
                    "predicted_prob_mean": stmt.excluded.predicted_prob_mean,
                    "actual_prob_mean": stmt.excluded.actual_prob_mean,
                    "sample_count": stmt.excluded.sample_count,
                },
            )
            conn.execute(upsert)

    logger.info(
        f"Persisted eval results for {model_target}/{model_version}/{fold_date}: "
        f"Acc={metrics.get('accuracy', 0.0):.4f} AUC={metrics.get('auc', 0.0):.4f}, "
        f"{len(records)} calibration bins."
    )


def fetch_eval_history(
    target: str,
    model_version: str,
    engine: Engine,
) -> pd.DataFrame:
    """Fetch all eval history records for a specific model version."""
    from algomlb.db.models import UraniumEvalHistoryORM
    from sqlalchemy import select

    with engine.connect() as conn:
        stmt = select(UraniumEvalHistoryORM).where(
            UraniumEvalHistoryORM.model_version == model_version
        )
        return pd.read_sql(stmt, conn)
