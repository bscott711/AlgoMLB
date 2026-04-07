"""Uranium evaluation harness — offline metrics, calibration, and persistence."""
from __future__ import annotations

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
) -> dict:
    """Return accuracy, AUC, log loss, and Brier score for a single fold."""
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


# ── Calibration bins ─────────────────────────────────────────────────────

def compute_calibration_bins(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Bin predictions into *n_bins* equal-width intervals and compute
    per-bin mean predicted probability, observed win rate, and count.
    """
    y_true = np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            pred_mean = (lo + hi) / 2
            obs_rate = 0.0
        else:
            pred_mean = float(y_prob[mask].mean())
            obs_rate = float(y_true[mask].mean())

        rows.append({
            "bin_index": i,
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "pred_mean": pred_mean,
            "obs_rate": obs_rate,
            "n_samples": n,
        })

    return pd.DataFrame(rows)


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
    out["is_high_conf_miss"] = (out["confidence_tier"] == "high") & (out["correct"] == 0)

    return out


# ── DB persistence ───────────────────────────────────────────────────────

def persist_eval_results(
    engine: Engine | None,
    model_version: str,
    test_year: int,
    train_start: int,
    train_end: int,
    n_games: int,
    metrics: dict,
    cal_bins: pd.DataFrame,
) -> None:
    """Upsert eval metrics and calibration bins into PostgreSQL."""
    eng = engine or get_engine()
    from algomlb.db.models import UraniumCalibrationBinORM, UraniumEvalHistoryORM

    # ── Eval history row ──────────────────────────────────────────────
    eval_row = {
        "model_version": model_version,
        "test_year": test_year,
        "train_start_year": train_start,
        "train_end_year": train_end,
        "n_games": n_games,
        "accuracy": metrics["accuracy"],
        "auc": metrics["auc"],
        "log_loss_val": metrics["log_loss"],
        "brier": metrics["brier"],
    }

    with eng.begin() as conn:
        stmt = pg_insert(UraniumEvalHistoryORM.__table__).values([eval_row])
        upsert = stmt.on_conflict_do_update(
            index_elements=["model_version", "test_year"],
            set_={
                "train_start_year": stmt.excluded.train_start_year,
                "train_end_year": stmt.excluded.train_end_year,
                "n_games": stmt.excluded.n_games,
                "accuracy": stmt.excluded.accuracy,
                "auc": stmt.excluded.auc,
                "log_loss_val": stmt.excluded.log_loss_val,
                "brier": stmt.excluded.brier,
            },
        )
        conn.execute(upsert)

    # ── Calibration bins ──────────────────────────────────────────────
    bin_records = cal_bins.copy()
    bin_records["model_version"] = model_version
    bin_records["test_year"] = test_year
    records = bin_records.to_dict(orient="records")

    with eng.begin() as conn:
        for rec in records:
            stmt = pg_insert(UraniumCalibrationBinORM.__table__).values([rec])
            upsert = stmt.on_conflict_do_update(
                index_elements=["model_version", "test_year", "bin_index"],
                set_={
                    "bin_lower": stmt.excluded.bin_lower,
                    "bin_upper": stmt.excluded.bin_upper,
                    "pred_mean": stmt.excluded.pred_mean,
                    "obs_rate": stmt.excluded.obs_rate,
                    "n_samples": stmt.excluded.n_samples,
                },
            )
            conn.execute(upsert)

    logger.info(
        f"Persisted eval results for {model_version}/{test_year}: "
        f"Acc={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f} "
        f"Brier={metrics['brier']:.4f}, {len(records)} calibration bins."
    )
