#!/usr/bin/env python3
"""
scripts/train_hook_model.py

Train and persist the manager hook decision model.

    Training data: manager_hook_events table (populated by backfill_manager_hooks.py)

    Label derivation: Within each (game_pk, team_id) group, the LAST pitcher
    (highest inning) completed their outing naturally (y=0). All preceding
    pitchers were hooked/replaced mid-game (y=1). This avoids the placeholder
    `removed_before_next_batter=True` ETL bug where all rows were labeled 1.

    LI recomputation: The `leverage_index_at_hook` column in the DB is
    always 1.0 (placeholder from ETL). This script recomputes it from
    base-out-inning-score state using compute_leverage_index() so the model
    learns meaningful signal from this feature.

    Validation: Temporal hold-out — events before TRAIN_CUTOFF = train set,
    events on/after TRAIN_CUTOFF = validation set (no future data leakage).

Usage:
    uv run python scripts/train_hook_model.py [--version v1.0] [--output .data/models]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Ensure project src/ is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from algomlb.core.logger import logger
from algomlb.db.session import get_session_factory
from algomlb.ml.hook_model import HookModel, compute_leverage_index

# All events before this date → train, on/after → temporal validation
TRAIN_CUTOFF = date(2024, 4, 1)


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_hook_events(engine) -> pd.DataFrame:
    """
    Load all manager hook events from the DB with required training columns.

    Columns fetched match the HookModel.FEATURE_NAMES (minus leverage_index,
    which is recomputed) plus identifiers for label derivation.
    """
    query = """
        SELECT
            game_pk,
            team_id,
            game_date,
            season,
            inning,
            outs_at_hook,
            pitches_thrown,
            tto_at_hook,
            score_diff_at_hook,
            base_state_at_hook,
            runs_allowed,
            hits_allowed,
            walks_allowed,
            strikeouts,
            is_starter
        FROM manager_hook_events
        ORDER BY game_date, game_pk, team_id, inning
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df):,} hook event rows from manager_hook_events.")
    return df


def derive_hook_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive binary hook labels from pitcher stint ordering within each game.

    Logic:
      - Group by (game_pk, team_id).
      - The pitcher with the HIGHEST inning per group is the one who
        completed/closed naturally → y=0 (not hooked).
      - All other pitchers in the group were replaced mid-game → y=1 (hooked).

    This approach bypasses the ETL bug where `removed_before_next_batter` is
    always True regardless of whether the pitcher was truly hooked.
    """
    df = df.copy()
    # Max inning rank per game-team group (rank=1 → last pitcher of the game)
    df["_inning_rank"] = df.groupby(["game_pk", "team_id"])["inning"].rank(
        method="first", ascending=False
    )
    df["was_hooked"] = (df["_inning_rank"] > 1).astype(int)
    df = df.drop(columns=["_inning_rank"])

    n_hooked = df["was_hooked"].sum()
    n_total = len(df)
    logger.info(
        f"Label derivation: {n_hooked:,} hooked ({n_hooked / n_total:.1%}), "
        f"{n_total - n_hooked:,} completed ({(n_total - n_hooked) / n_total:.1%})."
    )
    return df


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build aligned feature matrix X, label vector y, and date series.

    LI is recomputed from (inning, outs, base_state, score_diff) to replace
    the placeholder 1.0 values stored by the ETL.

    Returns:
        X:     DataFrame aligned to HookModel.FEATURE_NAMES.
        y:     Binary Series (1 = hooked, 0 = completed).
        dates: Series of game_date for temporal splitting.
    """
    df = df.copy()
    df["leverage_index_at_hook"] = [
        compute_leverage_index(
            inning=int(row["inning"]),
            outs=int(row["outs_at_hook"]),
            base_state=int(row["base_state_at_hook"]),
            score_diff=int(row["score_diff_at_hook"]),
        )
        for _, row in df.iterrows()
    ]
    df["is_starter"] = df["is_starter"].astype(int)

    X = df[HookModel.FEATURE_NAMES].copy().astype(float)
    y = df["was_hooked"].astype(int)
    dates = pd.to_datetime(df["game_date"])
    return X, y, dates


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(model: HookModel, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Compute held-out validation metrics for the trained hook model."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    proba = model.predict_proba(X_val)[:, 1]
    return {
        "log_loss": round(log_loss(y_val, proba), 4),
        "roc_auc": round(roc_auc_score(y_val, proba), 4),
        "brier": round(brier_score_loss(y_val, proba), 4),
        "n_train": int((y_val.index < len(y_val)).sum()),  # filled below
        "n_val": len(y_val),
        "hook_rate_actual": round(float(y_val.mean()), 4),
        "hook_rate_pred_mean": round(float(proba.mean()), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> dict:
    parser = argparse.ArgumentParser(
        description="Train the AlgoMLB manager hook decision model."
    )
    parser.add_argument(
        "--version", default="v1.0", help="Model version tag (default: v1.0)"
    )
    parser.add_argument(
        "--output",
        default=".data/models",
        help="Output directory for the joblib bundle (default: .data/models)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip isotonic calibration (faster, less accurate probabilities)",
    )
    args = parser.parse_args()

    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]

    # 1. Load raw events
    df = load_hook_events(engine)
    if df.empty:
        logger.error(
            "No hook events found. Run scripts/backfill_manager_hooks.py first."
        )
        sys.exit(1)

    # 2. Derive binary labels
    df = derive_hook_labels(df)

    # 3. Build feature matrix + temporal dates
    X, y, dates = build_feature_matrix(df)

    if X.empty or y.nunique() < 2:
        logger.error(
            "Insufficient training data. Need both hooked (y=1) and "
            "completed (y=0) events. Check that multiple pitchers appeared "
            "per game in manager_hook_events."
        )
        sys.exit(1)

    # 4. Temporal split (no future data leakage)
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    train_mask = dates < cutoff
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[~train_mask], y[~train_mask]

    logger.info(
        f"Temporal split at {TRAIN_CUTOFF}: "
        f"train={train_mask.sum():,} | val={(~train_mask).sum():,}"
    )

    if len(X_train) < 50:
        logger.warning(
            f"Only {len(X_train)} training examples — model reliability may be low. "
            "Check that manager_hook_events has been backfilled."
        )

    # 5. Train
    model = HookModel()
    calibrate = not args.no_calibrate
    logger.info(
        f"Fitting HookModel (calibrate={calibrate}, n_estimators=200, max_depth=5)..."
    )
    model.fit(X_train, y_train, calibrate=calibrate)
    logger.success("Model fit complete.")

    # 6. Validate
    metrics: dict = {}
    if len(X_val) > 0 and y_val.nunique() == 2:
        metrics = evaluate(model, X_val, y_val)
        metrics["n_train"] = int(train_mask.sum())
        logger.success(f"Validation metrics → {metrics}")
    else:
        logger.warning(
            "Skipping validation: insufficient held-out data or single class in y_val."
        )

    # 7. Persist bundle
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hook_model_{args.version}.joblib"
    model.save(out_path)
    logger.success(f"Hook model saved → {out_path}")

    return metrics


if __name__ == "__main__":
    main()
