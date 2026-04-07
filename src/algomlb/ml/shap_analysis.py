"""SHAP-based explainability for Uranium XGBoost models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from algomlb.core.logger import logger
from algomlb.db.session import get_engine


def compute_global_shap(
    model,  # MLBModel
    X: pd.DataFrame,
    sample_n: int = 5000,
) -> pd.DataFrame:
    """
    Compute mean |SHAP| and mean SHAP for each feature using TreeExplainer.

    Parameters
    ----------
    model : MLBModel
        A trained MLBModel (with or without calibration).
    X : pd.DataFrame
        Feature matrix (test or train).
    sample_n : int
        Max rows to sample for speed.

    Returns
    -------
    pd.DataFrame with columns: feature_name, mean_abs_shap, mean_shap
    """
    try:
        import shap
    except ImportError:
        logger.warning(
            "shap package not installed — skipping SHAP computation. "
            "Install with: uv add shap"
        )
        return pd.DataFrame(columns=["feature_name", "mean_abs_shap", "mean_shap"])

    # Subsample for performance
    if len(X) > sample_n:
        X_sample = X.sample(n=sample_n, random_state=42)
    else:
        X_sample = X

    # Extract the raw XGBoost estimator
    base_estimator = model.get_base_xgb_estimator()

    logger.info(f"Computing SHAP values on {len(X_sample)} samples, {X_sample.shape[1]} features...")
    explainer = shap.TreeExplainer(base_estimator)
    shap_values = explainer.shap_values(X_sample)

    # shap_values may be a list (multiclass) or ndarray (binary)
    if isinstance(shap_values, list):
        # For binary classification, take the positive class
        sv = np.array(shap_values[1])
    else:
        sv = np.array(shap_values)

    # Aggregate
    mean_abs = np.abs(sv).mean(axis=0)
    mean_val = sv.mean(axis=0)

    result = pd.DataFrame({
        "feature_name": X_sample.columns.tolist(),
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_val,
    })

    return result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def persist_global_shap(
    engine: Engine | None,
    model_version: str,
    dataset_label: str,
    shap_df: pd.DataFrame,
) -> None:
    """Upsert global SHAP values into uranium_shap_global."""
    if shap_df.empty:
        logger.warning("Empty SHAP DataFrame — skipping persistence.")
        return

    eng = engine or get_engine()
    from algomlb.db.models import UraniumShapGlobalORM

    records = shap_df.copy()
    records["model_version"] = model_version
    records["dataset_label"] = dataset_label
    rows = records.to_dict(orient="records")

    with eng.begin() as conn:
        for row in rows:
            stmt = pg_insert(UraniumShapGlobalORM.__table__).values([row])
            upsert = stmt.on_conflict_do_update(
                index_elements=["model_version", "dataset_label", "feature_name"],
                set_={
                    "mean_abs_shap": stmt.excluded.mean_abs_shap,
                    "mean_shap": stmt.excluded.mean_shap,
                },
            )
            conn.execute(upsert)

    logger.info(
        f"Persisted {len(rows)} SHAP features for {model_version}/{dataset_label}."
    )
