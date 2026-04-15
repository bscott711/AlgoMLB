"""
src/algomlb/ml/hook_model.py

XGBoost binary classifier for manager pitcher-change (hook) decisions.

Architecture mirrors MLBModel in model.py:
  - XGBClassifier + isotonic CalibratedClassifierCV
  - joblib bundle format: {clf, calibrated_clf, feature_names}
  - FEATURE_NAMES class constant locks the training/inference schema

Label convention: 1 = pitcher was removed (hooked), 0 = completed outing.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


# ── Leverage Index Approximation ─────────────────────────────────────────────
#
# Derived from Tangotiger's LI tables (simplified 24-cell base-out approximation).
# Maps (outs, base_state_bitmask) → base leverage before inning/score scaling.
#
# base_state bitmask: bit0=1B occupied, bit1=2B occupied, bit2=3B occupied.
#   0 = bases empty  1 = man on 1st  2 = man on 2nd  3 = 1st & 2nd
#   4 = man on 3rd   5 = 1st & 3rd   6 = 2nd & 3rd   7 = bases loaded

_LI_BASE_TABLE: dict[tuple[int, int], float] = {
    # (outs, base_state) → base_li
    (0, 0): 0.52, (0, 1): 0.96, (0, 2): 1.04, (0, 3): 1.57,
    (0, 4): 1.39, (0, 5): 1.69, (0, 6): 1.63, (0, 7): 2.09,
    (1, 0): 0.62, (1, 1): 1.14, (1, 2): 1.25, (1, 3): 1.91,
    (1, 4): 1.72, (1, 5): 2.07, (1, 6): 2.04, (1, 7): 2.71,
    (2, 0): 0.67, (2, 1): 1.04, (2, 2): 1.17, (2, 3): 1.62,
    (2, 4): 1.46, (2, 5): 1.74, (2, 6): 1.67, (2, 7): 2.01,
}

# Inning urgency multiplier: hook decisions become higher-stakes late in games.
_INNING_LI_SCALE: dict[int, float] = {
    1: 0.70, 2: 0.72, 3: 0.74, 4: 0.77,
    5: 0.82, 6: 0.92, 7: 1.10, 8: 1.40, 9: 1.70,
}


def compute_leverage_index(
    inning: int,
    outs: int,
    base_state: int,
    score_diff: int,
) -> float:
    """
    Approximate Leverage Index from base-out state, inning, and score differential.

    Uses a simplified 24-cell RE table scaled by inning urgency and game closeness.
    Returns values roughly in [0.10, 5.0], calibrated against historical LI
    distributions from FanGraphs and Retrosheet RE24 data.

    Args:
        inning:     Current inning (1-based). Values >9 use the inning-9 scale.
        outs:       Outs in current half-inning (0, 1, or 2; clamped).
        base_state: Bitmask (bit0=1B, bit1=2B, bit2=3B). Range [0, 7].
        score_diff: Pitching-team score minus opponent score (positive = leading).

    Returns:
        Approximate LI as a float, rounded to 3 decimal places.
    """
    outs_clamped = min(max(outs, 0), 2)
    base_clamped = base_state & 0b111
    base_li = _LI_BASE_TABLE.get((outs_clamped, base_clamped), 1.0)

    # Inning urgency: extra innings treated like the 9th
    inning_scale = _INNING_LI_SCALE.get(min(inning, 9), 1.70)

    # Score closeness multiplier — close games dramatically raise stakes
    abs_diff = abs(score_diff)
    if abs_diff == 0:
        score_scale = 1.20
    elif abs_diff == 1:
        score_scale = 1.00
    elif abs_diff == 2:
        score_scale = 0.80
    elif abs_diff == 3:
        score_scale = 0.55
    else:
        score_scale = 0.30

    return round(base_li * inning_scale * score_scale, 3)


# ── HookModel ─────────────────────────────────────────────────────────────────


class HookModel:
    """
    XGBoost binary classifier for manager pitcher-removal decisions.

    Predicts the probability that a pitcher will be removed (hooked) given
    the current game state and in-game pitcher fatigue metrics.

    Binary label: 1 = pitcher was removed (hooked mid-game),
                  0 = pitcher completed their outing or finished the game.

    FEATURE_NAMES is the authoritative locked schema. Any input DataFrame is
    aligned to this schema via reindex before training or inference, ensuring
    training/inference parity regardless of column ordering or extras.

    Usage::

        # Training
        model = HookModel()
        model.fit(X_train, y_train)
        model.save(Path(".data/models/hook_model_v1.0.joblib"))

        # Inference
        model = HookModel.load(Path(".data/models/hook_model_v1.0.joblib"))
        prob = model.predict_proba(features_df)[0][1]
    """

    FEATURE_NAMES: List[str] = [
        "inning",
        "outs_at_hook",
        "pitches_thrown",
        "tto_at_hook",
        "score_diff_at_hook",
        "base_state_at_hook",
        "leverage_index_at_hook",
        "is_starter",
    ]

    def __init__(self, **params: float) -> None:
        self.clf = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 5)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            random_state=42,
            eval_metric="logloss",
            n_jobs=int(params.get("n_jobs", 1)),
        )
        self.calibrated_clf: Optional[CalibratedClassifierCV] = None

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enforce FEATURE_NAMES schema: reorder, drop extras, fill missing with 0."""
        return X.reindex(columns=self.FEATURE_NAMES, fill_value=0.0)

    def fit(self, X: pd.DataFrame, y: pd.Series, calibrate: bool = True) -> None:
        """
        Fit the hook classifier with optional isotonic probability calibration.

        Args:
            X:         Feature DataFrame. Aligned to FEATURE_NAMES automatically.
            y:         Binary Series (1 = hooked, 0 = completed).
            calibrate: Apply CalibratedClassifierCV (isotonic) to improve
                       probability estimates for stochastic simulation sampling.
        """
        X_aligned = self._align(X)
        if calibrate:
            n_folds = 5 if len(X_aligned) >= 20 else 2
            self.calibrated_clf = CalibratedClassifierCV(
                self.clf, method="isotonic", cv=n_folds
            )
            self.calibrated_clf.fit(X_aligned, y)
        else:
            self.clf.fit(X_aligned, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return calibrated hook probabilities.

        Returns:
            Array of shape (n_samples, 2): [:, 0] = P(not hooked), [:, 1] = P(hooked).
        """
        X_aligned = self._align(X)
        if self.calibrated_clf is not None:
            return self.calibrated_clf.predict_proba(X_aligned)
        return self.clf.predict_proba(X_aligned)

    def save(self, path: Path) -> None:
        """Persist model bundle (classifier + calibration + schema) to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "clf": self.clf,
            "calibrated_clf": self.calibrated_clf,
            "feature_names": self.FEATURE_NAMES,
        }
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: Path) -> "HookModel":
        """
        Reconstruct a HookModel from a joblib bundle.

        Emits a UserWarning if the bundle's feature schema differs from the
        current FEATURE_NAMES (signals a need to retrain).
        """
        bundle = joblib.load(path)
        instance = cls()
        instance.clf = bundle["clf"]
        instance.calibrated_clf = bundle.get("calibrated_clf")

        saved_features = bundle.get("feature_names", cls.FEATURE_NAMES)
        if saved_features != cls.FEATURE_NAMES:
            warnings.warn(
                f"Loaded hook model feature schema differs from current FEATURE_NAMES. "
                f"Saved: {saved_features}. Current: {cls.FEATURE_NAMES}. "
                "Re-train the hook model to realign schemas.",
                UserWarning,
                stacklevel=2,
            )
        return instance
