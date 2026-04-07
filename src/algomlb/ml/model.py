from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV


class MLBModel:
    """XGBoost wrapper for MLB win probability classification with Isotonic Calibration."""

    def __init__(self, **params):
        # Pop monotone_constraints before building the generic kwargs
        mono = params.pop("monotone_constraints", None)

        # Default params for class imbalance and regularization
        # scale_pos_weight handles the slight Away-win bias if necessary
        self.clf = XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            scale_pos_weight=params.get("scale_pos_weight", 1.0),
            monotone_constraints=mono,
            random_state=42,
            **{
                k: v
                for k, v in params.items()
                if k
                not in [
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "scale_pos_weight",
                ]
            },
        )
        self.calibrated_clf = None


    def train(self, X: pd.DataFrame, y: pd.Series, calibrate: bool = True) -> None:
        """
        Fit the classifier and optionally apply Isotonic Calibration.
        Note: True temporal splitting should be handled by the caller or pipeline.
        """
        if calibrate:
            # For very small data (e.g. tests), use 2-fold; otherwise 5-fold
            n_folds = 5 if len(X) >= 10 else 2
            self.calibrated_clf = CalibratedClassifierCV(
                self.clf, method="isotonic", cv=n_folds
            )
            self.calibrated_clf.fit(X, y)
        else:
            self.clf.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated win probabilities for [away_team_win, home_team_win]."""
        if self.calibrated_clf:
            return self.calibrated_clf.predict_proba(X)
        return self.clf.predict_proba(X)

    def save(self, file_path: Path) -> None:
        """Persist the model bundle (including calibration) to disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {"clf": self.clf, "calibrated_clf": self.calibrated_clf}
        joblib.dump(bundle, file_path)

    @classmethod
    def load(cls, file_path: Path) -> MLBModel:
        """Reconstruct a model and its calibration from a joblib bundle."""
        bundle = joblib.load(file_path)
        model = cls()
        model.clf = bundle["clf"]
        model.calibrated_clf = bundle.get("calibrated_clf")
        return model
