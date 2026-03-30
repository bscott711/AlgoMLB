from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


class MLBModel:
    """XGBoost wrapper for MLB win probability classification."""

    def __init__(self, **params):
        # n_estimators=100, max_depth=3, etc.
        self.clf = XGBClassifier(**params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the classifier for home team win prediction."""
        self.clf.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return win probabilities for [away_team_win, home_team_win]."""
        return self.clf.predict_proba(X)

    def save(self, file_path: Path) -> None:
        """Persist the model bundle to disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, file_path)

    @classmethod
    def load(cls, file_path: Path) -> MLBModel:
        """Reconstruct a model from a saved joblib bundle."""
        model = cls()
        model.clf = joblib.load(file_path)
        return model
