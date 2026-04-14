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
            n_jobs=params.get("n_jobs", 1),
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
                    "n_jobs",
                ]
            },
        )
        self.calibrated_clf = None
        self.le = None

    def fit(self, X: pd.DataFrame, y: pd.Series, calibrate: bool = True) -> None:
        """
        Fit the classifier and optionally apply Isotonic Calibration.
        Note: True temporal splitting should be handled by the caller or pipeline.
        """
        y_encoded = y
        if not np.issubdtype(y.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder

            self.le = LabelEncoder()
            y_encoded = self.le.fit_transform(y)

        if calibrate:
            # For very small data (e.g. tests), use 2-fold; otherwise 5-fold
            n_folds = 5 if len(X) >= 10 else 2
            self.calibrated_clf = CalibratedClassifierCV(
                self.clf, method="isotonic", cv=n_folds
            )
            self.calibrated_clf.fit(X, y_encoded)
        else:
            self.clf.fit(X, y_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities."""
        if self.calibrated_clf:
            return self.calibrated_clf.predict_proba(X)
        return self.clf.predict_proba(X)

    def get_base_xgb_estimator(self):
        """
        Extract the raw XGBClassifier from inside CalibratedClassifierCV.
        Needed for SHAP TreeExplainer which requires a native tree model.
        """
        if self.calibrated_clf is not None:
            # CalibratedClassifierCV stores fitted sub-estimators
            calibrated_classifiers = getattr(
                self.calibrated_clf, "calibrated_classifiers_", None
            )
            if calibrated_classifiers:
                return calibrated_classifiers[0].estimator
        return self.clf

    def get_feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of (feature, importance) from the base XGB estimator."""
        estimator = self.get_base_xgb_estimator()
        if not hasattr(estimator, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])
        names = getattr(estimator, "feature_names_in_", None)
        if names is None:
            names = [f"f{i}" for i in range(len(estimator.feature_importances_))]
        return pd.DataFrame(
            {
                "feature": names,
                "importance": estimator.feature_importances_,
            }
        )

    def save(self, file_path: Path) -> None:
        """Persist the model bundle (including calibration and label encoding) to disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "clf": self.clf,
            "calibrated_clf": self.calibrated_clf,
            "le": self.le,
        }
        joblib.dump(bundle, file_path)

    @classmethod
    def load(cls, file_path: Path) -> MLBModel:
        """Reconstruct a model and its calibration from a joblib bundle."""
        bundle = joblib.load(file_path)
        model = cls()
        model.clf = bundle["clf"]
        model.calibrated_clf = bundle.get("calibrated_clf")
        model.le = bundle.get("le")
        return model
