import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from typing import Dict


class PropCalibrator:
    """
    Trains and executes 'model-on-model' calibration to adjust raw Monte Carlo
    probabilities based on historical biases and global context.
    """

    def __init__(self, method: str = "logistic"):
        """
        method: 'logistic' for multivariate context features,
                'isotonic' for 1D mapping of just the simulator probability.
        """
        self.method = method
        if self.method == "logistic":
            # Logistic regression provides smooth, multivariate calibration
            self.model = LogisticRegression(class_weight="balanced", random_state=42)
        elif self.method == "isotonic":
            # Isotonic requires 1D input but guarantees a monotonically increasing fit
            self.model = IsotonicRegression(out_of_bounds="clip")
        else:
            raise ValueError("Method must be 'logistic' or 'isotonic'.")

    def train(self, X_cal: pd.DataFrame, y_cal: pd.Series):
        """Fits the calibration model to historical simulation outputs vs reality."""
        if self.method == "isotonic":
            # Isotonic only uses the raw MC probability (assumed to be the first column)
            self.model.fit(X_cal.iloc[:, 0], y_cal)
        else:
            self.model.fit(X_cal, y_cal)

    def predict_p_over(self, X_cal: pd.DataFrame) -> np.ndarray:
        """Returns the calibrated probability of the prop going OVER."""
        if self.method == "isotonic":
            # IsotonicRegression.predict returns 1D array
            return self.model.predict(X_cal.iloc[:, 0])  # type: ignore
        else:
            # LogisticRegression.predict_proba returns 2D array
            return self.model.predict_proba(X_cal)[:, 1]  # type: ignore

    @staticmethod
    def calculate_ece(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculates Expected Calibration Error (ECE)."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1

        ece = 0.0
        for i in range(n_bins):
            bin_mask = binids == i
            if np.any(bin_mask):
                bin_count = np.sum(bin_mask)
                bin_prob = np.mean(y_prob[bin_mask])
                bin_true = np.mean(y_true[bin_mask])
                ece += (bin_count / len(y_true)) * np.abs(bin_prob - bin_true)
        return float(ece)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluates calibration quality using Brier Score and ECE."""
        y_prob = self.predict_p_over(X_test)

        return {
            "brier_score": float(brier_score_loss(y_test, y_prob)),
            "ece": self.calculate_ece(y_test.to_numpy(), y_prob),
        }
