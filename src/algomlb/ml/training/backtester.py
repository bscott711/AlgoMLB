"""
Granular temporal backtesting for AlgoMLB models.
Ensures strict chronological splitting and accumulates unbiased OOF predictions.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, List, Tuple, Protocol

import numpy as np
import pandas as pd
from algomlb.core.logger import logger


@dataclass
class TimeSplitConfig:
    """Config for temporal splitting logic."""

    train_window_days: int = 730  # 2 years
    test_window_days: int = 30  # 1 month
    buffer_days: int = 0  # Gap between train and test to prevent leakage


class TimeSeriesSplitter:
    """
    Handles day-based temporal splitting of game-level datasets.
    Prevents any form of future-lookahead or data leakage.
    """

    def __init__(self, config: TimeSplitConfig = TimeSplitConfig()):
        self.config = config

    def split(
        self, df: pd.DataFrame, date_col: str = "game_date"
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Iteratively expands the training window and slides the test window.

        Returns:
            List of (train_df, test_df) tuples.
        """
        if df.empty:
            return []

        # Ensure date column is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)

        min_date = df[date_col].min()
        max_date = df[date_col].max()

        splits = []
        # First train window ends after train_window_days
        current_train_end = min_date + timedelta(days=self.config.train_window_days)

        while (
            current_train_end + timedelta(days=self.config.test_window_days) <= max_date
        ):
            test_start = current_train_end + timedelta(days=self.config.buffer_days)
            test_end = test_start + timedelta(days=self.config.test_window_days)

            train_df = df[df[date_col] < current_train_end]
            test_df = df[(df[date_col] >= test_start) & (df[date_col] < test_end)]

            if not train_df.empty and not test_df.empty:
                # Strict chronological assertion
                assert train_df[date_col].max() < test_df[date_col].min(), (
                    f"Temporal leakage detected! Train max: {train_df[date_col].max()}, Test min: {test_df[date_col].min()}"
                )

                splits.append((train_df, test_df))
                logger.debug(
                    f"Fold created: Train ends {current_train_end.date()}, Test [{test_start.date()} to {test_end.date()})"
                )

            # Advance the train window end by the test window size (Expanding Window)
            current_train_end = test_end

        logger.info(
            f"Generated {len(splits)} walk-forward folds from {min_date.date()} to {max_date.date()}"
        )
        return splits


def calculate_20bin_ece(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculates Expected Calibration Error (ECE) using the 20-bin standard.

    ECE is the weighted average of the absolute difference between
    predicted probabilities and actual win rates within each bin.
    """
    n_bins = 20
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    # Use digitize to find which bin each probability belongs to (1-indexed)
    binids = np.digitize(y_prob, bins) - 1
    # Handle the probability = 1.0 edge case (it falls into bin index 20, map to 19)
    binids[y_prob == 1.0] = n_bins - 1

    ece = 0.0
    total_samples = len(y_true)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_mask = binids == i
        if np.any(bin_mask):
            bin_count = np.sum(bin_mask)
            bin_prob_mean = np.mean(y_prob[bin_mask])
            bin_true_mean = np.mean(y_true[bin_mask])
            ece += (bin_count / total_samples) * np.abs(bin_prob_mean - bin_true_mean)

    return float(ece)


class TrainableModel(Protocol):
    """Protocol for models used in the backtester."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class OOFAccumulator:
    """
    Orchestrates the backtesting of a model across temporal folds.
    Accumulates Out-Of-Fold (OOF) predictions for unbiased evaluation.
    """

    def __init__(self, model_class: Any, features: List[str], target: str):
        self.model_class = model_class
        self.features = features
        self.target = target

    def run_backtest(
        self, df: pd.DataFrame, splitter: TimeSeriesSplitter, **model_kwargs
    ) -> pd.DataFrame:
        """
        Runs walk-forward backtesting and returns a DataFrame of OOF predictions.
        """
        folds = splitter.split(df)
        all_oof_results = []

        for i, (train_df, test_df) in enumerate(folds):
            logger.info(f"Processing Fold {i + 1}/{len(folds)}...")

            # Setup data
            X_train, y_train = train_df[self.features], train_df[self.target]
            X_test = test_df[self.features]

            # Initialize and train model
            model = self.model_class(**model_kwargs)
            model.train(X_train, y_train)

            # Predict probabilities
            y_prob = model.predict_proba(X_test)

            # Store OOF results with original identifiers
            fold_results = test_df.copy()
            # If multi-class, store the full probability vector
            if y_prob.ndim > 1:
                fold_results["p_model"] = list(y_prob)
            else:
                fold_results["p_model"] = y_prob

            fold_results["fold_idx"] = i
            all_oof_results.append(fold_results)

        if not all_oof_results:
            logger.warning("No backtest results generated (zero folds).")
            return pd.DataFrame()

        return pd.concat(all_oof_results, ignore_index=True)
