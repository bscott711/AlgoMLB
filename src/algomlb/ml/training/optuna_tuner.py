"""
Optuna hyperparameter optimization for AlgoMLB.
Implements walk-forward validation and objective routing based on model target.
"""

from typing import List, Optional
from datetime import datetime
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
from xgboost import XGBClassifier

from algomlb.core.logger import logger
from algomlb.ml.training.backtester import TimeSeriesSplitter


class XGBoostOptunaObjective:
    """
    Optuna objective that minimizes average loss across walk-forward folds.
    Routes to mlogloss for multi-class targets and brier_score for binary props.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        splitter: TimeSeriesSplitter,
        metric: str = "auto",
    ):
        self.df = df.copy()
        self.features = features
        self.target = target
        self.splitter = splitter

        if metric == "auto":
            self.metric = "mlogloss" if target == "pa_outcome" else "brier_score"
        else:
            self.metric = metric

        # Hardening: Fit LabelEncoder once globally for multiclass
        self.le = None
        self.num_class = None
        if self.metric == "mlogloss":
            from sklearn.preprocessing import LabelEncoder

            self.le = LabelEncoder()
            self.df[self.target] = self.le.fit_transform(self.df[self.target])
            # Ensure self.le is not None for Pyright
            if self.le.classes_ is not None:
                self.num_class = len(self.le.classes_)

    def __call__(self, trial: optuna.Trial) -> float:
        # Define search space strictly to prevent overfitting baseball noise
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0,
        }

        # Handle objective and classes based on target
        if self.metric == "mlogloss":
            params["objective"] = "multi:softprob"
            params["num_class"] = self.num_class
        else:
            params["objective"] = "binary:logistic"

        folds = self.splitter.split(self.df)
        if not folds:
            raise ValueError("No temporal folds generated for objective evaluation.")

        fold_losses = []

        for i, (train_df, test_df) in enumerate(folds):
            X_train, y_train = train_df[self.features], train_df[self.target]
            X_test, y_test = test_df[self.features], test_df[self.target]

            # Initialize and fit model
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

            # Predict probabilities
            y_prob = model.predict_proba(X_test)

            # Calculate step loss
            if self.metric == "mlogloss":
                # Explicitly pass all labels to handle folds missing rare outcomes
                labels = (
                    np.arange(self.num_class) if self.num_class is not None else None
                )
                step_loss = log_loss(y_test, y_prob, labels=labels)
            else:
                # y_prob is (N, 2), we want P(class=1)
                step_loss = brier_score_loss(y_test, y_prob[:, 1])

            fold_losses.append(step_loss)

            # Report intermediate values for pruning
            trial.report(step_loss, i)

            # Prune if the current trial is unpromising
            if trial.should_prune():
                raise optuna.TrialPruned()

        avg_loss = float(np.mean(fold_losses))
        return avg_loss


def run_optuna_study(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    n_trials: int = 100,
    storage_url: str = "sqlite:///models/optuna_history.db",
    study_name: Optional[str] = None,
) -> optuna.Study:
    """
    Initializes and runs an Optuna study for the given target.
    """
    if study_name is None:
        # Standardized naming convention: {target}_{version}
        # version is usually passed from CLI, here we use a timestamp as default if missing
        study_name = f"{target}_v{datetime.now().strftime('%Y%m%d')}"

    # Setup pruner: MedianPruner is robust for cross-fold metrics
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
    )

    splitter = TimeSeriesSplitter()
    objective = XGBoostOptunaObjective(df, features, target, splitter)

    logger.info(f"Starting Optuna optimization session: {study_name}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.success(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters: {study.best_params}")

    return study
