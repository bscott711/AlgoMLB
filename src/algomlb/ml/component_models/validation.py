import pandas as pd
from sklearn.metrics import log_loss


class TemporalLeakageError(Exception):
    """Raised when test data precedes or overlaps with training data."""

    pass


class ComponentEvaluator:
    """
    Evaluates ML components enforcing strict walk-forward (time-aware) validation rules.
    """

    @staticmethod
    def check_temporal_leakage(
        train_df: pd.DataFrame, test_df: pd.DataFrame, date_col: str = "game_date"
    ):
        """Strictly enforces that test data occurs chronologically after train data."""
        max_train_date = pd.to_datetime(train_df[date_col]).max()
        min_test_date = pd.to_datetime(test_df[date_col]).min()

        if max_train_date >= min_test_date:
            raise TemporalLeakageError(
                f"Temporal Leakage Detected! Max train date ({max_train_date.date()}) "
                f"is >= min test date ({min_test_date.date()}). Random CV is prohibited."
            )

    def evaluate_walk_forward(
        self,
        model,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        X_cols: list,
        y_col: str,
        date_col: str = "game_date",
    ) -> dict:
        """
        Evaluates the model using log-loss while enforcing temporal constraints.
        """
        self.check_temporal_leakage(train_df, test_df, date_col)

        X_train, y_train = train_df[X_cols], train_df[y_col]
        X_test, y_test = test_df[X_cols], test_df[y_col]

        model.train(X_train, y_train)

        probs = model.model.predict_proba(X_test)
        y_test_encoded = model.label_encoder.transform(y_test)

        metrics = {
            "mlogloss": log_loss(y_test_encoded, probs),
            # Additional metric functions (Brier, Reliability) can be appended here
        }
        return metrics
