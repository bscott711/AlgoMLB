import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel, ConfigDict
from typing import Dict


# Domain model stubs for typing (Normally imported from algomlb.domain.state_models)
class BatterPreGameState(BaseModel):
    batter_id: int
    model_config = ConfigDict(extra="allow")


class PitcherPreGameState(BaseModel):
    pitcher_id: int
    model_config = ConfigDict(extra="allow")


class PAOutcomeModel:
    """
    The core multinomial classification model that predicts the outcome
    of a single plate appearance. Feeds the Monte Carlo simulation engine.
    """

    # Strict canonical list. Order here does not dictate XGBoost's internal order,
    # which is why we explicitly use a LabelEncoder to lock it.
    CANONICAL_OUTCOMES = [
        "strikeout",
        "walk",
        "hbp",
        "single",
        "double",
        "triple",
        "home_run",
        "out_in_play",
    ]

    def __init__(self, **xgb_kwargs):
        self.label_encoder = LabelEncoder()
        # Force a deterministic integer mapping (0 to 7)
        self.label_encoder.fit(self.CANONICAL_OUTCOMES)

        # Store the mapping explicitly so the MC engine knows index 0 = 'double', etc.
        classes = self.label_encoder.classes_
        self.class_mapping_ = {idx: label for idx, label in enumerate(classes)}  # type: ignore

        default_params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": len(self.CANONICAL_OUTCOMES),
            "use_label_encoder": False,  # We handle encoding explicitly
        }
        default_params.update(xgb_kwargs)
        self.model = XGBClassifier(**default_params)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Trains the XGBoost classifier on the feature matrix and targets."""
        # Ensure all incoming labels are strictly in the canonical list
        if not set(y.unique()).issubset(set(self.CANONICAL_OUTCOMES)):
            raise ValueError("Target array contains unrecognized PA outcomes.")

        y_encoded = self.label_encoder.transform(y)
        self.model.fit(X, y_encoded)

    def predict_matchup(
        self,
        batter_state: BatterPreGameState,
        pitcher_state: PitcherPreGameState,
        context: Dict,
    ) -> np.ndarray:
        """
        Generates a probability distribution for a single PA.
        Returns an array of shape (1, 8) where sum(array) == 1.0.
        """
        # Collapse the states into a single prediction row
        row_dict = {
            **batter_state.model_dump(),
            **pitcher_state.model_dump(),
            **context,
        }
        X_pred = pd.DataFrame([row_dict])

        # Select only the features the model was trained on
        X_pred = X_pred[self.model.feature_names_in_]

        probs = self.model.predict_proba(X_pred)
        return probs
