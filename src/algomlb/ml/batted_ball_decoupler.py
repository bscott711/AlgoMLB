"""
src/algomlb/ml/batted_ball_decoupler.py

Main engine for the Batted Ball Flight Decoupler.
Integrates XGBoost baseline with three-component environmental physics.
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb
from typing import Dict, Optional

from algomlb.ml.feature_engineering import add_spray_features
from algomlb.ml.weather_features import add_weather_features
from algomlb.ml.coefficients import calibrate_coefficients
from algomlb.ml.model_io import save_decoupler_assets, load_decoupler_assets


class BattedBallFlightDecoupler:
    """
    Decouples environmental effects from fly ball carry distance.
    Isolates 'spin_contact_factor' as the final residual.
    """

    def __init__(self, version: str = "v1"):
        self.version = version
        self.model: Optional[xgb.XGBRegressor] = None
        self.coeffs: Optional[Dict[str, float]] = None

        # Hyperparameters from Lead Architect's plan
        self.xgb_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": 42,
        }

    def train_baseline(self, train_df: pd.DataFrame) -> None:
        """
        Train XGBoost Regressor on neutral environmental BIP.
        Features: launch_speed, launch_angle, spray_angle, is_rhb.
        Target: hit_distance_sc.
        """
        df = add_spray_features(train_df.copy())

        # neutral filter: low wind, dry
        neutral_df = df[
            (df["wind_speed_mph"].fillna(0) < 3.0)
            & (df["precipitation_mm_hr"].fillna(0) < 0.05)
        ].copy()

        X = neutral_df[["launch_speed", "launch_angle", "spray_angle", "is_rhb"]]
        y = neutral_df["hit_distance_sc"]

        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X, y)

    def calibrate(
        self, val_df: pd.DataFrame, ballpark_coords: Optional[pd.DataFrame] = None
    ) -> None:
        """Fit β, γ, δ coefficients on validation data."""
        if self.model is None:
            raise ValueError(
                "Baseline model must be trained or loaded before calibration."
            )

        df = self.preprocess(val_df, ballpark_coords)

        # Calculate residuals (total_delta)
        X = df[["launch_speed", "launch_angle", "spray_angle", "is_rhb"]]
        df["baseline_distance"] = self.model.predict(X)
        df["total_delta"] = df["hit_distance_sc"] - df["baseline_distance"]

        self.coeffs = calibrate_coefficients(df)

    def preprocess(
        self, df: pd.DataFrame, ballpark_coords: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Apply full feature engineering and weather physics."""
        df = add_spray_features(df.copy())
        if ballpark_coords is not None:
            df = add_weather_features(df, ballpark_coords)
        return df

    def decouple(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core inference logic.
        1. Predict baseline carry.
        2. Calculate environmental deltas.
        3. Isolate spin/contact factor.
        """
        if self.model is None or self.coeffs is None:
            raise ValueError("Model and coefficients must be loaded before decoupling.")

        df = df.copy()
        X = df[["launch_speed", "launch_angle", "spray_angle", "is_rhb"]]

        # 1. Baseline
        df["baseline_distance"] = self.model.predict(X)
        df["total_delta"] = df["hit_distance_sc"] - df["baseline_distance"]

        # 2. Environmental Physics Components
        beta = self.coeffs["beta"]
        gamma = self.coeffs["gamma"]
        delta = self.coeffs["delta"]

        df["delta_density"] = beta * (1.0 - df["air_density_ratio"])
        df["delta_wind"] = gamma * df["tailwind_component"]
        df["delta_precip"] = delta * df["precipitation_mm_hr"]

        df["environmental_factor"] = (
            df["delta_density"] + df["delta_wind"] + df["delta_precip"]
        )

        # 3. Spin/Contact Factor (The Residual)
        df["spin_contact_factor"] = df["total_delta"] - df["environmental_factor"]

        return df

    def save(self) -> None:
        """Persist to disk."""
        if self.model and self.coeffs:
            save_decoupler_assets(self.model, self.coeffs, self.version)

    def load(self) -> bool:
        """Load from disk."""
        self.model, self.coeffs = load_decoupler_assets(self.version)
        return self.model is not None and self.coeffs is not None
