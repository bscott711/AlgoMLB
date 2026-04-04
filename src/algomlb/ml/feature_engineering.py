"""
src/algomlb/ml/feature_engineering.py

Standardized feature engineering for Statcast coordinates and batted ball geometry.
Standard Statcast Transform (derived from spray_charts.py):
    CENTER_X = 125.42
    CENTER_Y = 198.27
    SCALE = 2.33
"""

from typing import Any
import numpy as np
import pandas as pd


def compute_spray_angle(hc_x: float | pd.Series, hc_y: float | pd.Series) -> Any:
    """
    Compute the spray angle in degrees from Statcast hc_x, hc_y.
    0 = Center Field, Negative = Left Field, Positive = Right Field.
    """
    CENTER_X = 125.42
    CENTER_Y = 198.27

    dx = hc_x - CENTER_X
    dy = CENTER_Y - hc_y

    # arctan2(x, y) gives the angle from the vertical (y-axis)
    rad = np.arctan2(dx, dy)
    return np.degrees(rad)


def add_spray_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized addition of spray_angle and is_rhb."""
    if "hc_x" in df.columns and "hc_y" in df.columns:
        df["spray_angle"] = compute_spray_angle(df["hc_x"], df["hc_y"])

    if "stand" in df.columns:
        df["is_rhb"] = (df["stand"] == "R").astype(int)

    return df
