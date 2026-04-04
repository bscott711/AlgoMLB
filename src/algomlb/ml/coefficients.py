"""
src/algomlb/ml/coefficients.py

Sequential OLS calibration for environmental coefficients (β, γ, δ).
Each coefficient is isolated by subsetting to conditions where other
environmental effects are minimal.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.linear_model import LinearRegression


def calibrate_coefficients(
    df: pd.DataFrame,
    calm_wind_mph: float = 3.0,
    dry_precip_mm: float = 0.05,
    rainy_precip_mm: float = 0.1,
) -> Dict[str, float]:
    """
    Fit β, γ, δ from training data containing 'total_delta' (sc_distance - predicted_baseline).

    Calibration sequence:
    1. Fit β (density) on calm, dry days.
    2. Fit γ (wind) on all days after removing density effect.
    3. Fit δ (precip) on rainy, calm days after removing density + wind effect.
    """
    df = df.copy()

    # Required columns check
    required = [
        "total_delta",
        "air_density_ratio",
        "tailwind_component",
        "wind_speed_mph",
        "precipitation_mm_hr",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["density_deviation"] = 1.0 - df["air_density_ratio"]

    # Step 1 — β: density coefficient (ft per unit deviation)
    calm_dry = df[
        (df["wind_speed_mph"] < calm_wind_mph)
        & (df["precipitation_mm_hr"] < dry_precip_mm)
    ].copy()

    if len(calm_dry) < 100:
        beta = 400.0  # Physical fallback
    else:
        beta = float(
            LinearRegression(fit_intercept=False)
            .fit(calm_dry[["density_deviation"]], calm_dry["total_delta"])
            .coef_[0]
        )

    # Step 2 — γ: wind coefficient (ft per mph tailwind)
    df["delta_density"] = beta * df["density_deviation"]
    df["density_residual"] = df["total_delta"] - df["delta_density"]

    gamma = float(
        LinearRegression(fit_intercept=False)
        .fit(df[["tailwind_component"]], df["density_residual"])
        .coef_[0]
    )

    # Step 3 — δ: precipitation coefficient (ft per mm/hr precip)
    df["delta_wind"] = gamma * df["tailwind_component"]
    df["env_residual"] = df["density_residual"] - df["delta_wind"]

    rainy_calm = df[
        (df["precipitation_mm_hr"] > rainy_precip_mm)
        & (df["wind_speed_mph"] < calm_wind_mph)
    ].copy()

    if len(rainy_calm) < 100:
        delta = -5.0  # Physical fallback (drag increases with rain)
    else:
        delta = float(
            LinearRegression(fit_intercept=False)
            .fit(rainy_calm[["precipitation_mm_hr"]], rainy_calm["env_residual"])
            .coef_[0]
        )

    return {"beta": beta, "gamma": gamma, "delta": delta}
