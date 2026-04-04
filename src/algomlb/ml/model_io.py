"""
src/algomlb/ml/model_io.py

Persistence layer for Batted Ball Flight Decoupler assets.
Supports versioned XGBoost models and environmental coefficients.
"""

from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Dict, Optional

import xgboost as xgb


MODEL_DIR = Path("/home/opc/AlgoMLB/models")


def save_decoupler_assets(
    model: xgb.XGBRegressor, coeffs: Dict[str, float], version: str = "v1"
) -> None:
    """Save XGBoost model and coefficients to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"batted_ball_baseline_{version}.joblib"
    coeff_path = MODEL_DIR / f"env_coefficients_{version}.json"

    joblib.dump(model, model_path)

    with open(coeff_path, "w") as f:
        json.dump(coeffs, f, indent=4)


def load_decoupler_assets(
    version: str = "v1",
) -> tuple[Optional[xgb.XGBRegressor], Optional[Dict[str, float]]]:
    """Load XGBoost model and coefficients from disk."""
    model_path = MODEL_DIR / f"batted_ball_baseline_{version}.joblib"
    coeff_path = MODEL_DIR / f"env_coefficients_{version}.json"

    model = None
    if model_path.exists():
        model = joblib.load(model_path)

    coeffs = None
    if coeff_path.exists():
        with open(coeff_path, "r") as f:
            coeffs = json.load(f)

    return model, coeffs
