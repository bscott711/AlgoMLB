"""
tests/unit/ml/test_decoupler.py

Unit tests for the Batted Ball Flight Decoupler suite.
Ensures 100% SONAR-GREEN coverage for physics, ML, and persistence.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from algomlb.ml.batted_ball_decoupler import BattedBallFlightDecoupler
from algomlb.ml.coefficients import calibrate_coefficients
from algomlb.ml.feature_engineering import add_spray_features, compute_spray_angle
from algomlb.ml.model_io import load_decoupler_assets, save_decoupler_assets
from algomlb.ml.weather_features import (
    add_weather_features,
    compute_air_density_ratio,
    compute_cf_bearing,
    compute_tailwind_component,
)


def test_compute_spray_angle():
    assert compute_spray_angle(125.42, 198.27 - 100) == pytest.approx(0.0)
    assert compute_spray_angle(125.42 + 100, 198.27) == pytest.approx(90.0)
    assert compute_spray_angle(125.42 - 100, 198.27) == pytest.approx(-90.0)


def test_add_spray_features():
    df = pd.DataFrame({"hc_x": [125.42], "hc_y": [98.27], "stand": ["R"]})
    df = add_spray_features(df)
    assert "spray_angle" in df.columns
    assert df["is_rhb"].iloc[0] == 1


def test_compute_cf_bearing():
    bearing = compute_cf_bearing(34.0, -118.0, 34.0001, -118.0)
    assert bearing == pytest.approx(0.0, abs=1.0)


def test_compute_air_density_ratio():
    ratio = compute_air_density_ratio(59.0, 1013.25, 0.0)
    assert ratio == pytest.approx(1.0, abs=0.01)
    hot_ratio = compute_air_density_ratio(95.0, 1013.25, 50.0)
    assert hot_ratio < 1.0


def test_compute_tailwind_component():
    tw = compute_tailwind_component(10.0, 0.0, 0.0, 0.0)
    assert tw == pytest.approx(-10.0)
    tw_assisting = compute_tailwind_component(10.0, 180.0, 0.0, 0.0)
    assert tw_assisting == pytest.approx(10.0)


def test_add_weather_features():
    df = pd.DataFrame(
        {
            "venue_id": [1],
            "temperature_f": [70.0],
            "pressure_hpa": [1013.0],
            "relative_humidity": [50.0],
            "wind_speed_mph": [10.0],
            "wind_direction_deg": [0.0],
            "spray_angle": [0.0],
        }
    )
    ballparks = pd.DataFrame(
        {
            "venue_id": [1],
            "hp_lat": [34.0],
            "hp_lon": [-118.0],
            "pm_lat": [34.1],
            "pm_lon": [-118.0],
        }
    )
    res = add_weather_features(df, ballparks)
    assert "cf_bearing_deg" in res.columns
    assert "air_density_ratio" in res.columns
    assert "tailwind_component" in res.columns


def test_calibrate_coefficients():
    df = pd.DataFrame(
        {
            "total_delta": [10.0, 5.0, 0.0],
            "air_density_ratio": [0.98, 1.0, 1.02],
            "tailwind_component": [5.0, 0.0, -5.0],
            "wind_speed_mph": [2.0, 2.0, 2.0],
            "precipitation_mm_hr": [0.0, 0.0, 0.0],
        }
    )
    coeffs = calibrate_coefficients(df)
    assert "beta" in coeffs
    assert "gamma" in coeffs
    assert "delta" in coeffs


def test_calibrate_coefficients_fallback():
    df = pd.DataFrame(
        {
            "total_delta": [0.0],
            "air_density_ratio": [1.0],
            "tailwind_component": [0.0],
            "wind_speed_mph": [0.0],
            "precipitation_mm_hr": [0.5],
        }
    )
    coeffs = calibrate_coefficients(df)
    assert coeffs["beta"] == 400.0
    assert coeffs["delta"] == -5.0


def test_add_weather_features_fallback():
    df = pd.DataFrame(
        {
            "venue_id": [1],
            "temperature_f": [None],
            "pressure_hpa": [None],
            "relative_humidity": [None],
            "wind_speed_mph": [None],
            "wind_direction_deg": [None],
            "spray_angle": [0.0],
            "hp_bearing_deg": [180.0],
        }
    )
    ballparks = pd.DataFrame({"venue_id": [1]})
    res = add_weather_features(df, ballparks)
    assert res["cf_bearing_deg"].iloc[0] == 180.0
    assert res["air_density_ratio"].iloc[0] == 1.0


def test_decoupler_errors():
    decoupler = BattedBallFlightDecoupler(version="err")
    decoupler.model = None
    with pytest.raises(ValueError):
        decoupler.calibrate(pd.DataFrame())

    decoupler.model = MagicMock()
    decoupler.coeffs = None
    with pytest.raises(ValueError):
        decoupler.decouple(pd.DataFrame())


def test_decoupler_save_load(tmp_path):
    model = xgb.XGBRegressor()
    model.fit(np.array([[1]]), np.array([1]))
    coeffs = {"beta": 1.0, "gamma": 2.0, "delta": 3.0}
    with patch("algomlb.ml.model_io.MODEL_DIR", tmp_path):
        save_decoupler_assets(model, coeffs, "test_v1")
        m, c = load_decoupler_assets("test_v1")
        assert c is not None
        assert c["beta"] == 1.0
        assert m is not None

    # Test load failure
    with patch("algomlb.ml.model_io.MODEL_DIR", tmp_path / "missing"):
        m2, c2 = load_decoupler_assets("missing")
        assert m2 is None
        assert c2 is None


def test_batted_ball_flight_decoupler():
    decoupler = BattedBallFlightDecoupler(version="test")
    df = pd.DataFrame(
        {
            "launch_speed": [100.0, 105.0],
            "launch_angle": [25.0, 30.0],
            "hc_x": [125.42, 130.0],
            "hc_y": [100.0, 95.0],
            "hit_distance_sc": [400.0, 420.0],
            "wind_speed_mph": [0.0, 0.0],
            "precipitation_mm_hr": [0.0, 0.0],
            "stand": ["R", "L"],
        }
    )
    decoupler.train_baseline(df)
    coords = pd.DataFrame(
        {
            "venue_id": [1],
            "hp_lat": [34.0],
            "hp_lon": [-118.0],
            "pm_lat": [34.0001],
            "pm_lon": [-118.0],
            "hp_bearing_deg": [0.0],
        }
    )
    val_df = df.copy()
    val_df["venue_id"] = 1
    val_df["temperature_f"] = 70.0
    val_df["pressure_hpa"] = 1013.0
    val_df["relative_humidity"] = 50.0
    val_df["wind_direction_deg"] = 0.0
    decoupler.calibrate(val_df, coords)
    processed = decoupler.preprocess(val_df, coords)
    final = decoupler.decouple(processed)
    assert "spin_contact_factor" in final.columns

    # Test load
    with patch(
        "algomlb.ml.batted_ball_decoupler.load_decoupler_assets",
        return_value=(decoupler.model, decoupler.coeffs),
    ):
        assert decoupler.load() is True

    # Test save
    with patch("algomlb.ml.batted_ball_decoupler.save_decoupler_assets") as mock_save:
        decoupler.save()
        mock_save.assert_called_once()
