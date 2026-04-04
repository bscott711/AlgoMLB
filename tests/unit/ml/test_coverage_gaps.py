"""
tests/unit/ml/test_coverage_gaps.py

Targeted tests to fill coverage gaps in ml/coefficients.py and ml/weather_features.py.
"""

import numpy as np
import pandas as pd
import pytest
from algomlb.ml.coefficients import calibrate_coefficients
from algomlb.ml.weather_features import add_weather_features, compute_tailwind_component


def test_calibrate_coefficients_full_ols():
    # Hits beta else branch (110 calm dry rows)
    df_beta = pd.DataFrame(
        {
            "total_delta": np.random.normal(0, 5, 110),
            "air_density_ratio": np.random.uniform(0.9, 1.1, 110),
            "tailwind_component": np.random.normal(0, 10, 110),
            "wind_speed_mph": np.zeros(110),
            "precipitation_mm_hr": np.zeros(110),
        }
    )
    coeffs_beta = calibrate_coefficients(df_beta)
    assert isinstance(coeffs_beta["beta"], float)

    # Hits delta else branch (110 rainy calm rows)
    df_delta = pd.DataFrame(
        {
            "total_delta": np.random.normal(0, 5, 110),
            "air_density_ratio": np.random.uniform(0.9, 1.1, 110),
            "tailwind_component": np.random.normal(0, 10, 110),
            "wind_speed_mph": np.zeros(110),
            "precipitation_mm_hr": np.ones(110) * 0.5,
        }
    )
    coeffs_delta = calibrate_coefficients(df_delta)
    assert isinstance(coeffs_delta["delta"], float)


def test_calibrate_coefficients_missing_cols():
    df = pd.DataFrame({"total_delta": [1.0]})
    with pytest.raises(ValueError, match="Missing required column"):
        calibrate_coefficients(df)


def test_add_weather_features_extra_coverage():
    # 1. Test "id" rename in ballpark_coords
    df = pd.DataFrame(
        {
            "venue_id": [1],
            "temperature_f": [70],
            "pressure_hpa": [1013],
            "relative_humidity": [50],
            "wind_speed_mph": [5],
            "wind_direction_deg": [0],
            "spray_angle": [0],
        }
    )
    ballparks = pd.DataFrame(
        {
            "id": [1],
            "hp_lat": [34],
            "hp_lon": [-118],
            "pm_lat": [34.1],
            "pm_lon": [-118],
        }
    )

    res = add_weather_features(df, ballparks)
    assert "cf_bearing_deg" in res.columns

    # 2. Test hp_lat in main df
    df["hp_lat"] = 34.0
    df["hp_lon"] = -118.0
    df["pm_lat"] = 34.1
    df["pm_lon"] = -118.0
    res2 = add_weather_features(df, ballparks)
    assert "cf_bearing_deg" in res2.columns

    # 3. Test cf_bearing_deg fallback to 180 (no hp_lat, no hp_bearing_deg)
    df3 = pd.DataFrame(
        {
            "venue_id": [2],
            "temperature_f": [None],
            "pressure_hpa": [None],
            "relative_humidity": [None],
            "wind_speed_mph": [None],
            "wind_direction_deg": [None],
            "spray_angle": [0.0],
        }
    )
    ballparks3 = pd.DataFrame({"venue_id": [2], "hp_lat": [None]})
    res3 = add_weather_features(df3, ballparks3)
    assert res3["cf_bearing_deg"].iloc[0] == 180.0

    # 4. Test cf_bearing_deg fallback to hp_bearing_deg
    df4 = pd.DataFrame(
        {
            "venue_id": [3],
            "temperature_f": [70],
            "pressure_hpa": [1013],
            "relative_humidity": [50],
            "wind_speed_mph": [5],
            "wind_direction_deg": [0],
            "spray_angle": [0],
            "hp_bearing_deg": [90.0],
        }
    )
    ballparks4 = pd.DataFrame({"venue_id": [3], "hp_lat": [None]})
    res4 = add_weather_features(df4, ballparks4)
    assert res4["cf_bearing_deg"].iloc[0] == 90.0


def test_wind_vector_normalization_coverage():
    # Hits the relative_to_cf > 180.0 branch
    # wind_going_to = (0 + 180) % 360 = 180
    # relative_to_cf = (180 - 350) % 360 = 190. 190 > 180 -> 190 - 360 = -170
    tw = compute_tailwind_component(10.0, 0.0, 0.0, 350.0)
    assert isinstance(tw, float)
