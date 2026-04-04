"""
src/algomlb/ml/weather_features.py

Three-component environmental decomposition for the Batted Ball Flight Decoupler.
Physics formulas:
    Density: Air density ration relative to standard sea level.
    Bearing: Compass bearing from Home Plate to Center Field using HP/PM coordinates.
    Tailwind: Vector decomposition of wind speed along the flight path.
"""

from __future__ import annotations

import pandas as pd
from math import radians, sin, cos, degrees, atan2


# ------------------------------------------------------------------
# Stadium Physics: Bearing & Geometry
# ------------------------------------------------------------------


def compute_cf_bearing(
    hp_lat: float,
    hp_lon: float,
    pm_lat: float,
    pm_lon: float,
) -> float:
    """
    Compute compass bearing from home plate to center field using
    the home plate → pitcher's mound vector. PM is 60.5ft from HP on the CF axis.
    Returns degrees [0, 360).
    """
    d_lon = radians(pm_lon - hp_lon)
    hp_r = radians(hp_lat)
    pm_r = radians(pm_lat)
    x = sin(d_lon) * cos(pm_r)
    y = cos(hp_r) * sin(pm_r) - sin(hp_r) * cos(pm_r) * cos(d_lon)
    return degrees(atan2(x, y)) % 360.0


# ------------------------------------------------------------------
# Density Physics: Psychrometric Formula
# ------------------------------------------------------------------


def compute_air_density_ratio(
    temperature_f: float,
    pressure_hpa: float,
    relative_humidity: float,
) -> float:
    """
    Ratio of game air density to standard conditions
    (sea level, 59°F / 15°C, dry air, 1013.25 hPa).

    < 1.0 → thinner air (carrying farther)
    > 1.0 → denser air (carrying shorter)
    """
    temp_c = (temperature_f - 32.0) * 5.0 / 9.0
    T_K = temp_c + 273.15
    # Saturation vapor pressure (hPa)
    P_sat = 6.1078 * 10 ** (7.5 * temp_c / (237.3 + temp_c))
    P_vapor = (relative_humidity / 100.0) * P_sat
    P_dry = pressure_hpa - P_vapor
    # Standard: 1013.25 hPa, 288.15 K
    return float((P_dry / 1013.25) * (288.15 / T_K))


# ------------------------------------------------------------------
# Wind Physics: Vector Decomposition
# ------------------------------------------------------------------


def compute_tailwind_component(
    wind_speed_mph: float,
    wind_direction_deg: float,  # FROM direction
    spray_angle: float,  # deg: 0=CF, Pos=RF, Neg=LF
    cf_bearing_deg: float,  # compass bearing of stadium
) -> float:
    """
    Signed tailwind component for a hit.
    Positive = wind assisting. Negative = wind opposing.
    """
    wind_going_to = (wind_direction_deg + 180.0) % 360.0
    relative_to_cf = (wind_going_to - cf_bearing_deg) % 360.0
    if relative_to_cf > 180.0:
        relative_to_cf -= 360.0

    # Angle difference between wind vector and ball vector
    angle_diff = radians(relative_to_cf - spray_angle)
    return float(wind_speed_mph * cos(angle_diff))


# ------------------------------------------------------------------
# Pipelines
# ------------------------------------------------------------------


def add_weather_features(
    df: pd.DataFrame,
    ballpark_coords: pd.DataFrame,
) -> pd.DataFrame:
    """
    Augments a DataFrame with cf_bearing_deg, air_density_ratio, and tailwind_component.
    ballpark_coords must be indexed by venue_id with [hp_lat, hp_lon, pm_lat, pm_lon].
    """
    # Robustly handle Ballpark ID naming
    if "id" in ballpark_coords.columns:
        ballpark_coords = ballpark_coords.rename(columns={"id": "venue_id"})

    df = df.merge(ballpark_coords, on="venue_id", how="left", suffixes=("", "_bp"))

    # Vectorized Physics: Use hp_lat if present, else fallback to hp_bearing_deg
    hp_lat_col = "hp_lat" if "hp_lat" in df.columns else "hp_lat_bp"

    df["cf_bearing_deg"] = df.apply(
        lambda r: (
            compute_cf_bearing(r[hp_lat_col], r["hp_lon"], r["pm_lat"], r["pm_lon"])
            if pd.notna(r.get(hp_lat_col))
            else r.get("hp_bearing_deg", 180.0)
        ),
        axis=1,
    )

    df["air_density_ratio"] = df.apply(
        lambda r: (
            compute_air_density_ratio(
                r["temperature_f"], r["pressure_hpa"], r["relative_humidity"]
            )
            if pd.notna(r["temperature_f"])
            else 1.0
        ),
        axis=1,
    )

    df["tailwind_component"] = df.apply(
        lambda r: (
            compute_tailwind_component(
                r["wind_speed_mph"],
                r["wind_direction_deg"],
                r["spray_angle"],
                r["cf_bearing_deg"],
            )
            if pd.notna(r["wind_speed_mph"])
            else 0.0
        ),
        axis=1,
    )

    return df
