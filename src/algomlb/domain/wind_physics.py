import math
from typing import NamedTuple


class WindComponents(NamedTuple):
    headwind_mph: float  # >0 blowing OUT (tailwind), <0 blowing IN
    crosswind_mph: float  # >0 left-to-right from batter's perspective


def wind_components(
    wind_speed_mph: float,
    wind_dir_deg: float,
    hp_bearing_deg: float,
) -> WindComponents:
    """
    Decompose wind into components relative to the home plate axis.

    Args:
        wind_speed_mph: Observed wind speed in mph
        wind_dir_deg:   Meteorological wind direction (degrees FROM which wind blows)
        hp_bearing_deg: Compass bearing from home plate toward center field

    Returns:
        WindComponents with headwind and crosswind magnitudes
    """
    delta = math.radians(wind_dir_deg - hp_bearing_deg)
    return WindComponents(
        # If wind_dir == hp_bearing, wind is FROM CF TOWARD HP (Blowing IN).
        # cos(0) = 1. We want negative for IN. So negate.
        headwind_mph=-wind_speed_mph * math.cos(delta),
        # If wind_dir == hp_bearing + 90, wind is FROM batter's RIGHT.
        # sin(90) = 1. We want negative for Right-to-Left. So negate.
        crosswind_mph=-wind_speed_mph * math.sin(delta),
    )


def circular_std(angles_deg: list[float]) -> float:
    """
    Circular standard deviation for a list of angles (degrees).
    Used to measure wind direction instability over a game.
    Returns 0 (stable) to 180 (completely unstable).
    """
    if len(angles_deg) < 2:
        return 0.0
    radians = [math.radians(a) for a in angles_deg]
    mean_sin = sum(math.sin(r) for r in radians) / len(radians)
    mean_cos = sum(math.cos(r) for r in radians) / len(radians)
    r_bar = math.sqrt(mean_sin**2 + mean_cos**2)
    if r_bar > 0.9999999999:  # Handle floating point precision for identical angles
        r_bar = 1.0
    return math.degrees(math.sqrt(-2.0 * math.log(max(r_bar, 1e-9))))
