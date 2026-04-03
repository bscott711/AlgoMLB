from __future__ import annotations

import math
from datetime import date

import pandas as pd
from algomlb.config.settings import get_settings

SETTINGS = get_settings()
DEFAULT_BASELINE_WINDOW: int = SETTINGS.ml.quant_baseline_window


# ─── Coordinate transform ─────────────────────────────────────────────────────


def statcast_to_cartesian(
    hc_x: pd.Series, hc_y: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Convert Statcast hc_x/hc_y to feet. Home plate at (0, 0).
    The Statcast coordinate system has (125.42, 198.27) as home plate.
    """
    return (hc_x - 125.42), (198.27 - hc_y)


def compute_spray_angle(hit_x_ft: pd.Series, hit_y_ft: pd.Series) -> pd.Series:
    """
    Spray angle in degrees.
    0° = straight up the middle, negative = pull side (right-handed), positive = oppo.
    """

    def _angle(x: float, y: float) -> float:
        if pd.isna(x) or pd.isna(y) or y == 0:
            return 0.0
        return math.degrees(math.atan2(x, y))

    return pd.Series(
        [_angle(x, y) for x, y in zip(hit_x_ft, hit_y_ft)],
        index=hit_x_ft.index,
    )


# ─── Launch quality bucket ────────────────────────────────────────────────────


def assign_launch_quality(
    launch_speed: pd.Series, launch_angle: pd.Series
) -> pd.Series:
    """
    Classify each batted ball into a quality bucket (0–4).
    Based on Statcast launch_speed_angle categories.
      0 = Weak    (EV < 80)
      1 = Topped  (EV ≥ 80, LA ≤ -10)
      2 = Under   (EV ≥ 80, LA > 50)
      3 = Solid   (EV ≥ 80, -10 < LA ≤ 50, EV < 98)
      4 = Barrel  (EV ≥ 98, 26 ≤ LA ≤ 30 or within barrel zone)
    """

    def _classify(ev: float, la: float) -> int:
        if pd.isna(ev) or pd.isna(la):
            return -1
        if ev < 80:
            return 0
        if la <= -10:
            return 1
        if la > 50:
            return 2
        if ev >= 98 and 26 <= la <= 30:
            return 4
        return 3

    return pd.Series(
        [_classify(ev, la) for ev, la in zip(launch_speed, launch_angle)],
        index=launch_speed.index,
        dtype="Int8",
    )


# ─── Pitch movement standardization ──────────────────────────────────────────


def standardize_pitch_movement(
    raw: pd.DataFrame,
    baseline: pd.DataFrame,
    col: str,
) -> pd.Series:
    """
    Z-score `col` relative to the trailing baseline grouped by (pitcher, pitch_type).
    Lookahead guard: baseline must contain only rows with game_date < current game_date.
    Returns NaN for pitchers/pitch types with < 10 baseline pitches.
    """
    if baseline.empty:
        return pd.Series([float("nan")] * len(raw), index=raw.index)

    stats = (
        baseline.groupby(["pitcher", "pitch_type"])[col]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    merged = raw.merge(stats, on=["pitcher", "pitch_type"], how="left")
    valid = (merged["count"] >= 10) & (merged["std"] > 0)

    z_scores = (raw[col] - merged["mean"]) / merged["std"]
    return z_scores.where(valid, float("nan"))


# ─── xBA / xwOBA calibration ─────────────────────────────────────────────────


def calibrate_probability(
    raw_prob: pd.Series,
    season_mean: float,
    season_std: float,
) -> pd.Series:
    """
    Rescale raw Statcast probability to season baseline via z-score inversion.
    Clips output to [0.0, 1.0].
    Season mean/std must be computed from games strictly before the current batch.
    """
    if pd.isna(season_mean) or pd.isna(season_std) or season_std == 0:
        return raw_prob.clip(0.0, 1.0)

    z = (raw_prob - season_mean) / season_std
    return (z * season_std + season_mean).clip(0.0, 1.0)


# ─── Main transform ───────────────────────────────────────────────────────────


def build_quant_features(
    raw: pd.DataFrame,
    baseline: pd.DataFrame,
    as_of: date,
    baseline_window_days: int = DEFAULT_BASELINE_WINDOW,
) -> pd.DataFrame:
    """
    Produce one quant feature row per pitch/event in `raw`.

    Parameters
    ----------
    raw:
        Rows from statcast_raw for the target date or game_pk batch.
    baseline:
        Rows from statcast_raw for the window strictly before as_of.
    as_of:
        Temporal cutoff. No data from this date or later enters calibration.
    """
    if raw.empty:
        return pd.DataFrame()

    if not raw.empty:
        assert (raw["game_date"] == as_of).all() or (raw["game_date"] >= as_of).any(), (
            "Input raw contains dates before as_of"
        )

    if not baseline.empty:
        assert (baseline["game_date"] < as_of).all(), (
            "Lookahead violation: baseline contains rows on or after as_of"
        )

    out = raw[
        ["game_pk", "at_bat_number", "pitch_number", "game_date", "batter", "pitcher"]
    ].copy()

    # Hit coordinates
    out["hit_x_ft"], out["hit_y_ft"] = statcast_to_cartesian(raw["hc_x"], raw["hc_y"])
    out["spray_angle_deg"] = compute_spray_angle(out["hit_x_ft"], out["hit_y_ft"])

    # Launch quality
    out["launch_quality"] = assign_launch_quality(
        raw["launch_speed"], raw["launch_angle"]
    )

    # Pitch movement z-scores
    out["pfx_x_std"] = standardize_pitch_movement(raw, baseline, "pfx_x")
    out["pfx_z_std"] = standardize_pitch_movement(raw, baseline, "pfx_z")
    out["release_speed_std"] = standardize_pitch_movement(
        raw, baseline, "release_speed"
    )

    # xBA calibration
    if not baseline.empty:
        season_xba_mean = float(baseline["estimated_ba_using_speedangle"].mean())
        season_xba_std = float(baseline["estimated_ba_using_speedangle"].std())
        out["xba_raw"] = raw["estimated_ba_using_speedangle"]
        out["xba_calibrated"] = calibrate_probability(
            raw["estimated_ba_using_speedangle"], season_xba_mean, season_xba_std
        )

        # xwOBA calibration
        season_xwoba_mean = float(baseline["estimated_woba_using_speedangle"].mean())
        season_xwoba_std = float(baseline["estimated_woba_using_speedangle"].std())
        out["xwoba_raw"] = raw["estimated_woba_using_speedangle"]
        out["xwoba_calibrated"] = calibrate_probability(
            raw["estimated_woba_using_speedangle"], season_xwoba_mean, season_xwoba_std
        )
    else:
        out["xba_raw"] = raw["estimated_ba_using_speedangle"]
        out["xba_calibrated"] = raw["estimated_ba_using_speedangle"]
        out["xwoba_raw"] = raw["estimated_woba_using_speedangle"]
        out["xwoba_calibrated"] = raw["estimated_woba_using_speedangle"]

    out["baseline_window_days"] = baseline_window_days
    return out
