import pandas as pd
import pytest
from datetime import date

from algomlb.ml.quant_service import (
    assign_launch_quality,
    build_quant_features,
    calibrate_probability,
    compute_spray_angle,
    standardize_pitch_movement,
    statcast_to_cartesian,
)


class TestStatcastToCartesian:
    def test_home_plate_origin(self):
        x, y = statcast_to_cartesian(pd.Series([125.42]), pd.Series([198.27]))
        assert abs(x.iloc[0]) < 0.01
        assert abs(y.iloc[0]) < 0.01

    def test_right_field_line(self):
        # x increases to the right
        x, y = statcast_to_cartesian(pd.Series([250.84]), pd.Series([198.27]))
        assert abs(x.iloc[0] - 125.42) < 0.01

    def test_y_increases_toward_outfield(self):
        _, y_deep = statcast_to_cartesian(pd.Series([125.42]), pd.Series([50.0]))
        _, y_shallow = statcast_to_cartesian(pd.Series([125.42]), pd.Series([150.0]))
        assert y_deep.iloc[0] > y_shallow.iloc[0]


class TestComputeSprayAngle:
    def test_straight_up_middle(self):
        # x=0, y=100 -> 0°
        angle = compute_spray_angle(pd.Series([0.0]), pd.Series([100.0]))
        assert angle.iloc[0] == 0.0

    def test_pull_side_rhb(self):
        # x < 0
        angle = compute_spray_angle(pd.Series([-50.0]), pd.Series([50.0]))
        assert angle.iloc[0] == -45.0

    def test_oppo_side_rhb(self):
        # x > 0
        angle = compute_spray_angle(pd.Series([50.0]), pd.Series([50.0]))
        assert angle.iloc[0] == 45.0

    def test_zero_y_returns_zero(self):
        # Coverage for line 35 (y=0 or NaN)
        angle = compute_spray_angle(pd.Series([50.0]), pd.Series([0.0]))
        assert angle.iloc[0] == 0.0

        angle_nan = compute_spray_angle(pd.Series([float("nan")]), pd.Series([50.0]))
        assert angle_nan.iloc[0] == 0.0


class TestAssignLaunchQuality:
    @pytest.mark.parametrize(
        "ev,la,expected",
        [
            (70.0, 15.0, 0),  # Weak
            (85.0, -15.0, 1),  # Topped
            (85.0, 55.0, 2),  # Under
            (85.0, 20.0, 3),  # Solid
            (100.0, 28.0, 4),  # Barrel
            (float("nan"), 15.0, -1),
        ],
    )
    def test_classify(self, ev, la, expected):
        result = assign_launch_quality(pd.Series([ev]), pd.Series([la]))
        assert result.iloc[0] == expected


class TestCalibrateProb:
    def test_clips_to_unit_interval(self):
        result = calibrate_probability(pd.Series([2.0, -1.0]), 0.3, 0.05)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_zero_std_passthrough(self):
        s = pd.Series([0.3, 0.5, 0.7])
        result = calibrate_probability(s, 0.4, 0.0)
        pd.testing.assert_series_equal(result, s.clip(0.0, 1.0))

    def test_nan_passthrough(self):
        s = pd.Series([0.3])
        result = calibrate_probability(s, float("nan"), 0.05)
        pd.testing.assert_series_equal(result, s.clip(0.0, 1.0))


class TestStandardizePitchMovement:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pitcher": [101] * 10 + [102],
                "pitch_type": ["FF"] * 10 + ["SL"],
                "pfx_x": [1.0, 1.1, 0.9, 1.0, 1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 0.5],
            }
        )

    def test_known_pitcher_pitch_type_has_zscore(self):
        baseline = self._make_df()
        raw = pd.DataFrame({"pitcher": [101], "pitch_type": ["FF"], "pfx_x": [1.0]})
        result = standardize_pitch_movement(raw, baseline, "pfx_x")
        assert not pd.isna(result.iloc[0])

    def test_insufficient_baseline_returns_nan(self):
        baseline = pd.DataFrame(
            {
                "pitcher": [101] * 5,
                "pitch_type": ["FF"] * 5,
                "pfx_x": [1.0] * 5,
            }
        )
        raw = pd.DataFrame({"pitcher": [101], "pitch_type": ["FF"], "pfx_x": [1.0]})
        result = standardize_pitch_movement(raw, baseline, "pfx_x")
        assert pd.isna(result.iloc[0])

    def test_empty_baseline_returns_nan(self):
        raw = pd.DataFrame({"pitcher": [101], "pitch_type": ["FF"], "pfx_x": [1.0]})
        result = standardize_pitch_movement(raw, pd.DataFrame(), "pfx_x")
        assert pd.isna(result.iloc[0])


class TestBuildQuantFeatures:
    def _raw(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_pk": [700001],
                "at_bat_number": [1],
                "pitch_number": [1],
                "game_date": [date(2025, 4, 5)],
                "batter": [500001],
                "pitcher": [101],
                "hc_x": [180.0],
                "hc_y": [150.0],
                "launch_speed": [95.0],
                "launch_angle": [28.0],
                "pitch_type": ["FF"],
                "pfx_x": [1.0],
                "pfx_z": [0.5],
                "release_speed": [94.0],
                "estimated_ba_using_speedangle": [0.45],
                "estimated_woba_using_speedangle": [0.52],
            }
        )

    def _baseline(self) -> pd.DataFrame:
        rows = 20
        return pd.DataFrame(
            {
                "game_date": [date(2025, 4, 4)] * rows,
                "pitcher": [101] * rows,
                "pitch_type": ["FF"] * rows,
                "pfx_x": [1.0] * rows,
                "pfx_z": [0.5] * rows,
                "release_speed": [94.0] * rows,
                "estimated_ba_using_speedangle": [0.3] * rows,
                "estimated_woba_using_speedangle": [0.35] * rows,
            }
        )

    def test_output_has_all_expected_columns(self):
        result = build_quant_features(self._raw(), self._baseline(), date(2025, 4, 5))
        required = {
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "game_date",
            "batter",
            "pitcher",
            "hit_x_ft",
            "hit_y_ft",
            "spray_angle_deg",
            "launch_quality",
            "pfx_x_std",
            "pfx_z_std",
            "release_speed_std",
            "xba_raw",
            "xba_calibrated",
            "xwoba_raw",
            "xwoba_calibrated",
            "baseline_window_days",
        }
        assert required.issubset(set(result.columns))

    def test_lookahead_guard_raises_on_input_mismatch(self):
        raw = self._raw()
        raw["game_date"] = date(2025, 4, 4)
        with pytest.raises(
            AssertionError, match="Input raw contains dates before as_of"
        ):
            build_quant_features(raw, self._baseline(), date(2025, 4, 5))

    def test_lookahead_guard_raises_on_baseline(self):
        baseline_with_future = self._baseline().copy()
        baseline_with_future.loc[0, "game_date"] = date(2025, 4, 5)  # same as as_of
        with pytest.raises(AssertionError, match="Lookahead violation"):
            build_quant_features(self._raw(), baseline_with_future, date(2025, 4, 5))

    def test_empty_raw_returns_empty(self):
        result = build_quant_features(
            pd.DataFrame(), self._baseline(), date(2025, 4, 5)
        )
        assert result.empty

    def test_empty_baseline_logic(self):
        # Should still run hit_x_ft and spray_angle but z-scores will be NaN
        result = build_quant_features(self._raw(), pd.DataFrame(), date(2025, 4, 5))
        assert not result.empty
        assert pd.isna(result["pfx_x_std"].iloc[0])
        assert result["xba_calibrated"].iloc[0] == 0.45
