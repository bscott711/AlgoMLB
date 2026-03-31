import math
import pytest
from algomlb.domain.wind_physics import wind_components, circular_std
from algomlb.domain.stadium_bearings import STADIUM_HP_BEARINGS


def test_wind_components_pure_headwind():
    # If wind_dir == hp_bearing, wind is blowing FROM CF TOWARD HP (Blowing IN).
    # Expected result: negative headwind.
    res = wind_components(10.0, 180.0, 180.0)
    assert res.headwind_mph == pytest.approx(-10.0)
    assert res.crosswind_mph == pytest.approx(0.0)


def test_wind_components_pure_tailwind():
    # If wind_dir == hp_bearing + 180, wind is blowing FROM HP TOWARD CF (Blowing OUT).
    # Expected result: positive tailwind.
    res = wind_components(10.0, 0.0, 180.0)
    assert res.headwind_mph == pytest.approx(10.0)
    assert res.crosswind_mph == pytest.approx(0.0)


def test_wind_components_pure_crosswind():
    # delta = 90 - 180 = -90. -sin(-90)=1.
    # Wind from Left (90) to batter facing South (180) is Left-to-Right.
    res = wind_components(10.0, 90.0, 180.0)
    assert res.headwind_mph == pytest.approx(0.0)
    assert res.crosswind_mph == pytest.approx(10.0)


def test_wind_components_mixed():
    # speed 10, dir 135, bearing 180. delta = -45.
    # -cos(-45) = -1/sqrt(2), -sin(-45) = 1/sqrt(2)
    res = wind_components(10.0, 135.0, 180.0)
    expected = 10.0 / math.sqrt(2)
    assert res.headwind_mph == pytest.approx(-expected)
    assert res.crosswind_mph == pytest.approx(expected)


def test_wind_components_zero_speed():
    res = wind_components(0.0, 123.4, 56.7)
    assert res.headwind_mph == 0.0
    assert res.crosswind_mph == 0.0


def test_circular_std_identical():
    assert circular_std([10.0, 10.0, 10.0]) == pytest.approx(0.0)


def test_circular_std_empty_or_single():
    assert circular_std([]) == 0.0
    assert circular_std([45.0]) == 0.0


def test_circular_std_spread():
    # Two angles 180 deg apart.
    # mean_sin = (sin(0) + sin(pi))/2 = 0
    # mean_cos = (cos(0) + cos(pi))/2 = (1 - 1)/2 = 0
    # r_bar = 0.
    # result = sqrt(-2 * log(1e-9)) * 180 / pi
    val = circular_std([0.0, 180.0])
    assert val > 100.0


def test_stadium_bearings_count():
    assert len(STADIUM_HP_BEARINGS) == 30


def test_circular_std_extreme_unstable():
    # Uniformly spread angles (0, 90, 180, 270)
    # sin sum: 0 + 1 + 0 - 1 = 0
    # cos sum: 1 + 0 - 1 + 0 = 0
    # r_bar = 0
    val = circular_std([0.0, 90.0, 180.0, 270.0])
    assert val > 100.0
