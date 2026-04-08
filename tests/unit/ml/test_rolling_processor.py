import pytest
import pandas as pd
from datetime import date
from algomlb.ml.rolling_processor import RollingProcessor
from algomlb.domain import PlayerRole, BaselineQuality
from unittest.mock import MagicMock


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.pitcher_rolling_games = 5
    config.batter_rolling_games = 20
    config.league_mean_pitcher_xwoba = 0.320
    config.league_mean_batter_xwoba = 0.320
    config.rolling_shrinkage_k = 5.0
    return config


@pytest.fixture
def processor(mock_config):
    return RollingProcessor(mock_config)


def test_shrinkage_low_n(processor):
    # Weight = 1 / (1 + 5) = 1/6 = 0.166...
    # Result = 0.166 * 0.400 + 0.833 * 0.320 = 0.066 + 0.266 = 0.333
    res = processor.apply_shrinkage(0.400, 1, 0.320, 5.0)
    assert pytest.approx(res, rel=1e-3) == 0.333


def test_shrinkage_high_n(processor):
    # Weight = 100 / (100 + 5) = 0.952
    res = processor.apply_shrinkage(0.400, 100, 0.320, 5.0)
    assert pytest.approx(res, rel=1e-3) == 0.396


def test_compute_pitcher_full_window(processor):
    history = pd.DataFrame(
        [
            {
                "game_date": date(2024, 3, 20),
                "pitches": 100,
                "strikes": 60,
                "whiffs": 10,
                "k": 5,
                "bb": 2,
                "avg_release_speed": 95.0,
                "avg_pfx_x": -0.5,
                "avg_pfx_z": 1.0,
                "avg_pitcher_xwoba": 0.300,
                "edge_pct": 0.45,
                "fastball_velo_degradation": 0.0,
                "std_release_pos_z": 0.0,
                "avg_spin_rate": 2400.0,
                "avg_release_extension": 6.5,
                "fb_speed": 95.0,
                "xwoba_vs_rh": 0.320,
                "xwoba_vs_lh": 0.310,
            },
            {
                "game_date": date(2024, 3, 25),
                "pitches": 100,
                "strikes": 70,
                "whiffs": 15,
                "k": 8,
                "bb": 1,
                "avg_release_speed": 96.0,
                "avg_pfx_x": -0.6,
                "avg_pfx_z": 1.1,
                "avg_pitcher_xwoba": 0.250,
                "edge_pct": 0.48,
                "fastball_velo_degradation": -0.5,
                "std_release_pos_z": 0.0,
                "avg_spin_rate": 2450.0,
                "avg_release_extension": 6.6,
                "fb_speed": 96.0,
                "xwoba_vs_rh": 0.320,
                "xwoba_vs_lh": 0.310,
            },
            {
                "game_date": date(2024, 3, 30),
                "pitches": 100,
                "strikes": 65,
                "whiffs": 12,
                "k": 6,
                "bb": 3,
                "avg_release_speed": 94.0,
                "avg_pfx_x": -0.4,
                "avg_pfx_z": 0.9,
                "avg_pitcher_xwoba": 0.350,
                "edge_pct": 0.42,
                "fastball_velo_degradation": 0.5,
                "std_release_pos_z": 0.0,
                "avg_spin_rate": 2350.0,
                "avg_release_extension": 6.4,
                "fb_speed": 94.0,
                "xwoba_vs_rh": 0.320,
                "xwoba_vs_lh": 0.310,
            },
            {
                "game_date": date(2024, 4, 5),
                "pitches": 100,
                "strikes": 62,
                "whiffs": 11,
                "k": 7,
                "bb": 2,
                "avg_release_speed": 95.5,
                "avg_pfx_x": -0.5,
                "avg_pfx_z": 1.0,
                "avg_pitcher_xwoba": 0.310,
                "edge_pct": 0.44,
                "fastball_velo_degradation": 0.2,
                "std_release_pos_z": 0.0,
                "avg_spin_rate": 2400.0,
                "avg_release_extension": 6.5,
                "fb_speed": 95.5,
                "xwoba_vs_rh": 0.320,
                "xwoba_vs_lh": 0.310,
            },
            {
                "game_date": date(2024, 4, 10),
                "pitches": 100,
                "strikes": 68,
                "whiffs": 14,
                "k": 9,
                "bb": 0,
                "avg_release_speed": 96.5,
                "avg_pfx_x": -0.7,
                "avg_pfx_z": 1.2,
                "avg_pitcher_xwoba": 0.280,
                "edge_pct": 0.46,
                "fastball_velo_degradation": -0.2,
                "std_release_pos_z": 0.0,
                "avg_spin_rate": 2420.0,
                "avg_release_extension": 6.7,
                "fb_speed": 96.5,
                "xwoba_vs_rh": 0.320,
                "xwoba_vs_lh": 0.310,
            },
        ]
    )

    target_date = date(2024, 4, 15)
    season_start = date(2024, 3, 20)

    record = processor.compute_for_player(
        12345, target_date, PlayerRole.PITCHER, history, season_start
    )

    assert record.n_games_used == 5
    assert record.baseline_quality == BaselineQuality.FULL
    assert record.roll_pitches == 500.0
    assert record.roll_k_pct == (5 + 8 + 6 + 7 + 9) / 500.0
    assert record.roll_avg_release_speed == pytest.approx(95.4)
    # Shrinkage: n=5, k=5 -> weight = 0.5. (0.298 + 0.320) / 2 = 0.309
    assert record.roll_pitcher_xwoba_shrunk == pytest.approx(0.309, rel=1e-3)


def test_days_since_last_game(processor):
    history = pd.DataFrame([{"game_date": date(2024, 3, 20), "pas": 4, "hits": 1}])
    # Manually add missing columns for batter
    for col in [
        "batter_k", "batter_bb", "barrels", "avg_launch_speed", 
        "avg_launch_angle", "avg_batter_xwoba", "avg_bat_speed", 
        "avg_attack_angle", "chase_count", "in_zone_whiff_count", 
        "std_launch_angle", "xwoba_vs_rh", "xwoba_vs_lh", "edge_pct"
    ]:
        history[col] = 0.0

    target_date = date(2024, 3, 25)
    season_start = date(2024, 3, 20)
    record = processor.compute_for_player(
        6789, target_date, PlayerRole.BATTER, history, season_start
    )
    assert record.days_since_last_game == 5


def test_compute_batter_partial_window(processor):
    history = pd.DataFrame(
        [
            {
                "game_date": date(2024, 3, 20),
                "pas": 4,
                "hits": 1,
                "batter_k": 1,
                "batter_bb": 1,
                "barrels": 0,
                "avg_launch_speed": 90.0,
                "avg_launch_angle": 10.0,
                "avg_batter_xwoba": 0.300,
                "avg_bat_speed": 75.0,
                "avg_attack_angle": 12.0,
                "chase_count": 1,
                "in_zone_whiff_count": 0,
                "std_launch_angle": 5.0,
                "xwoba_vs_rh": 0.320,
                "xwoba_vs_lh": 0.310,
            },
        ]
    )

    target_date = date(2024, 3, 22)
    season_start = date(2024, 3, 20)

    record = processor.compute_for_player(
        6789, target_date, PlayerRole.BATTER, history, season_start
    )

    assert record.n_games_used == 1
    assert record.baseline_quality == BaselineQuality.PARTIAL
    assert record.roll_pas == 4.0
    assert record.roll_hits_per_pa == 0.25
    # Shrinkage: n=1, k=5 -> weight = 1/6. (1/6 * 0.3 + 5/6 * 0.32) = 0.05 + 0.266 = 0.3166
    assert record.roll_batter_xwoba_shrunk == pytest.approx(0.3166, rel=1e-3)


def test_season_boundary_exclusion(processor):
    history = pd.DataFrame(
        [
            {
                "game_date": date(2023, 10, 1),
                "pas": 4,
                "hits": 2,
                "batter_k": 0,
                "batter_bb": 0,
                "barrels": 0,
                "avg_launch_speed": 90.0,
                "avg_launch_angle": 10.0,
                "avg_batter_xwoba": 0.500,
                "avg_bat_speed": 78.0,
                "avg_attack_angle": 15.0,
                "chase_count": 0,
                "in_zone_whiff_count": 0,
                "std_launch_angle": 4.0,
                "xwoba_vs_rh": 0.320,
                "xwoba_vs_lh": 0.310,
            },  # Prior season
            {
                "game_date": date(2024, 3, 20),
                "pas": 4,
                "hits": 1,
                "batter_k": 0,
                "batter_bb": 0,
                "barrels": 0,
                "avg_launch_speed": 90.0,
                "avg_launch_angle": 10.0,
                "avg_batter_xwoba": 0.300,
                "avg_bat_speed": 75.0,
            },
        ]
    )

    target_date = date(2024, 3, 22)
    season_start = date(2024, 3, 20)

    record = processor.compute_for_player(
        6789, target_date, PlayerRole.BATTER, history, season_start
    )

    assert record.n_games_used == 1  # Only 2024 game used
    assert record.roll_pas == 4.0


def test_cold_start(processor):
    history = pd.DataFrame(columns=["game_date", "pas"])
    target_date = date(2024, 3, 20)
    season_start = date(2024, 3, 20)

    record = processor.compute_for_player(
        6789, target_date, PlayerRole.BATTER, history, season_start
    )

    assert record.n_games_used == 0
    assert record.baseline_quality == BaselineQuality.COLD_START
    assert record.roll_pas is None


def test_processor_edge_cases(processor):
    # 1. apply_shrinkage with n=0
    res = processor.apply_shrinkage(0.400, 0, 0.320, 5.0)
    assert res == 0.320

    # 2. Pitcher Partial (1 <= n < 5)
    history = pd.DataFrame(
        [
            {
                "game_date": date(2024, 3, 20),
                "pitches": 100,
                "strikes": 60,
                "whiffs": 10,
                "k": 5,
                "bb": 2,
                "avg_release_speed": 95.0,
                "avg_pfx_x": -0.5,
                "avg_pfx_z": 1.0,
                "avg_pitcher_xwoba": 0.300,
                "edge_pct": 0.45,
                "fastball_velo_degradation": 0.0,
                "std_release_pos_z": 0.0,
                "avg_spin_rate": 2400.0,
                "avg_release_extension": 6.5,
                "fb_speed": 95.0,
                "xwoba_vs_rh": 0.320,
                "xwoba_vs_lh": 0.310,
            }
        ]
    )
    record = processor.compute_for_player(
        1, date(2024, 3, 25), PlayerRole.PITCHER, history, date(2024, 3, 20)
    )
    assert record.baseline_quality == BaselineQuality.PARTIAL
    assert record.n_games_used == 1

    # 3. Total pitches = 0 (rare but for coverage)
    history["pitches"] = 0
    record = processor.compute_for_player(
        1, date(2024, 3, 25), PlayerRole.PITCHER, history, date(2024, 3, 20)
    )
    assert record.roll_pitches is None
    assert record.roll_strikes_pct is None

    # 4. Total pas = 0 (for batter coverage)
    history_b = pd.DataFrame(
        [
            {
                "game_date": date(2024, 3, 20),
                "pas": 0,
                "hits": 0,
                "batter_k": 0,
                "batter_bb": 0,
                "barrels": 0,
                "avg_launch_speed": 0.0,
                "avg_launch_angle": 0.0,
                "avg_batter_xwoba": 0.0,
                "avg_bat_speed": 0.0,
                "avg_attack_angle": 0.0,
                "chase_count": 0,
                "in_zone_whiff_count": 0,
                "std_launch_angle": 0.0,
                "xwoba_vs_rh": 0.320,
                "xwoba_vs_lh": 0.310,
                "edge_pct": 0.0,
            }
        ]
    )
    record_b = processor.compute_for_player(
        1, date(2024, 3, 25), PlayerRole.BATTER, history_b, date(2024, 3, 20)
    )
    assert record_b.roll_pas is None


def test_compute_batter_full_window(processor):
    # 20 games for better is FULL
    df = pd.DataFrame(
        [
            {
                "game_date": date(2024, 3, 20),
                "pas": 4,
                "hits": 1,
                "batter_k": 1,
                "batter_bb": 1,
                "barrels": 0,
                "avg_launch_speed": 90.0,
                "avg_launch_angle": 10.0,
                "avg_batter_xwoba": 0.300,
                "avg_bat_speed": 75.0,
                "avg_attack_angle": 12.0,
                "chase_count": 1,
                "in_zone_whiff_count": 0,
                "std_launch_angle": 5.0,
                "xwoba_vs_rh": 0.320,
                "xwoba_vs_lh": 0.310,
            }
        ]
        * 20
    )
    record = processor.compute_for_player(
        1, date(2024, 4, 15), PlayerRole.BATTER, df, date(2024, 3, 20)
    )
    assert record.n_games_used == 20
    assert record.baseline_quality == BaselineQuality.FULL
