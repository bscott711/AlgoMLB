import pandas as pd
from algomlb.ml.sabermetrics import (
    compute_pythagorean_features,
    _encode_base_out_state,
    build_run_expectancy_matrix,
    compute_re24_per_pa,
    compute_rolling_re24,
)


def test_compute_pythagorean_features():
    # 6 games to exceed the < 5 history limit
    games = pd.DataFrame(
        [
            {
                "game_pk": i,
                "game_date": f"2023-04-{i:02d}",
                "home_team": "ANA",
                "away_team": "HOU",
                "home_score": 10,
                "away_score": 0,
            }
            for i in range(1, 7)
        ]
    )
    result = compute_pythagorean_features(games, window=5)

    # First 5 games should have 0.5 (insufficient history)
    # The 6th game should have history from 1-5
    ana_6 = result[(result["team_id"] == "ANA") & (result["game_pk"] == 6)].iloc[0]
    assert ana_6["pythag_win_pct"] > 0.99  # Dominant wins
    assert ana_6["roll_run_diff"] == 10.0

    hou_6 = result[(result["team_id"] == "HOU") & (result["game_pk"] == 6)].iloc[0]
    assert hou_6["pythag_win_pct"] < 0.01


def test_encode_base_out_state():
    assert _encode_base_out_state(0, None, None, None) == "0_000"
    assert _encode_base_out_state(1, "123", None, None) == "1_100"
    assert _encode_base_out_state(2, "123", "456", "789") == "2_111"
    # Handling whitespace/strings
    assert _encode_base_out_state(2, " ", "", "  ") == "2_000"


def test_run_expectancy_matrix_and_pa():
    # Create 2 innings of data
    events = pd.DataFrame(
        [
            # Half Inning 1: 3 solo HRs, then 3 outs
            {
                "game_id": "G1",
                "inning": 1,
                "top_bot": 0,
                "outs_pre": 0,
                "br1_pre": None,
                "br2_pre": None,
                "br3_pre": None,
                "outs_post": 0,
                "runs": 1,
                "pa_flag": 1,
                "batter_id": 1,
                "pitcher_id": 10,
                "bat_team": "ANA",
                "pit_team": "HOU",
                "date": "2023-01-01",
            },
            {
                "game_id": "G1",
                "inning": 1,
                "top_bot": 0,
                "outs_pre": 0,
                "br1_pre": None,
                "br2_pre": None,
                "br3_pre": None,
                "outs_post": 0,
                "runs": 1,
                "pa_flag": 1,
                "batter_id": 2,
                "pitcher_id": 10,
                "bat_team": "ANA",
                "pit_team": "HOU",
                "date": "2023-01-01",
            },
            {
                "game_id": "G1",
                "inning": 1,
                "top_bot": 0,
                "outs_pre": 0,
                "br1_pre": None,
                "br2_pre": None,
                "br3_pre": None,
                "outs_post": 3,
                "runs": 1,
                "pa_flag": 1,
                "batter_id": 3,
                "pitcher_id": 10,
                "bat_team": "ANA",
                "pit_team": "HOU",
                "date": "2023-01-01",
            },
        ]
    )

    re_matrix = build_run_expectancy_matrix(events)
    # 0_000 should have some value
    assert re_matrix["0_000"] > 0
    # All 24 states present
    assert len(re_matrix) == 24

    re24_pa = compute_re24_per_pa(events, re_matrix)
    assert not re24_pa.empty

    # Coverage for line 219: re_matrix=None
    re24_none = compute_re24_per_pa(events, None)
    assert not re24_none.empty
    assert "re24_batter" in re24_pa.columns
    assert "re24_pitcher" in re24_pa.columns


def test_compute_rolling_re24():
    # Need 4 games to hit min_periods=3 with shift(1)
    re24_data = pd.DataFrame(
        [
            {
                "batter_id": 1,
                "pitcher_id": 10,
                "game_id": f"G{i}",
                "date": f"2023-04-{i:02d}",
                "re24_batter": 1.0,
                "re24_pitcher": -1.0,
            }
            for i in range(1, 6)
        ]
    )

    rolling = compute_rolling_re24(re24_data, window=10)
    # Should have rows for G4 and G5 (since G1-G3 are required for min_periods=3)
    # Wait, shift(1) means G4 uses [G1, G2, G3]. count=3.
    # G1, G2, G3 will have NaN.
    # G4 should be the first valid
    res_b = rolling[rolling["role"] == "BATTER"]
    assert len(res_b) == 2  # April 4 and April 5
    assert res_b.iloc[0]["roll_re24"] == 1.0  # Average of [1.0]
