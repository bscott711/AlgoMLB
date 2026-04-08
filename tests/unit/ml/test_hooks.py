import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from algomlb.ml.hooks import extract_hook_events, compute_hook_profiles


@pytest.fixture
def mock_retrosheet_df():
    """Two pitchers in one game, one hook."""
    return pd.DataFrame(
        [
            {
                "game_id": "LAN202304010",
                "date": "2023-04-01",
                "play_number": 1,
                "inning": 1,
                "top_bot": 0,
                "pitcher_id": "SP1",
                "pit_team": "LAN",
                "bat_team": "ARI",
                "pa_flag": 1,
                "nump": 15,
                "lp": 1,
                "outs_pre": 0,
                "outs_post": 1,
                "score_v": 0,
                "score_h": 0,
                "r1": 0,
                "r2": 0,
                "r3": 0,
                "single": 0,
                "double_flag": 0,
                "triple": 0,
                "hr": 0,
                "walk": 0,
                "hbp": 0,
                "k": 1,
                "runs": 0,
            },
            {
                "game_id": "LAN202304010",
                "date": "2023-04-01",
                "play_number": 2,
                "inning": 1,
                "top_bot": 0,
                "pitcher_id": "SP1",
                "pit_team": "LAN",
                "bat_team": "ARI",
                "pa_flag": 1,
                "nump": 20,
                "lp": 2,
                "outs_pre": 1,
                "outs_post": 2,
                "score_v": 0,
                "score_h": 0,
                "r1": 0,
                "r2": 0,
                "r3": 0,
                "single": 1,
                "double_flag": 0,
                "triple": 0,
                "hr": 0,
                "walk": 0,
                "hbp": 0,
                "k": 0,
                "runs": 0,
            },
            {
                "game_id": "LAN202304010",
                "date": "2023-04-01",
                "play_number": 3,
                "inning": 1,
                "top_bot": 0,
                "pitcher_id": "RP1",
                "pit_team": "LAN",
                "bat_team": "ARI",
                "pa_flag": 1,
                "nump": 10,
                "lp": 3,
                "outs_pre": 2,
                "outs_post": 3,
                "score_v": 0,
                "score_h": 0,
                "r1": 0,
                "r2": 0,
                "r3": 0,
                "single": 0,
                "double_flag": 0,
                "triple": 0,
                "hr": 0,
                "walk": 0,
                "hbp": 0,
                "k": 1,
                "runs": 0,
            },
        ]
    )


@pytest.fixture
def mock_managers_df():
    return pd.DataFrame(
        [{"team_abbr": "LAN", "manager_id": 123, "manager_name": "Dave Roberts"}]
    )


def test_extract_hook_events(mock_retrosheet_df, mock_managers_df):
    mock_engine = MagicMock()

    with patch("pandas.read_sql") as mock_read:
        # Return retrosheet PAs first, then manager lookup
        mock_read.side_effects = [mock_retrosheet_df, mock_managers_df]
        # Alternative to side_effect for multiple calls
        mock_read.side_effect = lambda sql, engine: (
            mock_retrosheet_df if "retrosheet_events" in sql else mock_managers_df
        )

        result = extract_hook_events(mock_engine, 2023)

        assert not result.empty
        assert len(result) == 2  # SP1 hooked, then RP1 (end of game)

        # Check SP1 record
        sp1 = result[result["pitcher_id"] == "SP1"].iloc[0]
        assert sp1["is_starter"]
        assert sp1["pitch_count"] == 35
        assert sp1["pa_count"] == 2
        assert sp1["manager_name"] == "Dave Roberts"
        assert sp1["hits_allowed"] == 1


def test_compute_hook_profiles():
    hooks = pd.DataFrame(
        [
            {
                "manager_id": 123,
                "manager_name": "Dave Roberts",
                "season": 2023,
                "is_starter": True,
                "pitch_count": 90,
                "inning": 6,
                "score_diff": 2,
                "hook_before_3rd_tto": False,
            },
            {
                "manager_id": 123,
                "manager_name": "Dave Roberts",
                "season": 2023,
                "is_starter": True,
                "pitch_count": 105,
                "inning": 7,
                "score_diff": -1,
                "hook_before_3rd_tto": False,
            },
        ]
    )

    profiles = compute_hook_profiles(hooks)
    assert not profiles.empty
    assert len(profiles) == 1
    prof = profiles.iloc[0]
    assert prof["manager_id"] == 123
    assert prof["total_sp_starts"] == 2
    assert prof["avg_sp_pitch_count"] == 97.5
    assert prof["long_hook_over_100_pitches_pct"] == 0.5
