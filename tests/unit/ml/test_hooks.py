import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from algomlb.ml.hooks import (
    extract_hook_events,
    compute_hook_profiles,
    persist_hook_events,
    persist_hook_profiles,
    backfill_hook_events,
    _calculate_outs_recorded,
)


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


def test_extract_hook_events_empty():
    mock_engine = MagicMock()
    with patch("pandas.read_sql") as mock_read:
        mock_read.return_value = pd.DataFrame()
        result = extract_hook_events(mock_engine, 2023)
        assert result.empty


def test_calculate_outs_recorded_inning_turnover():
    # Test stint spanning an inning turnover
    stint = pd.DataFrame(
        [
            {"outs_pre": 2, "outs_post": 3},  # 1 out
            {"outs_pre": 0, "outs_post": 1},  # Inning turn + 1 out
        ]
    )
    # Total outs: (3-2) + (1-0) = 2. But turnover logic: (3-2) + (3-0?) No.
    # Lines 122-125: if outs_post >= outs_pre: outs += post-pre else: outs += 3-pre
    # Row 1: 3 >= 2 -> outs = 1
    # Row 2: 1 >= 0 -> outs = 1+1 = 2
    assert _calculate_outs_recorded(stint) == 2

    stint_turnover = pd.DataFrame(
        [
            {"outs_pre": 2, "outs_post": 0},  # ERROR case or just turnover?
        ]
    )
    # 0 < 2 -> 3 - 2 = 1 out recorded
    assert _calculate_outs_recorded(stint_turnover) == 1


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
            {
                "manager_id": None,  # Should be skipped (line 224)
                "manager_name": "Unknown",
                "season": 2023,
                "is_starter": True,
                "pitch_count": 50,
                "inning": 4,
                "score_diff": 0,
                "hook_before_3rd_tto": True,
            },
        ]
    )

    profiles = compute_hook_profiles(hooks)
    assert not profiles.empty
    assert len(profiles) == 1  # Only Dave Roberts
    prof = profiles.iloc[0]
    assert prof["manager_id"] == 123
    assert prof["total_sp_starts"] == 2

    # Coverage for line 224: nan manager_id (casted from string)
    hooks_nan = pd.DataFrame(
        [
            {
                "manager_id": "nan",
                "season": 2023,
                "is_starter": True,
                "manager_name": "N/A",
                "pitch_count": 50,
                "inning": 5,
                "score_diff": 0,
                "hook_before_3rd_tto": True,
            }
        ]
    )
    assert compute_hook_profiles(hooks_nan).empty


def test_compute_hook_profiles_empty():
    assert compute_hook_profiles(pd.DataFrame()).empty
    assert compute_hook_profiles(pd.DataFrame({"is_starter": [False]})).empty


def test_persist_hook_events():
    mock_engine = MagicMock()
    hooks = pd.DataFrame(
        [{"game_id": "G1", "pitcher_id": "P1", "manager_id": 1, "is_starter": True}]
    )
    # Test empty
    assert persist_hook_events(mock_engine, pd.DataFrame()).empty

    with patch("algomlb.ml.hooks.pg_insert"):
        res_df = persist_hook_events(mock_engine, hooks)
        assert len(res_df) == 1
        assert mock_engine.begin.called


def test_persist_hook_profiles():
    mock_engine = MagicMock()
    profiles = pd.DataFrame([{"manager_id": 1, "season": 2023, "total_hooks": 10}])
    # Test empty
    assert persist_hook_profiles(mock_engine, pd.DataFrame()) == 0

    with patch("algomlb.ml.hooks.pg_insert"):
        count = persist_hook_profiles(mock_engine, profiles)
        assert count == 1
        assert mock_engine.begin.called


def test_backfill_hook_events_pipeline():
    mock_engine = MagicMock()
    with (
        patch("algomlb.ml.hooks.extract_hook_events") as mock_ext,
        patch("algomlb.ml.hooks.persist_hook_events") as mock_p1,
        patch("algomlb.ml.hooks.compute_hook_profiles") as mock_comp,
        patch("algomlb.ml.hooks.persist_hook_profiles") as mock_p2,
    ):
        mock_ext.return_value = pd.DataFrame([{"is_starter": True}])
        mock_comp.return_value = pd.DataFrame([{"mgr": 1}])

        backfill_hook_events(mock_engine, 2023, 2023)

        assert mock_ext.called
        assert mock_p1.called
        assert mock_comp.called
        assert mock_p2.called

    # Test spanning 2020 skip (line 327)
    with patch("algomlb.ml.hooks.extract_hook_events") as mock_ext:
        mock_ext.return_value = pd.DataFrame()
        backfill_hook_events(mock_engine, 2020, 2020)
        assert not mock_ext.called

    # Test no hooks found
    with patch("algomlb.ml.hooks.extract_hook_events") as mock_ext:
        mock_ext.return_value = pd.DataFrame()
        backfill_hook_events(mock_engine, 2023, 2023)
        # Should finish silently
