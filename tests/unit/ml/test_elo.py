import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import date
from algomlb.ml.elo import (
    EloConfig,
    _expected_home_win,
    _update_pair,
    run_elo_offline,
    backfill_team_elo_history,
)


def test_expected_home_win():
    cfg = EloConfig(base_rating=1500, hfa=30)
    # Equal ratings: with 30 pt HFA, home should be favored
    exp = _expected_home_win(1500, 1500, cfg)
    assert exp > 0.5

    # Home team much better
    exp = _expected_home_win(1700, 1500, cfg)
    assert exp > 0.7

    # Away team much better (negates HFA)
    exp = _expected_home_win(1500, 1600, cfg)
    assert exp < 0.5


def test_update_pair():
    cfg = EloConfig(k=10, hfa=30)
    # Home wins
    h_post, a_post = _update_pair(1500, 1500, 1, cfg)
    assert h_post > 1500
    assert a_post < 1500
    assert (h_post - 1500) == pytest.approx(1500 - a_post)


def test_run_elo_offline():
    cfg = EloConfig()
    games = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": date(2023, 4, 1),
                "home_team": "LAN",
                "away_team": "ARI",
                "home_score": 5,
                "away_score": 2,
            },
            {
                "game_pk": 2,
                "game_date": date(2023, 4, 2),
                "home_team": "LAN",
                "away_team": "ARI",
                "home_score": 1,
                "away_score": 3,
            },
        ]
    )

    result = run_elo_offline(games, cfg)
    assert len(result) == 4  # 2 games * 2 teams

    # Game 1: LAN wins
    lan_g1 = result[(result["game_pk"] == 1) & (result["team_id"] == "LAN")].iloc[0]
    assert lan_g1["elo_pre"] == 1500.0
    assert lan_g1["elo_post"] > 1500.0

    # Game 2: LAN loses. Pre-rating should match Game 1 post-rating.
    lan_g2 = result[(result["game_pk"] == 2) & (result["team_id"] == "LAN")].iloc[0]
    assert lan_g2["elo_pre"] == lan_g1["elo_post"]
    assert lan_g2["elo_post"] < lan_g2["elo_pre"]


def test_backfill_team_elo_history_logic():
    mock_engine = MagicMock()
    # Case 1: Empty results
    with (
        patch("pandas.read_sql", return_value=pd.DataFrame()),
        patch("algomlb.ml.elo.get_engine", return_value=mock_engine),
    ):
        backfill_team_elo_history(mock_engine)
        assert not mock_engine.begin.called

    # Case 2: Success path
    games_df = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": date(2023, 4, 1),
                "home_team": "LAN",
                "away_team": "ARI",
                "home_score": 5,
                "away_score": 2,
            }
        ]
    )
    with (
        patch("pandas.read_sql", return_value=games_df),
        patch("algomlb.ml.elo.get_engine", return_value=mock_engine),
    ):
        backfill_team_elo_history(mock_engine)
        assert mock_engine.begin.called
