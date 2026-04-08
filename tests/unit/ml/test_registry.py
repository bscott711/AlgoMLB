import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from algomlb.ml.registry import (
    _resolve_manager,
    _map_and_merge_games,
    _compute_tenure_metrics,
    build_manager_registry,
)


@pytest.fixture
def sample_mgrs():
    return pd.DataFrame(
        [
            {
                "team_id": 108,
                "manager_id": 1,
                "season": 2023,
                "effective_start_date": "2023-01-01",
            },
            {
                "team_id": 108,
                "manager_id": 2,
                "season": 2023,
                "effective_start_date": "2023-07-01",
            },
        ]
    )


def test_resolve_manager(sample_mgrs):
    # Before switch
    assert _resolve_manager(sample_mgrs, 108, "2023-06-01", 2023) == 1
    # After switch
    assert _resolve_manager(sample_mgrs, 108, "2023-08-01", 2023) == 2
    # No data
    assert _resolve_manager(sample_mgrs, 999, "2023-01-01", 2023) is None
    # Before any start date (Line 57 coverage)
    assert _resolve_manager(sample_mgrs, 108, "2022-12-31", 2023) == 1
    # Single manager case
    single = pd.DataFrame(
        [
            {
                "team_id": 109,
                "manager_id": 5,
                "season": 2023,
                "effective_start_date": "2023-01-01",
            }
        ]
    )
    assert _resolve_manager(single, 109, "2023-05-01", 2023) == 5


def test_map_and_merge_games():
    df_retro = pd.DataFrame([{"game_id": "ANA202304010", "date": "2023-04-01"}])
    df_results = pd.DataFrame(
        [
            {
                "game_pk": 101,
                "game_date": "2023-04-01",
                "home_team_id": 108,
                "doubleheader_num": 0,
            }
        ]
    )
    result = _map_and_merge_games(df_retro, df_results)
    assert not result.empty
    assert result.iloc[0]["game_pk"] == 101


def test_compute_tenure_metrics():
    df = pd.DataFrame(
        [
            {"team_id": 1, "manager_id": 10, "game_date": "2023-04-01"},
            {"team_id": 1, "manager_id": 10, "game_date": "2023-04-02"},
            {"team_id": 1, "manager_id": 11, "game_date": "2023-04-03"},
        ]
    )
    res = _compute_tenure_metrics(df)
    assert res.iloc[0]["manager_tenure_day"] == 1
    assert res.iloc[1]["manager_tenure_day"] == 2
    assert res.iloc[2]["manager_tenure_day"] == 1
    assert res.iloc[2]["days_since_manager_change"] == 0


def test_build_manager_registry_orchestration():
    mock_session = MagicMock()
    mock_engine = MagicMock()

    with (
        patch("algomlb.ml.registry.get_engine", return_value=mock_engine),
        patch("algomlb.ml.registry.pd.read_sql") as mock_read_sql,
    ):
        # 1. Team managers
        # 2. Year loop fetch (df_retro, df_results)
        mock_read_sql.side_effect = [
            pd.DataFrame(
                [
                    {
                        "team_id": 108,
                        "manager_id": 1,
                        "season": 2023,
                        "effective_start_date": "2023-01-01",
                    }
                ]
            ),  # df_mgrs
            pd.DataFrame(
                [{"game_id": "ANA202304010", "date": "2023-04-01"}]
            ),  # df_retro
            pd.DataFrame(
                [
                    {
                        "game_pk": 101,
                        "game_date": "2023-04-01",
                        "home_team_id": 108,
                        "away_team_id": 109,
                        "game_type": "R",
                        "doubleheader_num": 0,
                    }
                ]
            ),  # df_results
        ]

        build_manager_registry(mock_session, 2023, 2023)

        assert mock_session.bulk_save_objects.called
        assert mock_session.commit.called


def test_build_manager_registry_skips():
    mock_session = MagicMock()
    mock_engine = MagicMock()
    with (
        patch("algomlb.ml.registry.get_engine", return_value=mock_engine),
        patch("algomlb.ml.registry.pd.read_sql") as mock_read_sql,
    ):
        # Test 2020 skip and empty data returns
        mock_read_sql.side_effect = [
            pd.DataFrame(
                [
                    {
                        "team_id": 1,
                        "manager_id": 1,
                        "season": 2020,
                        "effective_start_date": "2020-01-01",
                    }
                ]
            ),  # df_mgrs
            pd.DataFrame(),  # df_retro for 2021
            pd.DataFrame(),  # df_results for 2021
        ]

        # This should skip 2020 and continue through 2021 without crashing
        build_manager_registry(mock_session, 2020, 2021)
        assert not mock_session.bulk_save_objects.called
