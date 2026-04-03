import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from datetime import date

from algomlb.ingestion.statcast_ingester import (
    StatcastIngester,
    RAW_COLUMNS,
)


@pytest.fixture
def mock_statcast_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": [100001, 100002],
            "at_bat_number": [1, 5],
            "pitch_number": [1, 3],
            "game_date": ["2025-04-01", "2025-04-01"],
            "home_team": ["CHC", "NYY"],
            "away_team": ["STL", "BOS"],
            "batter": [12345, 67890],
            "pitcher": [54321, 98765],
            "hc_x": [125.42, 210.0],
            "hc_y": [198.27, 150.0],
            "events": ["single", "home_run"],
            "launch_speed": [88.1, 103.4],
            "launch_angle": [12.0, 28.5],
        }
    )


@patch("algomlb.ingestion.statcast_ingester.statcast")
@patch("algomlb.ingestion.statcast_ingester.cache.enable")
def test_fetch_returns_only_known_columns(mock_sc, mock_statcast_df):
    mock_sc.return_value = mock_statcast_df
    ingester = StatcastIngester(repo=MagicMock())
    result = ingester.fetch_statcast_chunk(date(2025, 4, 1), date(2025, 4, 1))

    assert not result.empty
    assert all(c in RAW_COLUMNS for c in result.columns)
    assert "game_pk" in result.columns
    assert "hc_x" in result.columns


@patch("algomlb.ingestion.statcast_ingester.statcast")
def test_fetch_empty_returns_empty_df(mock_sc):
    mock_sc.return_value = pd.DataFrame()
    ingester = StatcastIngester(repo=MagicMock())
    result = ingester.fetch_statcast_chunk(date(2025, 4, 1), date(2025, 4, 1))
    assert result.empty


@patch("algomlb.ingestion.statcast_ingester.statcast")
def test_dry_run_writes_nothing(mock_sc, mock_statcast_df):
    mock_sc.return_value = mock_statcast_df
    mock_repo = MagicMock()
    ingester = StatcastIngester(repo=mock_repo)

    rows = ingester.ingest_range(date(2025, 4, 1), date(2025, 4, 1), dry_run=True)

    assert rows == 0
    mock_repo.save_statcast_raw.assert_not_called()


@patch("algomlb.ingestion.statcast_ingester.statcast")
def test_live_run_calls_repo_save(mock_sc, mock_statcast_df):
    mock_sc.return_value = mock_statcast_df
    mock_repo = MagicMock()
    mock_repo.save_statcast_raw.return_value = 2
    ingester = StatcastIngester(repo=mock_repo)

    rows = ingester.ingest_range(date(2025, 4, 1), date(2025, 4, 1), dry_run=False)

    assert rows == 2
    mock_repo.save_statcast_raw.assert_called_once()
    saved_rows = mock_repo.save_statcast_raw.call_args[0][0]
    assert len(saved_rows) == 2
    assert saved_rows[0]["game_pk"] == 100001
