import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from datetime import date

from algomlb.ingestion.statcast_ingester import (
    StatcastIngester,
    RAW_COLUMNS,
)


@pytest.fixture
def statcast_data_df() -> pd.DataFrame:
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
def test_fetch_returns_only_known_columns(_mock_enable, mock_sc, statcast_data_df):
    mock_sc.return_value = statcast_data_df
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
def test_dry_run_writes_nothing(mock_sc, statcast_data_df):
    mock_sc.return_value = statcast_data_df
    mock_repo = MagicMock()
    ingester = StatcastIngester(repo=mock_repo)

    rows = ingester.ingest_range(date(2025, 4, 1), date(2025, 4, 1), dry_run=True)

    assert rows == 0
    mock_repo.save_statcast_raw.assert_not_called()


@patch("algomlb.ingestion.statcast_ingester.statcast")
def test_live_run_calls_repo_save(mock_sc, statcast_data_df):
    mock_sc.return_value = statcast_data_df
    mock_repo = MagicMock()
    mock_repo.save_statcast_raw.return_value = 2
    ingester = StatcastIngester(repo=mock_repo)

    rows = ingester.ingest_range(date(2025, 4, 1), date(2025, 4, 1), dry_run=False)

    assert rows == 2
    saved_rows = mock_repo.save_statcast_raw.call_args[0][0]
    assert len(saved_rows) == 2
    assert saved_rows[0]["game_pk"] == 100001


@patch("algomlb.db.session.get_session_factory")
def test_constructor_no_repo(mock_session_factory):
    """Test constructor when no repo is provided."""
    mock_session = MagicMock()
    # Mock the return of get_session_factory() to be another mock (the factory)
    # and that mock when called matches our mock_session
    mock_factory = MagicMock(return_value=mock_session)
    mock_session_factory.return_value = mock_factory

    ingester = StatcastIngester()
    assert ingester.repo is not None
    assert ingester.repo.session == mock_session


@patch("algomlb.ingestion.statcast_ingester.statcast")
def test_fetch_statcast_empty_after_filtering(mock_sc):
    """Test fetch_statcast_chunk when all rows are filtered out."""
    # CASE: rows exist but none match regular season/postseason
    df = pd.DataFrame(
        {
            "game_pk": [1],
            "game_type": ["S"],  # Spring Training
            "at_bat_number": [1],
            "pitch_number": [1],
        }
    )
    mock_sc.return_value = df
    ingester = StatcastIngester(repo=MagicMock())
    result = ingester.fetch_statcast_chunk(date(2025, 4, 1), date(2025, 4, 1))
    assert result.empty

    # CASE: statcast returns None (Line 100)
    mock_sc.return_value = None
    result = ingester.fetch_statcast_chunk(date(2025, 4, 1), date(2025, 4, 1))
    assert result.empty


@patch("algomlb.ingestion.statcast_ingester.statcast")
def test_fetch_statcast_filtering(mock_sc):
    """Test that only R, F, D, L, W games are kept."""
    df = pd.DataFrame(
        {
            "game_pk": [1, 2],
            "game_type": ["R", "S"],  # Regular vs Spring Training
            "at_bat_number": [1, 1],
            "pitch_number": [1, 1],
        }
    )
    mock_sc.return_value = df
    ingester = StatcastIngester(repo=MagicMock())
    result = ingester.fetch_statcast_chunk(date(2025, 4, 1), date(2025, 4, 1))
    assert len(result) == 1
    assert result.iloc[0]["game_type"] == "R"


@patch("algomlb.ingestion.statcast_ingester.statcast")
def test_fetch_statcast_error_handling(mock_sc):
    """Test exception handling in fetch_statcast_chunk."""
    mock_sc.side_effect = Exception("API Down")
    ingester = StatcastIngester(repo=MagicMock())
    result = ingester.fetch_statcast_chunk(date(2025, 4, 1), date(2025, 4, 1))
    assert result.empty


def test_process_rows_nan_handling():
    """Test that NaNs are converted to None."""
    df = pd.DataFrame({"a": [1.0, float("nan")], "b": ["text", None]})
    ingester = StatcastIngester(repo=MagicMock())
    rows = ingester._process_rows(df)
    assert rows[1]["a"] is None
    assert rows[0]["a"] == 1.0


@patch("algomlb.ingestion.statcast_ingester.statcast")
def test_ingest_range_no_data(mock_sc):
    """Test ingest_range when no data is returned."""
    mock_sc.return_value = pd.DataFrame()
    mock_repo = MagicMock()
    ingester = StatcastIngester(repo=mock_repo)
    total = ingester.ingest_range(date(2025, 4, 1), date(2025, 4, 1))
    assert total == 0
    mock_repo.save_statcast_raw.assert_not_called()
