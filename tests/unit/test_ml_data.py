import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from algomlb.cli.main import app
from algomlb.ingestion.historical import HistoricalDataLoader

runner = CliRunner()


@pytest.fixture
def dummy_pitching_df():
    return pd.DataFrame({"Name": ["Gerrit Cole"], "ERA": [2.63]})


@pytest.fixture
def dummy_batting_df():
    return pd.DataFrame({"Team": ["NYY"], "AVG": [0.227]})


def test_data_loader_caching(tmp_path, dummy_pitching_df, dummy_batting_df):
    """Verify that HistoricalDataLoader correctly uses Parquet cache."""
    mock_repo = MagicMock()
    loader = HistoricalDataLoader(repo=mock_repo, cache_dir=tmp_path)

    # Mock pybaseball stats calls
    with (
        patch("pybaseball.pitching_stats") as mock_pitching,
        patch("pybaseball.team_batting") as mock_batting,
    ):
        mock_pitching.return_value = dummy_pitching_df
        mock_batting.return_value = dummy_batting_df

        # 1. Fetch first time - should call pybaseball and write cache
        pitching_res = loader.fetch_pitching_stats(2023, 2023)
        batting_res = loader.fetch_team_batting(2023, 2023)

        assert mock_pitching.call_count == 1
        assert mock_batting.call_count == 1
        assert "name" in pitching_res.columns  # Column name cleaned
        assert "team" in batting_res.columns

        # 2. Fetch second time - should use cache, no pybaseball call
        loader_cached = HistoricalDataLoader(repo=mock_repo, cache_dir=tmp_path)
        pitching_cached = loader_cached.fetch_pitching_stats(2023, 2023)
        batting_cached = loader_cached.fetch_team_batting(2023, 2023)

        assert mock_pitching.call_count == 1
        assert mock_batting.call_count == 1
        assert pitching_cached.equals(pitching_res)
        assert batting_cached.equals(batting_res)


@patch("algomlb.cli.ml.HistoricalDataLoader")
def test_ml_fetch_history_cli(mock_loader_class, dummy_pitching_df, dummy_batting_df):
    """Verify ML fetch-history CLI command outputs correctly in agent mode."""
    # Setup mock loader
    mock_loader = mock_loader_class.return_value
    mock_loader.fetch_pitching_stats.return_value = dummy_pitching_df
    mock_loader.fetch_team_batting.return_value = dummy_batting_df

    # Run CLI command with global --agent-mode
    result = runner.invoke(
        app,
        [
            "--agent-mode",
            "ml",
            "fetch-history",
            "--start-year",
            "2023",
            "--end-year",
            "2023",
        ],
    )

    assert result.exit_code == 0
    # Capture and verify JSON output
    lines = result.stdout.strip().split("\n")
    agent_output = None
    for line in lines:
        if line.startswith("{") and "ml.fetch-history" in line:
            agent_output = json.loads(line)
            break

    assert agent_output is not None
    assert agent_output["status"] == "success"
    assert "pitching_shape" in agent_output["data"]
    assert agent_output["data"]["pitching_shape"] == [1, 2]


def test_ml_optimize_cli():
    """Verify ml optimize CLI runs without hanging by mocking heavy logic."""
    with (
        patch("algomlb.ml.hyperopt.build_fold_data") as mock_build,
        patch("algomlb.ml.hyperopt.optimize_model") as mock_opt,
        patch("algomlb.cli.ml.pd.read_sql") as mock_read,
        patch("algomlb.cli.ml.get_session_factory"),
        patch("pathlib.Path.mkdir"),  # Prevent directory creation
        patch("builtins.open", MagicMock()),  # Prevent file writing
    ):
        mock_build.return_value = {"fold1": {}}
        mock_opt.return_value = ({}, MagicMock())
        mock_read.return_value = pd.DataFrame(
            [
                {
                    "game_pk": 1,
                    "game_date": "2023-04-01",
                    "home_team": "A",
                    "away_team": "B",
                    "home_pitcher_id": 1,
                    "away_pitcher_id": 2,
                    "home_score": 5,
                    "away_score": 3,
                }
            ]
        )

        result = runner.invoke(app, ["ml", "optimize", "--n-trials", "1"])
        assert result.exit_code == 0
        assert mock_opt.called
        assert mock_build.called
