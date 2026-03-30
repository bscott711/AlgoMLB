import json
from unittest.mock import patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from algomlb.cli.main import app
from algomlb.ml import FeaturePipeline, MLBModel


def test_ml_train_cli(dummy_stats, dummy_games):
    """Verify ml train CLI command executes end-to-end with mocks."""
    runner = CliRunner()
    pitching, batting = dummy_stats

    with patch("algomlb.cli.ml.HistoricalDataLoader") as mock_loader_class:
        mock_loader = mock_loader_class.return_value
        mock_loader.fetch_pitching_stats.return_value = pitching
        mock_loader.fetch_team_batting.return_value = batting

        # Invoke CLI with --agent-mode
        result = runner.invoke(app, ["--agent-mode", "ml", "train"])

        assert result.exit_code == 0
        # Capture and verify JSON output
        lines = result.stdout.strip().split("\n")
        agent_output = None
        for line in lines:
            if line.startswith("{") and "ml.train" in line:
                agent_output = json.loads(line)
                break

        assert agent_output is not None
        assert agent_output["status"] == "success"
        assert "feature_shape" in agent_output["data"]


@pytest.fixture
def dummy_stats():
    """Create dummy pitching and batting data for testing features."""
    pitching = pd.DataFrame(
        {"team": ["NYY", "BOS"], "era": [3.5, 4.2], "so": [200, 180]}
    )
    batting = pd.DataFrame(
        {"team": ["NYY", "BOS"], "avg": [0.260, 0.250], "hr": [20, 15]}
    )
    return pitching, batting


@pytest.fixture
def dummy_games():
    """Create dummy historical games for matrix merging."""
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "date": ["2024-04-01", "2024-04-02"],
            "team_h": ["NYY", "BOS"],
            "team_a": ["BOS", "NYY"],
            "home_win": [1, 0],
        }
    )


def test_feature_pipeline_merging(dummy_stats, dummy_games):
    """Verify that build_training_matrix correctly merges stats and drops IDs."""
    pitching, batting = dummy_stats
    pipeline = FeaturePipeline()
    X, y = pipeline.build_training_matrix(dummy_games, pitching, batting)

    # Check shape
    assert len(X) == 2
    assert len(y) == 2

    assert "h_avg" in X.columns
    assert "a_era" in X.columns
    assert "game_id" not in X.columns
    assert "team_h" not in X.columns
    assert "home_win" not in X.columns


def test_ml_model_io(tmp_path):
    """Verify training, saving, and loading of the MLBModel."""
    X = pd.DataFrame({"feat1": [1.0, 0.0, 1.0, 0.0], "feat2": [0.5, 1.5, 0.5, 1.5]})
    y = pd.Series([1, 0, 1, 0])

    model = MLBModel(n_estimators=10)
    model.train(X, y)

    # Proba check
    probs = model.predict_proba(X)
    assert probs.shape == (4, 2)

    # Persistence
    save_path = tmp_path / "model.joblib"
    model.save(save_path)

    loaded_model = MLBModel.load(save_path)
    # Verify behavior of loaded model
    loaded_probs = loaded_model.predict_proba(X)
    assert (loaded_probs == probs).all()


def test_ml_train_cli_failure():
    """Verify ml train CLI command handles loader failure correctly."""
    runner = CliRunner()

    with patch("algomlb.cli.ml.HistoricalDataLoader") as mock_loader_class:
        mock_loader = mock_loader_class.return_value
        mock_loader.fetch_pitching_stats.side_effect = Exception("Load error")

        # Invoke CLI - should exit with code 1
        result = runner.invoke(app, ["ml", "train"])

        assert result.exit_code == 1


@patch("algomlb.cli.ml.MLBModel")
@patch("algomlb.cli.ml.FeaturePipeline")
def test_ml_train_cli_no_team_column(
    mock_pipeline_class, mock_model_class, dummy_stats
):
    """Verify ml train CLI command fallback to default teams when 'team' column missing."""
    runner = CliRunner()
    pitching, batting = dummy_stats
    # Remove 'team' column to trigger fallback line 68
    pitching = pitching.drop(columns=["team"])

    # Setup mocks
    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.build_training_matrix.return_value = (
        pd.DataFrame({"feat": [1]}),
        pd.Series([1]),
    )
    mock_model = mock_model_class.return_value

    with patch("algomlb.cli.ml.HistoricalDataLoader") as mock_loader_class:
        mock_loader = mock_loader_class.return_value
        mock_loader.fetch_pitching_stats.return_value = pitching
        mock_loader.fetch_team_batting.return_value = batting

        result = runner.invoke(app, ["ml", "train"])

        assert result.exit_code == 0
        assert mock_pipeline.build_training_matrix.called
        assert mock_model.train.called
