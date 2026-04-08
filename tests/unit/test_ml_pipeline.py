import json
from unittest.mock import patch

import pandas as pd
import pytest
import numpy as np
from typer.testing import CliRunner

from algomlb.cli.main import app
from algomlb.ml import FeaturePipeline, MLBModel


@patch("algomlb.cli.ml.pd.read_sql")
def test_ml_train_cli(mock_read_sql, dummy_stats, dummy_games):
    """Verify ml train CLI command executes end-to-end with mocks."""
    runner = CliRunner()
    pitching, batting = dummy_stats

    # Setup mocks for multiple read_sql calls in cli/ml.py
    # Add varying features (roll_era, roll_avg) to prevent the pipeline from dropping all constant columns
    mock_read_sql.side_effect = [
        dummy_games,
        pd.DataFrame({
            "player_id": [1, 2, 2, 1], 
            "game_date": ["2024-04-01", "2024-04-01", "2024-04-02", "2024-04-02"], 
            "roll_era": [3.0, 4.0, 2.0, 5.0],
            "season": [2024, 2024, 2024, 2024], 
            "role": ["PITCHER", "PITCHER", "PITCHER", "PITCHER"]
        }),
        pd.DataFrame({
            "game_pk": [1001, 1002], 
            "game_date": ["2024-04-01", "2024-04-02"], 
            "player_id": [1, 2], 
            "team_side": ["home", "home"], 
            "batting_order": [1, 1]
        }),
        pd.DataFrame({
            "player_id": [1, 2], 
            "game_date": ["2024-04-01", "2024-04-02"], 
            "roll_avg_launch_speed": [90.0, 95.0],
            "season": [2024, 2024], 
            "role": ["BATTER", "BATTER"]
        }),
        pd.DataFrame(),  # Elo (optional)
    ]

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
        {
            "player_id": [1, 2, 2, 1, 1, 2, 2, 1],
            "team": ["NYY", "BOS", "BOS", "NYY", "NYY", "BOS", "BOS", "NYY"],
            "era": [3.5, 4.2, 3.0, 4.0, 3.2, 4.5, 2.8, 4.1],
            "so": [200, 180, 210, 190, 205, 175, 215, 185],
            "game_date": ["2024-04-01", "2024-04-01", "2024-04-02", "2024-04-02", "2024-04-03", "2024-04-03", "2024-04-04", "2024-04-04"],
        }
    )
    batting = pd.DataFrame(
        {
            "team": ["NYY", "BOS", "BOS", "NYY", "NYY", "BOS", "BOS", "NYY"], 
            "avg": [0.260, 0.250, 0.270, 0.240, 0.265, 0.245, 0.275, 0.235], 
            "hr": [20, 15, 22, 18, 21, 14, 23, 17],
            "game_date": ["2024-04-01", "2024-04-01", "2024-04-02", "2024-04-02", "2024-04-03", "2024-04-03", "2024-04-04", "2024-04-04"],
        }
    )
    return pitching, batting


@pytest.fixture
def dummy_games():
    """Create dummy historical games for matrix merging."""
    return pd.DataFrame(
        {
            "game_pk": [1001, 1002, 1003, 1004],
            "game_date": ["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"],
            "home_pitcher_id": [1, 2, 1, 2],
            "away_pitcher_id": [2, 1, 2, 1],
            "home_team": ["NYY", "BOS", "NYY", "BOS"],
            "away_team": ["BOS", "NYY", "BOS", "NYY"],
            "home_score": [5, 2, 6, 1],
            "away_score": [2, 5, 3, 4],
        }
    )


def test_feature_pipeline_merging(dummy_stats, dummy_games):
    """Verify that build_training_matrix correctly merges stats and drops IDs."""
    pitching, batting = dummy_stats
    # Combine stats as would happen in a real pipeline
    stats = pitching.merge(batting, on=["team", "game_date"])

    pipeline = FeaturePipeline()
    # Add roll_ prefix manually to mock what Silver/Gold layer would look like
    # and ensure columns are in BATTER_AGG_COLS or PITCHER features
    stats_prefixed = stats.copy()
    stats_prefixed.columns = [
        f"roll_{c}" if c not in ["player_id", "game_date", "team"] else c 
        for c in stats_prefixed.columns
    ]
    X, y = pipeline.build_uranium_matrix(dummy_games, stats_prefixed)

    # Check shape
    assert len(X) == 4
    assert len(y) == 4

    assert "h_sp_roll_era" in X.columns
    assert "a_sp_roll_avg" in X.columns
    assert "game_pk" not in X.columns
    assert "home_win" not in X.columns


def test_ml_model_io(tmp_path):
    """Verify training, saving, and loading of the MLBModel."""
    X = pd.DataFrame({"feat1": [1.0, 0.0, 1.0, 0.0], "feat2": [0.5, 1.5, 0.5, 1.5]})
    y = pd.Series([1, 0, 1, 0])

    model = MLBModel(n_estimators=10)
    # Use cv=2 for small test data

    model.train(X, y, calibrate=True)

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

    # Mock pd.read_sql to simulate data loading error
    with patch("algomlb.cli.ml.pd.read_sql") as mock_read_sql:
        mock_read_sql.side_effect = Exception("Load error")
        # Invoke CLI - should exit with code 1
        result = runner.invoke(app, ["ml", "train"])
        assert result.exit_code == 1


@patch("algomlb.cli.ml.pd.read_sql")
@patch("algomlb.cli.ml.MLBModel")
@patch("algomlb.cli.ml.FeaturePipeline")
def test_ml_train_cli_no_team_column(
    mock_pipeline_class, mock_model_class, mock_read_sql, dummy_stats
):
    """Verify ml train CLI command fallback to default teams when 'team' column missing."""
    runner = CliRunner()
    pitching, batting = dummy_stats
    # Remove 'team' column to trigger fallback line 68
    pitching = pitching.drop(columns=["team"])

    # Setup DB mocks
    mock_read_sql.side_effect = [
        pd.DataFrame({
            "game_pk": [1001, 1002], 
            "game_date": ["2024-04-01", "2024-04-02"], 
            "home_team": ["NYY", "BOS"], 
            "away_team": ["BOS", "NYY"],
            "home_score": [5, 2],
            "away_score": [2, 5]
        }),
        pd.DataFrame({
            "player_id": [1, 2], 
            "game_date": ["2024-04-01", "2024-04-02"], 
            "season": [2024, 2024], 
            "role": ["PITCHER", "PITCHER"]
        }),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    ]

    # Setup mocks
    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.build_uranium_matrix.return_value = (
        pd.DataFrame({"feat": [1, 1]}),
        pd.Series([1, 0]),
    )
    mock_model = mock_model_class.return_value
    mock_model.predict_proba.return_value = np.array([[0.5, 0.5], [0.5, 0.5]])

    with patch("algomlb.cli.ml.HistoricalDataLoader") as mock_loader_class:
        mock_loader = mock_loader_class.return_value
        mock_loader.fetch_pitching_stats.return_value = pitching
        mock_loader.fetch_team_batting.return_value = batting

        runner.invoke(app, ["ml", "train"])

        assert mock_model.train.called


@patch("algomlb.cli.ml.pd.read_sql")
@patch("algomlb.cli.ml.MLBModel")
@patch("algomlb.cli.ml.FeaturePipeline")
def test_ml_train_cli_single_team(
    mock_pipeline_class, mock_model_class, mock_read_sql, dummy_stats, dummy_games
):
    """Verify ml train CLI command fallback when only 1 team available."""
    runner = CliRunner()
    pitching, batting = dummy_stats

    # Setup DB mocks
    mock_read_sql.side_effect = [
        dummy_games,
        pd.DataFrame({
            "player_id": [1, 2, 2, 1], 
            "game_date": ["2024-04-01", "2024-04-01", "2024-04-02", "2024-04-02"], 
            "season": [2024, 2024, 2024, 2024], 
            "role": ["PITCHER", "PITCHER", "PITCHER", "PITCHER"]
        }),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    ]
    # Ensure predict_proba returns a numeric array to avoid TypeErrors during evaluation
    mock_model = mock_model_class.return_value
    mock_model.predict_proba.return_value = np.array([[0.5, 0.5], [0.5, 0.5]])
    # Use empty dataframes to trigger fallback
    pitching = pd.DataFrame(columns=["team", "player_id"])
    batting = pd.DataFrame(columns=["team"])

    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.build_uranium_matrix.return_value = (
        pd.DataFrame({"feat": [1, 1]}),
        pd.Series([1, 0]),
    )
    mock_model = mock_model_class.return_value

    with patch("algomlb.cli.ml.HistoricalDataLoader") as mock_loader_class:
        mock_loader = mock_loader_class.return_value
        mock_loader.fetch_pitching_stats.return_value = pitching
        mock_loader.fetch_team_batting.return_value = batting

        result = runner.invoke(app, ["ml", "train"])
        assert result.exit_code == 0
        assert mock_model.train.called


def test_feature_pipeline_empty_games():
    """Verify build_training_matrix returns empty results for empty games."""
    pipeline = FeaturePipeline()
    X, y = pipeline.build_uranium_matrix(pd.DataFrame(), pd.DataFrame())
    assert X.empty
    assert y.empty


def test_feature_pipeline_missing_target(dummy_stats, dummy_games):
    """Verify build_training_matrix handles missing target column."""
    pitching, _ = dummy_stats
    # Remove scores so home_win cannot be calculated
    games = dummy_games.drop(columns=["home_score", "away_score"])
    pipeline = FeaturePipeline()
    X, y = pipeline.build_uranium_matrix(games, pitching)
    assert X.empty
    assert y.empty


def test_ml_model_train_no_calibrate():
    """Verify train method with calibrate=False."""
    X = pd.DataFrame({"feat1": [1.0, 0.0], "feat2": [0.5, 1.5]})
    y = pd.Series([1, 0])
    model = MLBModel(n_estimators=10)
    model.train(X, y, calibrate=False)
    assert model.calibrated_clf is None
    # Just verify predict still works
    probs = model.predict_proba(X)
    assert probs.shape == (2, 2)


def test_feature_pipeline_with_statcast():
    """Verify build_training_matrix with pitch events."""
    games = pd.DataFrame(
        [
            {
                "home_pitcher_id": 1,
                "away_pitcher_id": 2,
                "home_score": 5,
                "away_score": 3,
                "game_pk": 1001,
                "game_date": "2024-04-01",
            },
            {
                "home_pitcher_id": 1,
                "away_pitcher_id": 2,
                "home_score": 1,
                "away_score": 6,
                "game_pk": 1002,
                "game_date": "2024-04-02",
            },
        ]
    )
    # Varyera in second row for home pitcher to avoid constant column drop
    stats_varied = pd.DataFrame(
        [
            {"player_id": 1, "team": "NYY", "era": 3.0, "game_date": "2024-04-01"},
            {"player_id": 2, "team": "BOS", "era": 4.0, "game_date": "2024-04-02"},
            {"player_id": 3, "team": "TOR", "era": 2.5, "game_date": "2024-04-02"},
        ]
    )
    # Give game 2 a different home pitcher
    games.loc[1, "home_pitcher_id"] = 3

    pitches = pd.DataFrame(
        [
            {"player_id": 1, "roll_avg_launch_speed": 90.0, "game_date": "2024-04-01"},
            {"player_id": 3, "roll_avg_launch_speed": 95.0, "game_date": "2024-04-02"},
        ]
    )
    lineups = pd.DataFrame([
        {"game_pk": 1001, "player_id": 1, "team_side": "home", "game_date": "2024-04-01"},
        {"game_pk": 1002, "player_id": 3, "team_side": "home", "game_date": "2024-04-02"},
    ])
    pipeline = FeaturePipeline()
    X, y = pipeline.build_uranium_matrix(games, stats_varied, lineups_df=lineups, batter_gold_df=pitches)
    # era should remain
    assert "h_sp_era" in X.columns
    # Team batting aggregate adds h_bat_ prefix
    assert "h_bat_roll_avg_launch_speed" in X.columns
    assert len(X) == 2


def test_historical_loader_cache_logic(tmp_path):
    """Verify caching in HistoricalDataLoader."""
    from algomlb.ingestion.historical import HistoricalDataLoader
    from unittest.mock import MagicMock, patch

    mock_repo = MagicMock()
    loader = HistoricalDataLoader(repo=mock_repo, cache_dir=tmp_path)

    # Pre-create parquet
    cache_file = tmp_path / "statcast_2024-04-01_2024-04-01.parquet"
    pd.DataFrame(
        {"game_date": ["2024-04-01"], "pitcher": [1], "batter": [2]}
    ).to_parquet(cache_file)

    with patch("pybaseball.statcast") as mock_scrape:
        df = loader.fetch_statcast("2024-04-01", "2024-04-01", persist=False)
    assert not mock_scrape.called
    assert len(df) == 1


def test_historical_loader_clean_columns():
    """Verify columns are cleaned correctly."""
    from algomlb.ingestion.historical import HistoricalDataLoader
    from unittest.mock import MagicMock

    loader = HistoricalDataLoader(repo=MagicMock())
    df = pd.DataFrame(
        {
            "wOBA": [0.3],
            "ERA": [3.0],
            "Player Id": [1],
            "Team": ["NYY"],
            "unrelated": [1],
        }
    )
    # Clean check
    cleaned = loader._clean_columns(df)
    assert "player_id" in cleaned.columns
    assert "team" in cleaned.columns
