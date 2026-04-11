from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from typer.testing import CliRunner
from algomlb.cli.ml import app
from pathlib import Path

runner = CliRunner()


@pytest.fixture
def mock_session_factory():
    with patch("algomlb.cli.ml.get_session_factory") as mock:
        factory = mock.return_value
        factory.kw = {"bind": MagicMock()}
        yield mock


def test_feature_pipeline_merging():
    """Verifies that the Uranium matrix generates numeric features correctly."""
    from algomlb.ml.features import FeaturePipeline

    pipeline = FeaturePipeline()

    # Structural alignment: Prefixed non-constant numeric features are mandatory
    games_df = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2023-01-01",
                "home_team_id": 160,
                "away_team_id": 135,
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 5,
                "away_score": 2,
                "is_home": 1,
                "home_pitcher_id": 123,
                "away_pitcher_id": 456,
                "home_win": 1,
            },
            {
                "game_pk": 2,
                "game_date": "2023-01-02",
                "home_team_id": 135,
                "away_team_id": 160,
                "home_team": "Team B",
                "away_team": "Team A",
                "home_score": 1,
                "away_score": 4,
                "is_home": 1,
                "home_pitcher_id": 456,
                "away_pitcher_id": 123,
                "home_win": 0,
            },
        ]
    )
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])

    pitcher_gold = pd.DataFrame(
        [
            {"player_id": 123, "season": 2023, "game_date": "2023-01-01", "era": 3.5},
            {"player_id": 456, "season": 2023, "game_date": "2023-01-01", "era": 4.2},
            {"player_id": 456, "season": 2023, "game_date": "2023-01-02", "era": 4.1},
            {"player_id": 123, "season": 2023, "game_date": "2023-01-02", "era": 3.6},
        ]
    )
    pitcher_gold["game_date"] = pd.to_datetime(pitcher_gold["game_date"])

    X, y = pipeline.build_uranium_matrix(games_df, pitcher_gold)

    # Selection logic requires h_sp_ or a_sp_ prefixes for retention
    assert not X.empty
    assert any(c.startswith("h_sp_") for c in X.columns)
    assert len(y) == 2


def test_ml_train_cli(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.run_optuna_study") as mock_opt,
    ):
        mock_load.return_value = {
            "games": pd.DataFrame([{"game_pk": 1, "game_date": "2023-01-01"}]),
            "pitcher_gold": pd.DataFrame(),
            "lineups": pd.DataFrame(),
            "batter_gold": pd.DataFrame(),
            "elo": pd.DataFrame(),
            "pythag": pd.DataFrame(),
            "re24": pd.DataFrame(),
        }
        mock_pipe.return_value.build_uranium_matrix.return_value = (
            pd.DataFrame({"h_sp_era": [3.5, 4.2]}, index=[0, 1]),
            pd.Series([1, 0], name="home_win"),
        )
        mock_opt.return_value.best_params = {"max_depth": 5}

        result = runner.invoke(
            app, ["tune", "--target", "home_win", "--trials", "1"], obj={}
        )
        assert result.exit_code == 0
        assert mock_opt.called


def test_ml_model_io():
    from algomlb.ml.model import MLBModel
    import tempfile

    model = MLBModel(max_depth=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.joblib"
        model.save(path)
        assert path.exists()

        new_model = MLBModel.load(path)
        assert new_model.clf.get_params()["max_depth"] == 3


def test_ml_train_cli_no_team_column(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.run_optuna_study") as mock_opt,
    ):
        mock_load.return_value = {
            "games": pd.DataFrame([{"game_pk": 1, "game_date": "2023-01-01"}]),
            "pitcher_gold": pd.DataFrame(),
            "lineups": pd.DataFrame(),
            "batter_gold": pd.DataFrame(),
            "elo": pd.DataFrame(),
            "pythag": pd.DataFrame(),
            "re24": pd.DataFrame(),
        }
        mock_pipe.return_value.build_uranium_matrix.return_value = (
            pd.DataFrame({"h_sp_era": [3.5, 4.2]}, index=[0, 1]),
            pd.Series([1, 0], name="home_win"),
        )
        mock_opt.return_value.best_params = {"max_depth": 3}

        result = runner.invoke(
            app, ["tune", "--target", "home_win", "--trials", "1"], obj={}
        )
        assert result.exit_code == 0
        assert mock_opt.called


def test_ml_train_cli_single_team(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.run_optuna_study") as mock_opt,
    ):
        mock_load.return_value = {
            "games": pd.DataFrame([{"game_pk": 99, "game_date": "2023-05-01"}]),
            "pitcher_gold": pd.DataFrame(),
            "lineups": pd.DataFrame(),
            "batter_gold": pd.DataFrame(),
            "elo": pd.DataFrame(),
            "pythag": pd.DataFrame(),
            "re24": pd.DataFrame(),
        }
        mock_pipe.return_value.build_uranium_matrix.return_value = (
            pd.DataFrame({"h_sp_era": [3.5, 4.2]}, index=[0, 1]),
            pd.Series([0, 1], name="home_win"),
        )
        mock_opt.return_value.best_params = {"n_estimators": 10}

        result = runner.invoke(
            app, ["tune", "--target", "home_win", "--trials", "1"], obj={}
        )
        assert result.exit_code == 0
        assert mock_opt.called


def test_feature_pipeline_empty_games():
    from algomlb.ml.features import FeaturePipeline

    pipeline = FeaturePipeline()
    X, y = pipeline.build_uranium_matrix(pd.DataFrame(), pd.DataFrame())
    assert X.empty
    assert y.empty


def test_historical_loader_cache_logic():
    from algomlb.cli.ml import _load_ml_data

    mock_engine = MagicMock()
    with patch("pandas.read_sql") as mock_read:
        # Structural alignment: mandatory columns for pythagorean features
        mock_read.return_value = pd.DataFrame(
            {
                "game_pk": [1],
                "game_date": ["2023-01-01"],
                "home_team": ["Team A"],
                "away_team": ["Team B"],
                "home_score": [5],
                "away_score": [2],
            }
        )
        data = _load_ml_data(mock_engine, "2023")
        assert "games" in data
        assert mock_read.call_count >= 4
