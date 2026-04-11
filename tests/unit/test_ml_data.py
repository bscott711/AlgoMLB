from unittest.mock import patch
import pandas as pd
from typer.testing import CliRunner
from algomlb.cli.ml import app

runner = CliRunner()


def test_ml_optimize_cli():
    """Verifies that the tune (formerly optimize) command invokes the expected logic."""
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
            pd.DataFrame({"f": [1]}, index=[0]),
            pd.Series([1], name="home_win"),
        )
        mock_opt.return_value.best_params = {"max_depth": 3}

        result = runner.invoke(
            app, ["tune", "--target", "home_win", "--trials", "1"], obj={}
        )
        assert result.exit_code == 0
        assert mock_opt.called


def test_ml_fetch_history_cli():
    """Verifies that fetch-history invokes the new fetch_eval_history function."""
    with patch("algomlb.cli.ml.fetch_eval_history") as mock_fetch:
        mock_fetch.return_value = pd.DataFrame([{"id": 1}])

        result = runner.invoke(app, ["fetch-history", "--target", "home_win"], obj={})
        assert result.exit_code == 0
        assert mock_fetch.called
