from typer.testing import CliRunner
from algomlb.cli.ml import app
from unittest.mock import patch
import pandas as pd

runner = CliRunner()


def test_cli_ml_stubs():
    """Verifies that the CLI commands exist and accept modernized parameters."""
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.run_optuna_study"),
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
            pd.DataFrame({"f": [1]}),
            pd.Series([1], name="home_win"),
        )

        # Verify tune (modernized renamed command)
        result = runner.invoke(
            app, ["tune", "--target", "home_win", "--trials", "1"], obj={}
        )
        assert result.exit_code == 0

        # Verify elo-backfill
        with patch("algomlb.ml.elo.backfill_team_elo_history"):
            result = runner.invoke(app, ["elo-backfill"], obj={})
            assert result.exit_code == 0
