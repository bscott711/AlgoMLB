import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from algomlb.ml.hyperopt import build_fold_data, optimize_model


@pytest.fixture
def dummy_data():
    years = [2021, 2022, 2023]
    games = pd.DataFrame(
        [
            {"game_pk": 1, "year": 2021, "game_date": "2021-04-01"},
            {"game_pk": 2, "year": 2022, "game_date": "2022-04-01"},
            {"game_pk": 3, "year": 2023, "game_date": "2023-04-01"},
        ]
    )
    pitcher = pd.DataFrame(
        [
            {"player_id": 1, "season": 2021},
            {"player_id": 1, "season": 2022},
            {"player_id": 1, "season": 2023},
        ]
    )
    return years, games, pitcher


def test_build_fold_data(dummy_data):
    years, games, pitcher = dummy_data

    # Mock FeaturePipeline to return dummy X, y
    with patch("algomlb.ml.hyperopt.FeaturePipeline") as mock_pipe_cls:
        mock_pipe = mock_pipe_cls.return_value

        def mock_build(games, *args, **kwargs):
            # Return a DF with same index as input games
            return pd.DataFrame(
                {"feat": [0.1] * len(games)}, index=games.index
            ), pd.Series([1] * len(games), index=games.index)

        mock_pipe.build_uranium_matrix.side_effect = mock_build

        empty_df = pd.DataFrame(columns=["season", "game_pk"])

        folds = build_fold_data(
            years, games, pitcher, empty_df, empty_df, empty_df, empty_df, empty_df
        )

        assert len(folds) == 2


@patch("optuna.create_study")
def test_optimize_model(mock_create_study):
    mock_study = MagicMock()
    mock_create_study.return_value = mock_study
    mock_study.best_params = {"n_estimators": 100}
    mock_study.best_value = 0.5

    fold_data = [
        (
            pd.DataFrame({"a": [1]}),
            pd.Series([1]),
            pd.DataFrame({"a": [1]}),
            pd.Series([1]),
        )
    ]
    params, study = optimize_model(fold_data, n_trials=1)

    assert params["n_estimators"] == 100
    assert mock_study.optimize.called
