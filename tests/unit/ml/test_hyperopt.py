import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import optuna
import json

from algomlb.ml.hyperopt import (
    _run_fold,
    walk_forward_objective,
    build_fold_data,
    optimize_model,
    load_optimized_params,
)


@pytest.fixture
def mock_fold_data():
    X_tr = pd.DataFrame({"feat": [1, 2]})
    y_tr = pd.Series([0, 1])
    X_te = pd.DataFrame({"feat": [3]})
    y_te = pd.Series([1])
    return [(X_tr, y_tr, X_te, y_te)]


def test_run_fold():
    with patch("algomlb.ml.hyperopt.MLBModel") as mock_model_cls:
        mock_model = mock_model_cls.return_value
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])

        # log_loss needs at least two distinct labels in y_true or explicit labels arg
        score = _run_fold(
            pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series([1, 0]), {}
        )
        assert isinstance(score, float)
        assert mock_model.train.called


def test_walk_forward_objective_success(mock_fold_data):
    trial = MagicMock(spec=optuna.Trial)
    trial.suggest_int.return_value = 100
    trial.suggest_float.return_value = 0.05
    trial.should_prune.return_value = False

    with patch("algomlb.ml.hyperopt._run_fold", return_value=0.5):
        avg = walk_forward_objective(trial, mock_fold_data)
        assert avg == 0.5
        assert trial.report.called


def test_walk_forward_objective_pruning(mock_fold_data):
    trial = MagicMock(spec=optuna.Trial)
    trial.should_prune.return_value = True

    with patch("algomlb.ml.hyperopt._run_fold", return_value=10.0):
        with pytest.raises(optuna.TrialPruned):
            walk_forward_objective(trial, mock_fold_data)


def test_build_fold_data_empty_logic():
    # Test skipping fold if X is empty (Line 129)
    games_df = pd.DataFrame(
        {"game_pk": [1], "year": [2023], "game_date": ["2023-01-01"]}
    )
    years = [2022, 2023]

    with patch("algomlb.ml.hyperopt.FeaturePipeline") as mock_pipe_cls:
        mock_pipe = mock_pipe_cls.return_value
        mock_pipe.build_uranium_matrix.return_value = (pd.DataFrame(), pd.Series())

        folds = build_fold_data(
            years,
            games_df,
            pd.DataFrame(columns=["season"]),
            pd.DataFrame(columns=["season"]),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        assert len(folds) == 0


def test_optimize_model_validation():
    with pytest.raises(ValueError, match="No fold data"):
        optimize_model([])


def test_load_optimized_params_fallback():
    # Test fallback (Line 271)
    with patch("pathlib.Path.exists", return_value=False):
        params = load_optimized_params("nonexistent")
        assert params["n_estimators"] == 300


def test_load_optimized_params_success(tmp_path):
    # Test loading from file
    model_dir = tmp_path / ".data" / "models"
    model_dir.mkdir(parents=True)
    params_file = model_dir / "optuna_best_params_v99.json"
    data = {"n_estimators": 999}
    params_file.write_text(json.dumps(data))

    # Patching pathlib.Path directly since it's locally imported in the function
    with patch("pathlib.Path") as mock_path:
        mock_path.return_value = params_file
        params = load_optimized_params("v99")
        assert params["n_estimators"] == 999
