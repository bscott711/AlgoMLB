import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from algomlb.ml.shap_analysis import compute_global_shap, persist_global_shap


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.get_base_xgb_estimator.return_value = MagicMock()
    return model


@pytest.fixture
def sample_x():
    return pd.DataFrame({"feat1": np.random.rand(10), "feat2": np.random.rand(10)})


def test_compute_global_shap_binary(mock_model, sample_x):
    with patch("shap.TreeExplainer") as mock_explainer_cls:
        mock_explainer = mock_explainer_cls.return_value
        # Binary case: ndarray output
        mock_explainer.shap_values.return_value = np.random.rand(10, 2)

        result = compute_global_shap(mock_model, sample_x, sample_n=5)
        assert len(result) == 2
        assert "feature_name" in result.columns
        assert not result.empty


def test_compute_global_shap_multiclass(mock_model, sample_x):
    with patch("shap.TreeExplainer") as mock_explainer_cls:
        mock_explainer = mock_explainer_cls.return_value
        # Multiclass case: list of ndarrays
        mock_explainer.shap_values.return_value = [
            np.random.rand(10, 2),
            np.random.rand(10, 2),
        ]

        result = compute_global_shap(mock_model, sample_x)
        assert len(result) == 2


def test_compute_global_shap_import_error(mock_model, sample_x):
    # Simulate shap not installed
    with patch.dict("sys.modules", {"shap": None}):
        result = compute_global_shap(mock_model, sample_x)
        assert result.empty
        assert list(result.columns) == ["feature_name", "mean_abs_shap", "mean_shap"]


def test_persist_global_shap_empty():
    persist_global_shap(None, "v1", "test", pd.DataFrame())
    # Should just return early (covered by lack of crash)


def test_persist_global_shap_workflow():
    mock_engine = MagicMock()
    mock_conn = mock_engine.begin.return_value.__enter__.return_value

    shap_df = pd.DataFrame(
        [{"feature_name": "feat1", "mean_abs_shap": 0.5, "mean_shap": 0.1}]
    )

    with (
        patch("algomlb.ml.shap_analysis.get_engine", return_value=mock_engine),
        patch("algomlb.ml.shap_analysis.pg_insert") as mock_insert,
    ):
        persist_global_shap(mock_engine, "v1", "test", shap_df)
        assert mock_conn.execute.called
        assert mock_insert.called
