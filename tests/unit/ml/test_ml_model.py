import pandas as pd
from algomlb.ml.model import MLBModel


def test_model_init():
    model = MLBModel(n_estimators=50, max_depth=5, monotone_constraints="(1,0)")
    assert model.clf.n_estimators == 50
    assert model.clf.max_depth == 5
    # monotonicity is checked via params internal to xgb or via get_params
    assert model.clf.get_params()["monotone_constraints"] == "(1,0)"


def test_model_train_no_calibrate():
    model = MLBModel()
    X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    model.fit(X, y, calibrate=False)
    assert model.calibrated_clf is None
    # Check that it predicts something
    probs = model.predict_proba(X)
    assert probs.shape == (3, 2)


def test_model_train_calibrate():
    model = MLBModel()
    # Small data triggers 2-fold CV
    X = pd.DataFrame({"f1": range(5), "f2": range(5, 10)})
    y = pd.Series([0, 1, 0, 1, 0])
    model.fit(X, y, calibrate=True)
    assert model.calibrated_clf is not None
    probs = model.predict_proba(X)
    assert probs.shape == (5, 2)


def test_get_base_xgb_estimator():
    model = MLBModel()
    X = pd.DataFrame({"f1": range(5)})
    y = pd.Series([0, 1, 0, 1, 0])
    model.fit(X, y, calibrate=True)

    base = model.get_base_xgb_estimator()
    # Should be the XGBClassifier instance
    from xgboost import XGBClassifier

    assert isinstance(base, XGBClassifier)


def test_get_feature_importance():
    model = MLBModel()
    X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    y = pd.Series([0, 1])
    model.fit(X, y, calibrate=False)

    importance = model.get_feature_importance()
    assert len(importance) == 2
    assert "f1" in importance["feature"].values
    assert "importance" in importance.columns


def test_get_feature_importance_no_train():
    model = MLBModel()
    importance = model.get_feature_importance()
    assert importance.empty


def test_save_load(tmp_path):
    model = MLBModel(n_estimators=10)
    X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    y = pd.Series([0, 1])
    model.fit(X, y, calibrate=False)

    file_path = tmp_path / "model.joblib"
    model.save(file_path)
    assert file_path.exists()

    new_model = MLBModel.load(file_path)
    assert new_model.clf.n_estimators == 10
    assert new_model.calibrated_clf is None
