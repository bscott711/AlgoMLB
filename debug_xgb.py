from algomlb.ml.model import MLBModel
from algomlb.db.session import get_session_factory
from algomlb.ml.monte_carlo.engine import SimulationEngine
from algomlb.ml.monte_carlo.loader import MatchupLoader
from pathlib import Path
import numpy as np

model = MLBModel.load(Path('.data/models/pa_outcome_v1.1.joblib'))
session = get_session_factory()()
loader = MatchupLoader(session)
ctx = loader.load_matchup(746849)

engine = SimulationEngine(model)
original = model.predict_proba

X_saved = []
def patched_predict(X):
    X_saved.append(X)
    return original(X)
model.predict_proba = patched_predict
engine.pa_model = model

engine._precompute_matchups(ctx)
X = X_saved[0]

# XGBoost probabilities without calibration (if using CalibratedClassifierCV, get_base_xgb_estimator gets one)
base = model.get_base_xgb_estimator()
base_probs = base.predict_proba(X)
print("BASE XGBOOST VAR:", np.var(base_probs, axis=0))
print("BASE XGBOOST PROBS:\n", base_probs[:2])

# Feature Importances
try:
    imp = model.get_feature_importance()
    print("\nFEATURE IMPORTANCES:")
    print(imp.sort_values(by='importance', ascending=False).head(15))
except Exception as e:
    print(e)
