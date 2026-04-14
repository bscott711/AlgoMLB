from algomlb.ml.model import MLBModel
from algomlb.db.session import get_session_factory
from algomlb.ml.monte_carlo.engine import SimulationEngine
from algomlb.ml.monte_carlo.loader import MatchupLoader
from pathlib import Path
import pandas as pd
import numpy as np

model = MLBModel.load(Path('.data/models/pa_outcome_v1.1.joblib'))
session = get_session_factory()()
loader = MatchupLoader(session)
ctx = loader.load_matchup(746849)

engine = SimulationEngine(model)
engine._precompute_matchups(ctx)

# Patch the engine temporarily to capture X_batch
original_predict = model.predict_proba
def patched_predict(X):
    print("X_batch shape:", X.shape)
    print("X_batch head:\n", X.head(2).T)
    return original_predict(X)

model.predict_proba = patched_predict
engine.pa_model = model

engine._precompute_matchups(ctx)
