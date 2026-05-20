from algomlb.ml.model import MLBModel
from algomlb.db.session import get_session_factory
from algomlb.ml.monte_carlo.engine import SimulationEngine
from algomlb.ml.monte_carlo.loader import MatchupLoader
from pathlib import Path
import numpy as np

model = MLBModel.load(Path(".data/models/pa_outcome_v1.1.joblib"))
session = get_session_factory()()
loader = MatchupLoader(session)
ctx = loader.load_matchup(746849)

engine = SimulationEngine(model)
engine._precompute_matchups(ctx)
probs = list(engine.matchup_cache.values())
if probs:
    print("PROBS TYPE:", type(probs[0]), probs[0].shape)
    print("SAMPLE PROBS:", probs[0])
    print("VARIANCE:", np.var(probs, axis=0))
else:
    print("NO PROBS")
