import datetime
from algomlb.db.session import get_session_factory
from algomlb.ml.monte_carlo.loader import MatchupLoader
from algomlb.ml.component_models.pa_model import PAOutcomeModel
from algomlb.ml.monte_carlo.engine import SimulationEngine

session = get_session_factory()()
loader = MatchupLoader(session)
ctx = loader.load_matchup(746849) # A random game, use anything
model = PAOutcomeModel.load("v1.1")
engine = SimulationEngine(model)
engine._precompute_matchups(ctx)
probs = list(engine.matchup_cache.values())
if probs:
    print("PROBS SAMPLE 1:", probs[0])
    print("PROBS SAMPLE 2:", probs[1])
    print("SAME?", (probs[0] == probs[1]).all())
    # Let's see what expected_features was
    import pandas as pd
    actual_model = getattr(engine.pa_model, "model", engine.pa_model)
    print("EXPECTED FEATS:", list(actual_model.feature_names_in_)[:10] if hasattr(actual_model, "feature_names_in_") else "Unknown")
