from algomlb.ml.model import MLBModel
from algomlb.db.session import get_session_factory
from algomlb.ml.monte_carlo.loader import MatchupLoader
from pathlib import Path

model = MLBModel.load(Path('.data/models/pa_outcome_v1.1.joblib'))
session = get_session_factory()()
loader = MatchupLoader(session)
ctx = loader.load_matchup(746849)

b_feats = ctx.batter_features.get(ctx.away_lineup[0].player_id, {})
combined = {}
for k,v in b_feats.items():
    combined[f"batter_{k}"] = v

print("Sample Batter Injected Keys:", list(combined.keys())[:15])

base = model.get_base_xgb_estimator()
expected = list(base.feature_names_in_)
print("\nEXPECTED XGBOOST FEATURES:", expected[:15])
print("\nEXPECTED XGBOOST FEATURES END:", expected[-15:])

# Check intersection
intersect = set(expected).intersection(set(combined.keys()))
print("\nINTERSECTION COUNT (BATTER):", len(intersect))
print("\nINTERSECTING KEYS:", list(intersect)[:15])

