"""Verify the marginalized probabilities match expectations."""

import sys

sys.path.insert(0, "/home/opc/AlgoMLB/src")
import numpy as np
from pathlib import Path
from algomlb.ml.model import MLBModel
from algomlb.ml.monte_carlo.engine import TERMINAL_COUNT_WEIGHTS, COUNT_STATES
import pandas as pd

model = MLBModel.load(Path(".data/models/pa_outcome_v1.3.joblib"))
base = model.get_base_xgb_estimator()
expected_features = list(base.feature_names_in_)
outcome_map = list(model.le.classes_)

# Build a generic feature row
row_data = {f: 0.0 for f in expected_features}
row_data["batter_roll_hits_per_pa"] = 0.250
row_data["batter_roll_k_pct_batter"] = 0.220
row_data["batter_roll_bb_pct_batter"] = 0.080
row_data["batter_roll_avg_batter_xwoba"] = 0.320
row_data["batter_roll_batter_xwoba_shrunk"] = 0.320
row_data["batter_roll_pas"] = 50
row_data["pitcher_roll_k_pct"] = 0.220
row_data["pitcher_roll_bb_pct"] = 0.080
row_data["pitcher_roll_avg_pitcher_xwoba"] = 0.320
row_data["pitcher_roll_pitcher_xwoba_shrunk"] = 0.320
row_data["pitcher_roll_strikes_pct"] = 0.62
row_data["batter_n_games_used"] = 12
row_data["batter_days_since_last_game"] = 1
row_data["pitcher_n_games_used"] = 4
row_data["pitcher_days_since_last_game"] = 5

# Compute marginalized probabilities (same algorithm as engine)
weight_vec = np.array([TERMINAL_COUNT_WEIGHTS[cs] for cs in COUNT_STATES])
weight_vec = weight_vec / weight_vec.sum()

rows = []
for cs in COUNT_STATES:
    r = row_data.copy()
    for c in COUNT_STATES:
        r[f"cnt_{c}"] = 1.0 if c == cs else 0.0
    rows.append(r)

X = pd.DataFrame(rows)[expected_features]
prob_matrix = model.predict_proba(X)
marginalized = np.dot(weight_vec, prob_matrix)

print("=== COUNT-MARGINALIZED PROBABILITIES ===")
expected_mlb = {
    "single": 0.155,
    "double": 0.047,
    "triple": 0.004,
    "home_run": 0.033,
    "walk": 0.085,
    "hbp": 0.012,
    "strikeout": 0.227,
    "out_in_play": 0.437,
}
for name, prob in zip(outcome_map, marginalized):
    real_p = expected_mlb.get(name, 0)
    delta = prob - real_p
    flag = "⚠️" if abs(delta) > 0.03 else "✅"
    print(f"  {flag} {name:15s}: model={prob:.4f}  real≈{real_p:.3f}  Δ={delta:+.4f}")

obp = sum(
    marginalized[outcome_map.index(o)]
    for o in ["single", "double", "triple", "home_run", "walk", "hbp"]
)
out_rate = sum(marginalized[outcome_map.index(o)] for o in ["strikeout", "out_in_play"])
print(f"\n  OBP:      {obp:.4f}  (MLB ~.320)")
print(f"  Out rate: {out_rate:.4f}  (MLB ~.664)")
