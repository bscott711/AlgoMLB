"""Deeper investigation into feature mapping issues."""
import sys
sys.path.insert(0, "/home/opc/AlgoMLB/src")

import pandas as pd
from pathlib import Path
from algomlb.db.session import get_session_factory
from algomlb.db.models import PlayerRollingFeaturesORM
from algomlb.ml.model import MLBModel
from sqlalchemy import select, desc

session = get_session_factory()()

# 1. Check what columns PlayerRollingFeaturesORM actually has
print("=== PlayerRollingFeaturesORM columns ===")
for col in PlayerRollingFeaturesORM.__table__.columns:
    if col.name.startswith(('roll_', 'ema_', 'std_', 'seasonal_', 'fatigue_', 'delta_', 'n_games', 'window_', 'days_')):
        print(f"  {col.name}")

# 2. Check what model v1.2 expects vs what's available
model = MLBModel.load(Path(".data/models/pa_outcome_v1.2.joblib"))
base = model.get_base_xgb_estimator()
expected_features = list(base.feature_names_in_)

# Features the model needs that have "n_games_used" or "days_since_last_game"
metadata_feats = [f for f in expected_features if 'n_games' in f or 'days_since' in f]
print("\n=== Model metadata features ===")
for f in metadata_feats:
    print(f"  {f}")

# 3. Check if n_games_used and days_since_last_game are in the ORM
has_n_games = hasattr(PlayerRollingFeaturesORM, 'n_games_used')
has_days = hasattr(PlayerRollingFeaturesORM, 'days_since_last_game')
print(f"\nORM has n_games_used: {has_n_games}")
print(f"ORM has days_since_last_game: {has_days}")

# 4. Check a sample feature row to see what's actually populated
stmt = select(PlayerRollingFeaturesORM).order_by(desc(PlayerRollingFeaturesORM.game_date)).limit(1)
row = session.execute(stmt).scalar()
if row:
    print(f"\n=== Sample feature row (player {row.player_id}, {row.game_date}) ===")
    for col in row.__table__.columns:
        val = getattr(row, col.name)
        if val is not None and col.name not in ['id', 'player_id', 'game_date', 'role', 'season', 'game_pk']:
            print(f"  {col.name:40s} = {val}")

# 5. Examine the actual probabilities more carefully with ALL count states
print("\n=== PROBABILITY ANALYSIS ACROSS COUNT STATES ===")
b_feats_row = {}
# Use a generic batter 
for feat in expected_features:
    b_feats_row[feat] = 0.0

# Set some reasonable values for pitcher and batter features
b_feats_row['batter_roll_hits_per_pa'] = 0.250
b_feats_row['batter_roll_k_pct_batter'] = 0.220
b_feats_row['batter_roll_bb_pct_batter'] = 0.080
b_feats_row['batter_roll_avg_batter_xwoba'] = 0.320
b_feats_row['batter_roll_batter_xwoba_shrunk'] = 0.320
b_feats_row['batter_roll_pas'] = 50
b_feats_row['pitcher_roll_k_pct'] = 0.220
b_feats_row['pitcher_roll_bb_pct'] = 0.080
b_feats_row['pitcher_roll_avg_pitcher_xwoba'] = 0.320
b_feats_row['pitcher_roll_pitcher_xwoba_shrunk'] = 0.320
b_feats_row['pitcher_roll_strikes_pct'] = 0.62

outcome_map = list(model.le.classes_)
print(f"Outcomes: {outcome_map}")

# Test with each count state to see how probs change
count_states = ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]
print(f"\n{'Count':>5s}  {'out':>8s}  {'K':>8s}  {'BB':>8s}  {'1B':>8s}  {'2B':>8s}  {'HR':>8s}  {'OBP':>8s}")
print("-" * 75)

for count in count_states:
    row_data = b_feats_row.copy()
    for cs in count_states:
        row_data[f"cnt_{cs}"] = 1.0 if cs == count else 0.0
    
    X = pd.DataFrame([row_data])[expected_features]
    probs = model.predict_proba(X)[0]
    
    obp = sum(probs[outcome_map.index(o)] for o in ["single", "double", "triple", "home_run", "walk", "hbp"])
    p_k = probs[outcome_map.index("strikeout")]
    p_bb = probs[outcome_map.index("walk")]
    p_out = probs[outcome_map.index("out_in_play")]
    p_1b = probs[outcome_map.index("single")]
    p_2b = probs[outcome_map.index("double")]
    p_hr = probs[outcome_map.index("home_run")]
    
    print(f"{count:>5s}  {p_out:>8.4f}  {p_k:>8.4f}  {p_bb:>8.4f}  {p_1b:>8.4f}  {p_2b:>8.4f}  {p_hr:>8.4f}  {obp:>8.4f}")

# 6. Show what happens when ALL cnt_ features are 0 (the current bug)
row_data_no_cnt = b_feats_row.copy()
for cs in count_states:
    row_data_no_cnt[f"cnt_{cs}"] = 0.0  # All zeros means NO count info

X_no_cnt = pd.DataFrame([row_data_no_cnt])[expected_features]
probs_no_cnt = model.predict_proba(X_no_cnt)[0]
obp_nc = sum(probs_no_cnt[outcome_map.index(o)] for o in ["single", "double", "triple", "home_run", "walk", "hbp"])
print(f"\n{'NO CNT':>5s}  {probs_no_cnt[outcome_map.index('out_in_play')]:>8.4f}  {probs_no_cnt[outcome_map.index('strikeout')]:>8.4f}  {probs_no_cnt[outcome_map.index('walk')]:>8.4f}  {probs_no_cnt[outcome_map.index('single')]:>8.4f}  {probs_no_cnt[outcome_map.index('double')]:>8.4f}  {probs_no_cnt[outcome_map.index('home_run')]:>8.4f}  {obp_nc:>8.4f}")
print("⚠️  This is what the engine currently computes with all cnt_ = 0.0!")

# 7. Check the _simulate_count method and see how count interacts
print("\n\n=== KEY FINDING SUMMARY ===")
print(f"1. Model has {len(expected_features)} features, {len([f for f in expected_features if f.startswith('cnt_')])} are count features")
print("2. Engine's _model_has_count_features() should detect and enable 3D caching")
print(f"3. Walk prob at 0-0: {probs_no_cnt[outcome_map.index('walk')]:.4f} (baseline)")
walk_0_0 = None
walk_3_2 = None
for count in count_states:
    row_data = b_feats_row.copy()
    for cs in count_states:
        row_data[f"cnt_{cs}"] = 1.0 if cs == count else 0.0
    X = pd.DataFrame([row_data])[expected_features]
    probs_c = model.predict_proba(X)[0]
    if count == "0-0":
        walk_0_0 = probs_c[outcome_map.index('walk')]
    if count == "3-2":
        walk_3_2 = probs_c[outcome_map.index('walk')]

print(f"4. Walk prob at 0-0 count: {walk_0_0:.4f}")
print(f"5. Walk prob at 3-2 count: {walk_3_2:.4f}")
print(f"6. The count is making a {'BIG' if abs(walk_3_2 - walk_0_0) > 0.1 else 'small'} difference")

session.close()
