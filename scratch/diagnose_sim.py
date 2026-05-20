"""Diagnostic script to trace the root causes of unrealistic Monte Carlo projections."""

import sys

sys.path.insert(0, "/home/opc/AlgoMLB/src")

import numpy as np
import pandas as pd
from pathlib import Path
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM
from algomlb.ml.model import MLBModel
from algomlb.ml.monte_carlo.loader import MatchupLoader
from algomlb.domain import GameStatus
from sqlalchemy import select, desc

session = get_session_factory()()

# 1. Find a recent completed game
stmt = (
    select(GameResultORM)
    .where(GameResultORM.status == GameStatus.COMPLETED)
    .where(GameResultORM.home_score.isnot(None))
    .order_by(desc(GameResultORM.game_date))
    .limit(5)
)
games = session.execute(stmt).scalars().all()
print("=== RECENT COMPLETED GAMES ===")
for g in games:
    print(
        f"  game_id={g.game_id} {g.game_date} {g.away_team}@{g.home_team} score={g.away_score}-{g.home_score}"
    )

game = games[0]
game_pk = int(game.game_id)
actual_home = game.home_score
actual_away = game.away_score
print(f"\nUsing game_pk={game_pk}, actual: {actual_away}-{actual_home}")

# 2. Load context
loader = MatchupLoader(session)
ctx = loader.load_matchup(game_pk)
print(f"\nHome lineup: {len(ctx.home_lineup)} batters")
print(f"Away lineup: {len(ctx.away_lineup)} batters")
print(f"Batter features loaded: {len(ctx.batter_features)} players")
print(f"Pitcher features loaded: {len(ctx.pitcher_features)} players")

# 3. Load model using MLBModel.load (the correct way)
for ver in ["v1.2", "v1.1", "v1.0"]:
    model_path = Path(f".data/models/pa_outcome_{ver}.joblib")
    if model_path.exists():
        break

print(f"\n=== MODEL: {model_path} ===")
pa_model = MLBModel.load(model_path)
print(f"Model type: {type(pa_model)}")
print(f"Has .le: {pa_model.le is not None}")
if pa_model.le is not None:
    print(f"Label classes: {list(pa_model.le.classes_)}")
print(f"Has calibrated_clf: {pa_model.calibrated_clf is not None}")

base = pa_model.get_base_xgb_estimator()
print(f"Base estimator type: {type(base)}")

expected_features = None
if hasattr(base, "feature_names_in_"):
    expected_features = list(base.feature_names_in_)
elif hasattr(base, "feature_names") and base.feature_names is not None:
    expected_features = list(base.feature_names)
elif hasattr(base, "get_booster"):
    expected_features = base.get_booster().feature_names

print(
    f"Expected features count: {len(expected_features) if expected_features else 'N/A'}"
)
if expected_features:
    print(f"First 30 features: {expected_features[:30]}")
    print(f"Has cnt_ features: {any(f.startswith('cnt_') for f in expected_features)}")

    # Group features by prefix
    prefixes = {}
    for f in expected_features:
        prefix = f.split("_")[0] if "_" in f else f
        # Better grouping
        for p in [
            "batter_",
            "pitcher_",
            "h_bat_",
            "a_bat_",
            "cnt_",
            "elo_",
            "home_",
            "away_",
        ]:
            if f.startswith(p):
                prefix = p
                break
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    print(f"Feature prefixes: {dict(sorted(prefixes.items()))}")

# 4. Matchup test
b_id = ctx.home_lineup[0].player_id
p_id = ctx.away_starter.pitcher_id
b_feats = ctx.batter_features.get(b_id, {})
p_feats = ctx.pitcher_features.get(p_id, {})

print(f"\n=== SAMPLE MATCHUP: batter {b_id} vs pitcher {p_id} ===")
print(f"Batter features: {len(b_feats)}, keys: {sorted(b_feats.keys())[:15]}")
print(f"Pitcher features: {len(p_feats)}, keys: {sorted(p_feats.keys())[:15]}")

if expected_features:
    combined = {}
    for k, v in b_feats.items():
        combined[f"batter_{k}"] = v
    for k, v in p_feats.items():
        combined[f"pitcher_{k}"] = v
    for k in ["elo_diff", "home_team_elo_pre", "away_team_elo_pre"]:
        if k in ctx.matchup_features:
            combined[k] = ctx.matchup_features[k]

    # Add the legacy shims the engine adds
    away_batters = [b.player_id for b in ctx.away_lineup]
    home_batters = [b.player_id for b in ctx.home_lineup]

    def get_avg(team_batters):
        res = {}
        count = 0
        for b in team_batters:
            feats = ctx.batter_features.get(b, {})
            for k, v in feats.items():
                res[k] = res.get(k, 0) + v
            count += 1
        if count > 0:
            for k in res:
                res[k] /= count
        return res

    away_avg = get_avg(away_batters)
    home_avg = get_avg(home_batters)
    for k, v in home_avg.items():
        combined[f"h_bat_{k}"] = v
    for k, v in away_avg.items():
        combined[f"a_bat_{k}"] = v

    if "roll_re24" in p_feats:
        combined["pitcher_roll_re24_x"] = p_feats["roll_re24"]
        combined["pitcher_roll_re24_y"] = p_feats["roll_re24"]
    if "roll_re24" in b_feats:
        combined["batter_roll_re24_x"] = b_feats["roll_re24"]
        combined["batter_roll_re24_y"] = b_feats["roll_re24"]
    for f_name in [
        "n_games_used",
        "window_games",
        "days_since_last_game",
        "fatigue_index_7d",
        "fatigue_index_14d",
    ]:
        if f_name in p_feats:
            combined[f"pitcher_{f_name}"] = p_feats[f_name]
        if f_name in b_feats:
            combined[f"batter_{f_name}"] = b_feats[f_name]

    matched = [f for f in expected_features if f in combined]
    missing = [f for f in expected_features if f not in combined]
    print(f"\nFeatures matched: {len(matched)} / {len(expected_features)}")
    print(f"Features MISSING (filled with 0.0): {len(missing)}")
    if missing:
        print(f"Missing features: {missing}")

    row = pd.DataFrame([combined]).reindex(columns=expected_features, fill_value=0.0)
    zero_cols = int((row.iloc[0] == 0.0).sum())
    nonzero_cols = int((row.iloc[0] != 0.0).sum())
    print(f"ZERO columns: {zero_cols}, NON-ZERO: {nonzero_cols}")

    # Show actual feature values for non-zero columns
    print("\nNon-zero feature values:")
    for col in expected_features:
        val = row.iloc[0][col]
        if val != 0.0:
            print(f"  {col:45s} = {val:.6f}")

    probs = pa_model.predict_proba(row)
    outcome_map = (
        list(pa_model.le.classes_)
        if pa_model.le
        else [
            "double",
            "hbp",
            "home_run",
            "out_in_play",
            "single",
            "strikeout",
            "triple",
            "walk",
        ]
    )

    print("\n=== PREDICTED PROBABILITIES vs MLB BASELINE ===")
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
    for name, prob in zip(outcome_map, probs[0]):
        real_p = expected_mlb.get(name, 0)
        delta = prob - real_p
        flag = "⚠️" if abs(delta) > 0.05 else "  "
        print(
            f"  {flag} {name:15s}: model={prob:.4f}  real≈{real_p:.3f}  Δ={delta:+.4f}"
        )

    on_base_rate = sum(
        probs[0][outcome_map.index(o)]
        for o in ["single", "double", "triple", "home_run", "walk", "hbp"]
        if o in outcome_map
    )
    out_rate = sum(
        probs[0][outcome_map.index(o)]
        for o in ["strikeout", "out_in_play"]
        if o in outcome_map
    )
    print(f"\n  OBP (model): {on_base_rate:.4f}")
    print(f"  Out rate:    {out_rate:.4f}")

# 5. Simulation
from algomlb.ml.monte_carlo.engine import SimulationEngine

engine = SimulationEngine(pa_model, seed=42)
print("\n=== RUNNING 100 TRIAL SIMULATION ===")
results = engine.run_trials(ctx, trials=100)

home_scores = [r.home_score for r in results]
away_scores = [r.away_score for r in results]
print(
    f"Home: mean={np.mean(home_scores):.2f}, std={np.std(home_scores):.2f}, range=[{min(home_scores)},{max(home_scores)}]"
)
print(
    f"Away: mean={np.mean(away_scores):.2f}, std={np.std(away_scores):.2f}, range=[{min(away_scores)},{max(away_scores)}]"
)
print(f"Total: mean={np.mean([h + a for h, a in zip(home_scores, away_scores)]):.2f}")
print(f"Actual: {actual_away}-{actual_home}")

# 6. Player props
from algomlb.ml.monte_carlo.aggregator import SimulationAggregator

agg = SimulationAggregator()
df = agg.aggregate_results(game_pk, 2026, results, ctx)

r_props = df[df["stat_type"] == "R"]
h_props = df[df["stat_type"] == "H"]
print("\n=== PLAYER: xR, xH ===")
for b in ctx.home_lineup + ctx.away_lineup:
    pid = b.player_id
    name = b.player_name or str(pid)
    xr = r_props[r_props["player_id"] == pid]["mean"].values
    xh = h_props[h_props["player_id"] == pid]["mean"].values
    side = "HOME" if b in ctx.home_lineup else "AWAY"
    print(
        f"  [{side}] {name:25s}: xR={xr[0]:.3f}  xH={xh[0]:.3f}"
        if len(xr) > 0 and len(xh) > 0
        else f"  [{side}] {name:25s}: NO DATA"
    )

h_pids = [b.player_id for b in ctx.home_lineup]
a_pids = [b.player_id for b in ctx.away_lineup]
h_xr = r_props[r_props["player_id"].isin(h_pids)]["mean"].sum()
a_xr = r_props[r_props["player_id"].isin(a_pids)]["mean"].sum()
print(f"\nTeam xR sums: Home={h_xr:.2f}, Away={a_xr:.2f}")
print(f"Sim means:    Home={np.mean(home_scores):.2f}, Away={np.mean(away_scores):.2f}")
print(
    f"GAP:          Home={np.mean(home_scores) - h_xr:.2f}, Away={np.mean(away_scores) - a_xr:.2f}"
)

# 7. Single trial trace
print("\n=== SINGLE TRIAL TRACE (trial 0) ===")
r0 = results[0]
print(f"Score: Away {r0.away_score} - Home {r0.home_score}, innings={r0.inning_count}")
from algomlb.ml.monte_carlo.state import BatterSimState as BSS

total_pa = 0
total_h = 0
total_hr = 0
total_bb = 0
total_k = 0
for pid, state in r0.player_states.items():
    if isinstance(state, BSS):
        name = state.player_name or str(pid)
        total_pa += state.pa_count
        total_h += state.hits
        total_hr += state.hr
        total_bb += state.walks
        total_k += state.strikeouts
        print(
            f"  {name:25s}: PA={state.pa_count} H={state.hits}(1B={state.singles} 2B={state.doubles} 3B={state.triples} HR={state.hr}) R={state.runs} RBI={state.rbi} BB={state.walks} K={state.strikeouts}"
        )
print(f"\n  TOTALS: PA={total_pa} H={total_h} HR={total_hr} BB={total_bb} K={total_k}")
print(f"  Hit rate: {total_h / total_pa:.3f}" if total_pa > 0 else "")
print(f"  K rate:   {total_k / total_pa:.3f}" if total_pa > 0 else "")
print(f"  BB rate:  {total_bb / total_pa:.3f}" if total_pa > 0 else "")

session.close()
print("\n=== DIAGNOSIS COMPLETE ===")
