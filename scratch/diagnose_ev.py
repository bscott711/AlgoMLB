"""
Diagnostic script to trace the EV calculation pipeline for today's slate.
Focuses on COL @ LAD to understand why COL is showing +9% EV.
"""
import sys
sys.path.insert(0, "src")

from datetime import date
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM, LiveOddsORM, PlayerRollingFeaturesORM, TeamEloHistoryORM, TeamSabermetricsHistoryORM, GameLineupORM
from algomlb.ml.monte_carlo.loader import MatchupLoader
from algomlb.domain import PlayerRole
from pathlib import Path
import pandas as pd

today = date.today()
session = get_session_factory()()

print(f"="*80)
print(f"DIAGNOSTIC: EV Pipeline Audit for {today}")
print(f"="*80)

# 1. Find today's games
games = session.query(GameResultORM).filter(GameResultORM.game_date == today).all()
print(f"\n📅 Games found for {today}: {len(games)}")
for g in games:
    print(f"  {g.away_team} @ {g.home_team} (id={g.game_id}, SP: {g.away_pitcher} vs {g.home_pitcher})")
    print(f"    home_pitcher_id={g.home_pitcher_id}, away_pitcher_id={g.away_pitcher_id}")

# 2. Focus on COL game
col_game = None
for g in games:
    if "Colorado" in (g.home_team or "") or "Colorado" in (g.away_team or ""):
        col_game = g
        break

if not col_game:
    print("\n❌ No COL game found today!")
    sys.exit(1)

print(f"\n{'='*80}")
print(f"🔍 DEEP DIVE: {col_game.away_team} @ {col_game.home_team} (game_id={col_game.game_id})")
print(f"{'='*80}")

# 3. Check Odds Data
print(f"\n--- LAYER 1: ODDS DATA ---")
all_odds = (
    session.query(LiveOddsORM)
    .filter(LiveOddsORM.game_result_id == str(col_game.game_id))
    .filter(LiveOddsORM.market_type.in_(["moneyline", "h2h"]))
    .order_by(LiveOddsORM.timestamp.desc())
    .all()
)
print(f"  Total moneyline/h2h odds records: {len(all_odds)}")
for o in all_odds[:10]:
    implied = 1.0 / o.price if o.price > 0 else 0.0
    print(f"    [{o.sportsbook}] outcome={o.outcome}, price={o.price:.4f} (decimal), implied={implied:.4f}, ts={o.timestamp}")
    
# Replicate what picks.py does
latest_odds = (
    session.query(LiveOddsORM)
    .filter(LiveOddsORM.game_result_id == str(col_game.game_id))
    .filter(LiveOddsORM.market_type.in_(["moneyline", "h2h"]))
    .order_by(LiveOddsORM.timestamp.desc())
    .first()
)

if latest_odds:
    print(f"\n  📊 Latest odds used by picks.py:")
    print(f"    outcome: {latest_odds.outcome}")
    print(f"    price (decimal): {latest_odds.price}")
    
    implied_prob = 1.0 / latest_odds.price if latest_odds.price > 0 else 0.5
    print(f"    implied_prob (raw): {implied_prob:.4f}")
    
    if latest_odds.outcome == col_game.home_team:
        h_implied = implied_prob
    else:
        h_implied = 1.0 - implied_prob
    
    print(f"    h_implied (home win implied): {h_implied:.4f}")
    print(f"    home_team: {col_game.home_team}, away_team: {col_game.away_team}")
    print(f"    ⚠️ outcome matches home? {latest_odds.outcome == col_game.home_team}")
else:
    print("  ❌ NO ODDS FOUND!")

# 4. Check Elo
print(f"\n--- LAYER 2: ELO DATA ---")
for team_name, label in [(col_game.home_team, "HOME"), (col_game.away_team, "AWAY")]:
    elo = (
        session.query(TeamEloHistoryORM)
        .filter(
            TeamEloHistoryORM.team_id == str(team_name),
            TeamEloHistoryORM.game_date < today,
        )
        .order_by(TeamEloHistoryORM.game_date.desc(), TeamEloHistoryORM.id.desc())
        .first()
    )
    if elo:
        print(f"  {label} ({team_name}): elo_post={elo.elo_post:.1f}, as_of={elo.game_date}")
    else:
        print(f"  {label} ({team_name}): ❌ NO ELO FOUND (will default to 1500!)")

# 5. Check Sabermetrics
print(f"\n--- LAYER 3: SABERMETRICS/PYTHAG DATA ---")
for team_name, label in [(col_game.home_team, "HOME"), (col_game.away_team, "AWAY")]:
    saber = (
        session.query(TeamSabermetricsHistoryORM)
        .filter(
            TeamSabermetricsHistoryORM.team_id == str(team_name),
            TeamSabermetricsHistoryORM.game_date < today,
        )
        .order_by(TeamSabermetricsHistoryORM.game_date.desc(), TeamSabermetricsHistoryORM.id.desc())
        .first()
    )
    if saber:
        print(f"  {label} ({team_name}): pythag={saber.pythag_win_pct:.4f}, run_diff={saber.roll_run_diff:.2f}, RS/G={saber.roll_rs_per_game:.2f}, RA/G={saber.roll_ra_per_game:.2f}, as_of={saber.game_date}")
    else:
        print(f"  {label} ({team_name}): ❌ NO SABER FOUND (will default to 0.500 / 4.5!)")

# 6. Check Lineups
print(f"\n--- LAYER 4: LINEUP DATA ---")
for side in ["home", "away"]:
    lineup = (
        session.query(GameLineupORM)
        .filter(GameLineupORM.game_pk == int(col_game.game_id))
        .filter(GameLineupORM.team_side == side)
        .order_by(GameLineupORM.batting_order)
        .all()
    )
    team = col_game.home_team if side == "home" else col_game.away_team
    print(f"  {side.upper()} ({team}): {len(lineup)} players")
    for p in lineup:
        # Check if this player has rolling features
        rf = (
            session.query(PlayerRollingFeaturesORM)
            .filter(
                PlayerRollingFeaturesORM.player_id == p.player_id,
                PlayerRollingFeaturesORM.game_date <= today,
                PlayerRollingFeaturesORM.role == PlayerRole.BATTER,
            )
            .order_by(PlayerRollingFeaturesORM.game_date.desc())
            .first()
        )
        if rf:
            print(f"    #{p.batting_order} {p.player_name} (id={p.player_id}): xwoba_shrunk={rf.roll_batter_xwoba_shrunk}, n_games={rf.n_games_used}, as_of={rf.game_date}, quality={rf.baseline_quality}")
        else:
            print(f"    #{p.batting_order} {p.player_name} (id={p.player_id}): ❌ NO ROLLING FEATURES!")

# 7. Check Pitchers
print(f"\n--- LAYER 5: PITCHER FEATURES ---")
for pid, name, label in [
    (col_game.home_pitcher_id, col_game.home_pitcher, "HOME SP"),
    (col_game.away_pitcher_id, col_game.away_pitcher, "AWAY SP"),
]:
    if not pid:
        print(f"  {label}: ❌ NO PITCHER ID!")
        continue
    rf = (
        session.query(PlayerRollingFeaturesORM)
        .filter(
            PlayerRollingFeaturesORM.player_id == pid,
            PlayerRollingFeaturesORM.game_date <= today,
            PlayerRollingFeaturesORM.role == PlayerRole.PITCHER,
            PlayerRollingFeaturesORM.n_games_used > 0,
        )
        .order_by(PlayerRollingFeaturesORM.game_date.desc())
        .first()
    )
    if rf:
        print(f"  {label} ({name}, id={pid}):")
        print(f"    xwoba_shrunk={rf.roll_pitcher_xwoba_shrunk}, n_games={rf.n_games_used}, as_of={rf.game_date}")
        print(f"    k_pct={rf.roll_k_pct}, bb_pct={rf.roll_bb_pct}, whiff={rf.roll_whiff_pct}")
        print(f"    fatigue_7d={rf.fatigue_index_7d}, fatigue_14d={rf.fatigue_index_14d}")
        print(f"    quality={rf.baseline_quality}")
    else:
        # Try without n_games_used filter
        rf_fallback = (
            session.query(PlayerRollingFeaturesORM)
            .filter(
                PlayerRollingFeaturesORM.player_id == pid,
                PlayerRollingFeaturesORM.game_date <= today,
                PlayerRollingFeaturesORM.role == PlayerRole.PITCHER,
            )
            .order_by(PlayerRollingFeaturesORM.game_date.desc())
            .first()
        )
        if rf_fallback:
            print(f"  {label} ({name}, id={pid}): ⚠️ ONLY FALLBACK (n_games_used={rf_fallback.n_games_used}, quality={rf_fallback.baseline_quality})")
        else:
            print(f"  {label} ({name}, id={pid}): ❌ NO FEATURES AT ALL!")

# 8. Run the actual prediction
print(f"\n--- LAYER 6: MODEL PREDICTION ---")
try:
    loader = MatchupLoader(session)
    ctx = loader.load_matchup(int(col_game.game_id))
    
    print(f"  Matchup Features:")
    for k, v in sorted(ctx.matchup_features.items()):
        print(f"    {k}: {v:.4f}")
    
    print(f"\n  Home Lineup Coverage: {len([b for b in ctx.home_lineup if b.player_id in ctx.batter_features])}/{len(ctx.home_lineup)}")
    print(f"  Away Lineup Coverage: {len([b for b in ctx.away_lineup if b.player_id in ctx.batter_features])}/{len(ctx.away_lineup)}")
    
    # Count missing batter features
    for side, lineup, label in [("home", ctx.home_lineup, "HOME"), ("away", ctx.away_lineup, "AWAY")]:
        missing = [b for b in lineup if b.player_id not in ctx.batter_features]
        if missing:
            print(f"\n  ⚠️ {label} batters WITHOUT features:")
            for b in missing:
                print(f"    - {b.player_name} (id={b.player_id})")
    
    # Check pitcher features
    print(f"\n  Home SP features present: {ctx.home_starter.pitcher_id in ctx.pitcher_features}")
    print(f"  Away SP features present: {ctx.away_starter.pitcher_id in ctx.pitcher_features}")
    
    if ctx.home_starter.pitcher_id in ctx.pitcher_features:
        pf = ctx.pitcher_features[ctx.home_starter.pitcher_id]
        print(f"  Home SP feature count: {len(pf)}")
        zero_feats = [k for k, v in pf.items() if v == 0.0]
        print(f"  Home SP zero-value features: {len(zero_feats)}/{len(pf)}")
        if zero_feats:
            print(f"    Zero features: {zero_feats[:10]}...")
    
    if ctx.away_starter.pitcher_id in ctx.pitcher_features:
        pf = ctx.pitcher_features[ctx.away_starter.pitcher_id]
        print(f"  Away SP feature count: {len(pf)}")
        zero_feats = [k for k, v in pf.items() if v == 0.0]
        print(f"  Away SP zero-value features: {len(zero_feats)}/{len(pf)}")
        if zero_feats:
            print(f"    Zero features: {zero_feats[:10]}...")
    
    # Now run prediction
    from algomlb.ui.utils import get_uranium_prediction
    model_prob, is_fallback = get_uranium_prediction(ctx)
    
    print(f"\n  🎯 Model home_win_prob: {model_prob:.4f} ({model_prob*100:.1f}%)")
    print(f"  📊 Is Elo fallback: {is_fallback}")
    
    if latest_odds:
        print(f"\n  --- EV CALCULATION ---")
        print(f"  h_implied (market): {h_implied:.4f} ({h_implied*100:.1f}%)")
        print(f"  model_prob (home):  {model_prob:.4f} ({model_prob*100:.1f}%)")
        edge = model_prob - h_implied
        print(f"  Edge (home):        {edge:+.4f} ({edge*100:+.1f}%)")
        
        # If COL is away, then the edge ON COL would be negative edge (away is better)
        away_model = 1.0 - model_prob
        away_implied = 1.0 - h_implied
        away_edge = away_model - away_implied
        print(f"\n  Away model prob:    {away_model:.4f} ({away_model*100:.1f}%)")
        print(f"  Away implied:       {away_implied:.4f} ({away_implied*100:.1f}%)")
        print(f"  Away edge:          {away_edge:+.4f} ({away_edge*100:+.1f}%)")
        
        # What selection does picks.py choose?
        selection = col_game.home_team if edge > 0 else col_game.away_team
        ev_pct = abs(edge)
        print(f"\n  🎲 Selection: {selection}")
        print(f"  📈 EV %: {ev_pct:.4f} ({ev_pct*100:.1f}%)")

    # Check the model features more deeply
    print(f"\n--- LAYER 7: MODEL FEATURE VECTOR ---")
    from algomlb.ml.model import MLBModel
    
    model_path = Path(".data/models/home_win_v1.0.joblib")
    if not model_path.exists():
        model_path = Path(".data/models/uranium_win_model.joblib")
    
    if model_path.exists():
        model = MLBModel.load(model_path)
        base_est = model.get_base_xgb_estimator()
        expected_features = (
            list(base_est.feature_names_in_)
            if hasattr(base_est, "feature_names_in_")
            else []
        )
        print(f"  Model expects {len(expected_features)} features")
        
        # Build the feature row the same way get_uranium_prediction does
        sp_keys = {f.replace("h_sp_", "") for f in expected_features if f.startswith("h_sp_")}
        bat_keys = {f.replace("h_bat_", "") for f in expected_features if f.startswith("h_bat_")}
        
        row = {}
        row.update(ctx.matchup_features)
        
        h_sp_id = ctx.home_starter.pitcher_id
        a_sp_id = ctx.away_starter.pitcher_id
        
        for prefix, pid in [("h_sp_", h_sp_id), ("a_sp_", a_sp_id)]:
            raw_feats = ctx.pitcher_features.get(pid, {})
            if "roll_re24" in raw_feats:
                raw_feats["roll_re24_x"] = raw_feats["roll_re24"]
                raw_feats["roll_re24_y"] = raw_feats["roll_re24"]
            mapped = 0
            for k in sp_keys:
                if k in raw_feats:
                    row[f"{prefix}{k}"] = float(raw_feats[k])
                    mapped += 1
            total_sp_keys = len(sp_keys)
            print(f"  {prefix}: mapped {mapped}/{total_sp_keys} keys")
        
        for prefix, lineup in [("h_bat_", ctx.home_lineup), ("a_bat_", ctx.away_lineup)]:
            agg = {}
            count = 0
            side = "home" if prefix == "h_bat_" else "away"
            for batter in lineup:
                feats = ctx.batter_features.get(batter.player_id, {})
                if not feats:
                    continue
                for k in bat_keys:
                    if k == f"roll_re24_{side}_re24_agg":
                        src_key = "roll_re24"
                    else:
                        src_key = k
                    if src_key in feats:
                        agg[k] = agg.get(k, 0.0) + float(feats[src_key])
                count += 1
            if count > 0:
                for k in agg:
                    row[f"{prefix}{k}"] = agg[k] / count
            print(f"  {prefix}: {count} batters contributed, {len(agg)} keys aggregated")
        
        X = pd.DataFrame([row])
        X = X.reindex(columns=expected_features, fill_value=0.0)
        
        missing = [f for f in expected_features if f not in row]
        filled_zero = [f for f in expected_features if row.get(f) == 0.0 or f not in row]
        
        print(f"\n  Features missing (filled with 0.0): {len(missing)}/{len(expected_features)}")
        if missing:
            print(f"    Missing features: {missing}")
        
        print(f"\n  Features that are zero: {len(filled_zero)}/{len(expected_features)}")
        if filled_zero:
            print(f"    Zero features: {filled_zero}")
        
        # Print the actual feature values
        print(f"\n  --- FULL FEATURE VECTOR ---")
        for f in expected_features:
            val = X[f].iloc[0]
            marker = "⚠️ ZERO" if val == 0.0 and f in missing else ""
            print(f"    {f}: {val:.6f} {marker}")
    else:
        print(f"  ❌ Model file not found!")

except Exception as e:
    import traceback
    print(f"  ❌ Error: {e}")
    traceback.print_exc()

session.close()
