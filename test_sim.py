import sys
sys.path.append("/home/opc/AlgoMLB/src")
from algomlb.db.session import get_session_factory
from algomlb.ml.monte_carlo.loader import MatchupLoader
from algomlb.ml.monte_carlo.engine import SimulationEngine
from algomlb.ml.monte_carlo.aggregator import SimulationAggregator

import joblib

session = get_session_factory()()
loader = MatchupLoader(session)
game_pk = 716867 # A random game, I'll query one first.

# Get a valid game_pk
import pandas as pd
games = pd.read_sql("SELECT game_pk FROM games LIMIT 1", session.connection())
game_pk = games.iloc[0]['game_pk']
print(f"Testing game_pk={game_pk}")

ctx = loader.load_matchup(game_pk)

model_path = "/home/opc/AlgoMLB/models/pa_outcome_v1.1.pkl"
try:
    pa_model = joblib.load(model_path)
except Exception as e:
    print(f"Could not load model: {e}")
    sys.exit(1)

engine = SimulationEngine(pa_model, seed=42)
results = engine.run_trials(ctx, trials=10)

agg = SimulationAggregator()
df = agg.aggregate_results(game_pk, 2026, results, ctx)

sum_r = df[df['stat_type'] == 'R']['mean'].sum()
print("Total Expected Runs (Batters):", sum_r)
for h_id in [b.player_id for b in ctx.home_lineup]:
    print(df[(df['stat_type']=='R') & (df['player_id']==h_id)]['mean'].values)

session.close()
