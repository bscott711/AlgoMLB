import sys
sys.path.insert(0, "/home/opc/AlgoMLB/src")
from algomlb.ml.monte_carlo.state import BatterSimState, PitcherSimState, SimulationResult
from algomlb.ml.monte_carlo.prop_aggregator import PropAggregator

# Create some dummy trials
b1 = BatterSimState(player_id=100, hits=1, hr=0, total_bases=1, rbi=0, runs=0, strikeouts=1)
p1 = PitcherSimState(pitcher_id=200, strikeouts=3, walks_allowed=1, hits_allowed=2, outs_recorded=9)

trial1 = SimulationResult(
    home_score=1, away_score=0, late_inning_runs=0,
    player_states={100: b1, 200: p1},
    inning_count=9
)

trials = [trial1]
prop_engine = PropAggregator(trials)
player_stats = prop_engine.aggregate_player_stats()
records = prop_engine.calculate_prop_probabilities(player_stats)

print(records)
