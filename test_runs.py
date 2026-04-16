import sys
sys.path.append("/home/opc/AlgoMLB/src")
from algomlb.ml.monte_carlo.state import GameState, BatterSimState, PitcherSimState
from algomlb.ml.monte_carlo.engine import SimulationEngine
from typing import List

b_reg = {i: BatterSimState(player_id=i) for i in range(1, 10)}
p_reg = {100: PitcherSimState(pitcher_id=100)}

class MockEngine:
    def _attribute_stats(self, outcome, batter_id, pitcher_id, scored_ids, b_reg, p_reg):
        b = b_reg[batter_id]
        p = p_reg[pitcher_id]
        
        for rid in scored_ids:
            b_reg[rid].runs += 1
            if rid != batter_id or outcome == "home_run":
                b.rbi += 1
            p.runs_allowed += 1

engine = MockEngine()
state = GameState()
for i in range(1, 6):
    scored = state.process_event("single", i)
    engine._attribute_stats("single", i, 100, scored, b_reg, p_reg)

print(f"Team Score: {state.away_score}")
print(f"Batter Runs: {[b_reg[i].runs for i in range(1, 10)]}")
print(f"Batter RBI: {[b_reg[i].rbi for i in range(1, 10)]}")
