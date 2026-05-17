import sys
from algomlb.db.session import get_session_factory
from algomlb.ml.monte_carlo.loader import MatchupLoader
from algomlb.ml.monte_carlo.engine import SimulationEngine
from algomlb.ml.monte_carlo.aggregator import SimulationAggregator
from algomlb.ml.model import MLBModel
from pathlib import Path

db = get_session_factory()()
try:
    loader = MatchupLoader(db)
    # game_pk = 825013
    ctx = loader.load_matchup(825013)
    if not ctx:
        print("Could not load matchup")
        sys.exit(1)
        
    model_path = Path(".data/models/pa_outcome_v1.5.joblib")
    if not model_path.exists():
        model_path = Path(".data/models/pa_outcome_v1.6.joblib")
        
    model = MLBModel.load(model_path)
    engine = SimulationEngine(pa_model=model)
    trials = engine.run_trials(ctx, trials=10)
    
    # check first trial player states
    if trials:
        t0 = trials[0]
        print("Player states keys:", list(t0.player_states.keys()))
        if t0.player_states:
            k = list(t0.player_states.keys())[0]
            v = t0.player_states[k]
            print(f"Sample state for {k}: {type(v)}")
            print("Fields:", v.model_dump() if hasattr(v, 'model_dump') else v)
            
            print(f"Has hits: {hasattr(v, 'hits')}, Has hr: {hasattr(v, 'hr')}")
            
    aggregator = SimulationAggregator()
    res = aggregator.aggregate_results(825013, 2026, trials, ctx)
    print("Columns:", res.columns)
    print("Stat types:", res['stat_type'].unique())
    print("Rows:", len(res))
    
except Exception:
    import traceback
    traceback.print_exc()
finally:
    db.close()
