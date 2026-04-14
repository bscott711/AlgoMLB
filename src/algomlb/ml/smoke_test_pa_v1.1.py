import pandas as pd
import numpy as np
from pathlib import Path
from algomlb.cli.ml import _load_ml_data
from algomlb.ml.features import FeaturePipeline
from algomlb.ml.model import MLBModel
from algomlb.core.logger import logger
from algomlb.db.session import get_session_factory
from algomlb.ml.monte_carlo.engine import SimulationEngine
from algomlb.ml.monte_carlo.loader import MatchupLoader

def run_smoke_test():
    logger.info("🚀 Starting PA v1.1 Smoke Test...")
    
    session_factory = get_session_factory()
    engine = session_factory.kw["bind"]
    
    # 1. Load Minimal Data (2025 only)
    logger.info("Loading 2025 data for smoke test...")
    data = _load_ml_data(engine, "2025")
    
    pipeline = FeaturePipeline()
    
    # 2. Build Matrix (This is where the Bugs are hiding)
    try:
        logger.info("Building PA Matrix...")
        X, y = pipeline.build_pa_matrix(
            data["pas"],
            data["pitcher_gold"],
            data["batter_gold"],
            lineups_df=data["lineups"],
            elo_df=data["elo"],
            re24_df=data["re24"]
        )
        logger.success(f"PA Matrix Built Successfully! Shape: {X.shape}")
        logger.info(f"Sample Features: {list(X.columns)[:10]}")
    except Exception as e:
        logger.error(f"❌ Matrix Building FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Tiny Model Fit
    logger.info("Fitting tiny baseline model...")
    tiny_model = MLBModel(n_estimators=3, max_depth=3)
    tiny_model.fit(X, y)
    
    # 4. Simulation Engine Alignment Test
    try:
        logger.info("Testing Simulation Engine Alignment...")
        sim_engine = SimulationEngine(pa_model=tiny_model)
        
        # Load a sample game for the simulation
        # Using a very common game pk from 2025 if possible
        loader = MatchupLoader(session_factory())
        pas_sample = data["pas"]
        sample_pk = pas_sample["game_pk"].iloc[0] if "game_pk" in pas_sample.columns else pas_sample["game_id"].iloc[0]
        logger.info(f"Loading context for sample game {sample_pk}...")
        context = loader.load_matchup(sample_pk)
        
        # This is the 'precompute_matchups' call that fails in prod
        sim_engine._precompute_matchups(context)
        logger.success("✅ SMOKE TEST PASSED: Simulation Engine and Pipeline are ALIGNED!")
        
    except Exception as e:
        logger.error(f"❌ Simulation Alignment FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_smoke_test()
