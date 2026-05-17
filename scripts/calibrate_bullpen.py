import optuna
import numpy as np
from datetime import date
from sqlalchemy import select, text
from algomlb.db.session import get_session_factory
from algomlb.db.models import GameResultORM
from algomlb.ml.monte_carlo.loader import MatchupLoader
from algomlb.ml.monte_carlo.engine import SimulationEngine
from algomlb.core.logger import logger

# Game Selection: 50 representative games from 2024
SAMPLE_SIZE = 50
TARGET_YEAR = 2024

def get_actual_late_inning_runs(game_pk):
    """Retrieves actual total runs scored in innings 7-9 from retrosheet_events."""
    session = get_session_factory()()
    query = text("""
        SELECT sum(runs)
        FROM retrosheet_events
        WHERE game_id = (SELECT retrosheet_game_id FROM game_manager_registry WHERE game_pk = :game_pk LIMIT 1)
          AND inning >= 7 AND pa_flag = 1
    """)
    res = session.execute(query, {"game_pk": game_pk}).scalar()
    session.close()
    return float(res) if res is not None else 0.0

from algomlb.ml.model import MLBModel
from algomlb.ml.hook_model import HookModel
from pathlib import Path

class BullpenCalibrationObjective:
    def __init__(self, game_pks, pa_model_version="v1.2"):
        self.game_pks = game_pks
        self.session_factory = get_session_factory()
        self.loader = MatchupLoader(self.session_factory())
        
        # Load Models
        logger.info("Loading models for calibration...")
        pa_model_path = Path(f".data/models/pa_outcome_{pa_model_version}.joblib")
        pa_model = MLBModel.load(pa_model_path)
        
        hook_path = Path(".data/models/hook_model_v1.0.joblib")
        hook_model = HookModel.load(hook_path) if hook_path.exists() else None
        
        self.engine = SimulationEngine(pa_model=pa_model, hook_model=hook_model)
        
        # Pre-load matchup contexts and PRECOMPUTE to save time during trials
        logger.info(f"Pre-loading and precomputing {len(game_pks)} games for calibration...")
        self.contexts = []
        self.actuals = []
        for g_pk in game_pks:
            try:
                ctx = self.loader.load_matchup(g_pk)
                actual = get_actual_late_inning_runs(g_pk)
                # Precompute matchups for this context to fill self.engine.matchup_cache
                self.engine._precompute_matchups(ctx)
                self.contexts.append(ctx)
                self.actuals.append(actual)
            except Exception as e:
                logger.warning(f"Skipping game {g_pk}: {e}")
        logger.success(f"Ready with {len(self.contexts)} games.")

    def __call__(self, trial: optuna.Trial):
        # Suggest parameters
        platoon_advantage = trial.suggest_float("platoon_advantage", 1.05, 1.25)
        fatigue_decay = trial.suggest_float("fatigue_decay", 0.001, 0.006)
        
        params = {
            "platoon_advantage": platoon_advantage,
            "fatigue_decay": fatigue_decay,
            "min_fatigue_floor": 0.82
        }
        
        errors = []
        if not self.contexts:
            logger.error("No games to simulate! Calibration aborted.")
            return 999.0

        for i, ctx in enumerate(self.contexts):
            # Reduced for speed: 20 simulations per game is enough to estimate mean late-inning runs
            n_sims = 25 
            trial_late_runs = []
            
            try:
                for _ in range(n_sims):
                    result = self.engine.simulate_game(ctx, bullpen_params=params)
                    trial_late_runs.append(result.late_inning_runs)
                
                avg_sim_late = np.mean(trial_late_runs)
                errors.append((avg_sim_late - self.actuals[i]) ** 2)
            except Exception as e:
                logger.error(f"Error simulating game {self.game_pks[i]}: {e}")
                continue
            
        if not errors:
            return 999.0
            
        rmse = np.sqrt(np.mean(errors))
        logger.info(f"Trial {trial.number}: RMSE={rmse:.4f}")
        return rmse

def run_calibration():
    session = get_session_factory()()
    from algomlb.domain import GameStatus
    # Sample 40 games from 2024 with valid results
    query = select(GameResultORM.game_id).where(
        GameResultORM.game_date >= date(2024, 4, 1),
        GameResultORM.status == GameStatus.COMPLETED
    ).limit(100) # Load 100, then we'll pick successful ones
    
    ids = session.execute(query).scalars().all()
    session.close()
    
    import random
    sampled_ids = random.sample(ids, min(len(ids), 40))
    # Convert to int for matchup loader
    sampled_pks = [int(gid) for gid in sampled_ids]
    
    objective = BullpenCalibrationObjective(sampled_pks)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    
    logger.success(f"Best RMSE: {study.best_value:.4f}")
    logger.info(f"Best Bullpen Params: {study.best_params}")
    
    # Persist to SQL
    from algomlb.db.models import SimulationConfigORM
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    
    session = get_session_factory()()
    try:
        stmt = pg_insert(SimulationConfigORM).values(
            config_key="bullpen_v1.0",
            config_value=study.best_params,
            version="v1.0",
            description="Tuned bullpen constants (platoon_advantage, fatigue_decay) for late-game run distro."
        ).on_conflict_do_update(
            index_elements=["config_key"],
            set_={
                "config_value": study.best_params,
                "updated_at": func.now()
            }
        )
        session.execute(stmt)
        session.commit()
        logger.success("✅ Successfully persisted bullpen calibration to SQL table: simulation_configs")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to persist to SQL: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    run_calibration()

# Note: I need to update engine.py to return total runs and late-inning runs 
# to make this objective efficient.
