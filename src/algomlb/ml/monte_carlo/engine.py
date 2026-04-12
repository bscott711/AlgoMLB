import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from algomlb.core.logger import logger
from algomlb.ml.monte_carlo.state import GameState, BatterSimState, PitcherSimState, ManagerHookProfile
from algomlb.ml.monte_carlo.loader import MatchupContext
from algomlb.ml.monte_carlo.bullpen import BullpenManager


class SimulationEngine:
    """Core engine executing thousands of Markov-chain game trials using ML outcomes."""

    def __init__(self, pa_model: Any, seed: int = 42):
        self.pa_model = pa_model
        # Explicit Random Generator for strict reproducibility
        self.rng = np.random.default_rng(seed)
        self.matchup_cache: Dict[Tuple[int, int], np.ndarray] = {}

        # Top 8 classes from pa_outcome_v1.0 (ALPHABETICAL order for LabelEncoder)
        self.outcome_map = [
            "double",
            "home_run",
            "out_in_play",
            "out_in_play",  # mapping variety of outs
            "single",
            "strikeout",
            "triple",
            "walk",
        ]

    def simulate_game(
        self, context: MatchupContext
    ) -> Dict[int, BatterSimState | PitcherSimState]:
        """Simulates a single 9-inning game trial and returns the final player states."""
        state = GameState()

        # Manager & Bullpen Setup
        # Assuming a default empty bullpen if none provided to avoid crashes
        home_pen_df = pd.DataFrame([{"team_id": 0, "pitcher_id": p.pitcher_id, "availability_score": 1.0, "role": "mid_rel"} for p in context.home_relievers])
        away_pen_df = pd.DataFrame([{"team_id": 1, "pitcher_id": p.pitcher_id, "availability_score": 1.0, "role": "mid_rel"} for p in context.away_relievers])
        bp_manager = BullpenManager(pd.concat([home_pen_df, away_pen_df]), context.manager_profiles)

        # Local copies of states to accumulate stats in this trial
        batter_registry: Dict[int, BatterSimState] = {
            b.player_id: b.model_copy(deep=True)
            for b in (context.home_lineup + context.away_lineup)
        }
        pitcher_registry: Dict[int, PitcherSimState] = {
            context.home_starter.pitcher_id: context.home_starter.model_copy(deep=True),
            context.away_starter.pitcher_id: context.away_starter.model_copy(deep=True),
        }
        # Add relievers to registry
        for p in (context.home_relievers + context.away_relievers):
            pitcher_registry[p.pitcher_id] = p.model_copy(deep=True)

        # Tracking active pitchers
        active_pitchers = {
            "home": context.home_starter.pitcher_id,
            "away": context.away_starter.pitcher_id,
        }
        
        # Bullpen queue (simple pops for this simulation)
        queues = {
            "home": [p.pitcher_id for p in context.home_relievers],
            "away": [p.pitcher_id for p in context.away_relievers],
        }

        # Lineup pointers
        home_ptr = 0
        away_ptr = 0

        while state.inning <= 9 or state.home_score == state.away_score:
            for half in [True, False]:  # True = Top (Away), False = Bottom (Home)
                state.top_half = half
                state.outs = 0
                state.clear_bases()

                while state.outs < 3:
                    # Determine active pitcher and batter
                    pitching_side = "home" if half else "away"
                    batting_side = "away" if half else "home"
                    
                    pitcher_id = active_pitchers[pitching_side]
                    
                    # 1. Check for Hook
                    pitcher_state = pitcher_registry[pitcher_id]
                    mgr_id = context.home_manager_id if pitching_side == "home" else context.away_manager_id
                    
                    if bp_manager.should_hook(pitcher_state, state, mgr_id or 0):
                        if queues[pitching_side]:
                            new_pid = queues[pitching_side].pop(0)
                            active_pitchers[pitching_side] = new_pid
                            pitcher_id = new_pid
                            logger.debug(f"[{pitching_side}] Hooked pitcher at Inning {state.inning}. New: {pitcher_id}")

                    # 2. Current batter
                    if half:
                        batter = context.away_lineup[away_ptr]
                        away_ptr = (away_ptr + 1) % 9
                    else:
                        batter = context.home_lineup[home_ptr]
                        home_ptr = (home_ptr + 1) % 9

                    # 3. Infer PA Probability Distribution
                    outcome = self._sample_pa(batter.player_id, pitcher_id, context, batting_side)

                    # 4. Update States
                    scorer_ids = state.process_event(outcome, batter.player_id)

                    # 5. Attribute Stats
                    self._attribute_stats(
                        outcome,
                        batter.player_id,
                        pitcher_id,
                        scorer_ids,
                        batter_registry,
                        pitcher_registry,
                    )

                    # End game immediately if home team takes lead in bottom of 9th+
                    if (
                        not state.top_half
                        and state.inning >= 9
                        and state.home_score > state.away_score
                    ):
                        return {**batter_registry, **pitcher_registry}

            state.inning += 1
            if state.inning > 20:  # Infinite loop protection
                break

        return {**batter_registry, **pitcher_registry}

    def _sample_pa(
        self, batter_id: int, pitcher_id: int, context: MatchupContext, batting_side: str
    ) -> str:
        """Merges features and samples one outcome from the ML model."""
        # 1. Check cache for this specific Batter/Pitcher pair
        cache_key = (batter_id, pitcher_id)
        if cache_key in self.matchup_cache:
            probs = self.matchup_cache[cache_key]
        else:
            # 2. Get features
            b_feats = context.batter_features.get(batter_id, {})
            p_feats = context.pitcher_features.get(pitcher_id, {})

            # 3. Assemble feature vector with SIDE-AWARE prefixes
            # pa_outcome_v1.0 was trained on Uranium columns like h_sp_... and a_bat_...
            combined = {}
            p_prefix = "h_sp_" if batting_side == "away" else "a_sp_" # Pitcher side is opposite to batting side
            b_prefix = "a_bat_" if batting_side == "away" else "h_bat_"
            
            for k, v in b_feats.items():
                combined[f"{b_prefix}{k}"] = v
            for k, v in p_feats.items():
                combined[f"{p_prefix}{k}"] = v

            # 4. Predict probabilities
            if self.pa_model and hasattr(self.pa_model, "predict_proba"):
                try:
                    # Convert to DataFrame to allow XGBoost to align features by name
                    X_input = pd.DataFrame([combined])
                    prob_vector = self.pa_model.predict_proba(X_input)
                    probs = prob_vector[0]
                except Exception as e:
                    # fallback to a reasonable generic distribution if inference fails
                    probs = np.array([0.05, 0.03, 0.40, 0.05, 0.15, 0.22, 0.01, 0.09])
            else:
                probs = np.array([0.05, 0.03, 0.40, 0.05, 0.15, 0.22, 0.01, 0.09])

            # 5. Store in cache
            self.matchup_cache[cache_key] = probs

        # 6. Sample outcome
        return self.rng.choice(self.outcome_map, p=probs)

    def _attribute_stats(
        self,
        outcome: str,
        batter_id: int,
        pitcher_id: int,
        scored_ids: List[int],
        batter_registry: Dict[int, BatterSimState],
        pitcher_registry: Dict[int, PitcherSimState],
    ):
        b = batter_registry[batter_id]
        p = pitcher_registry[pitcher_id]

        b.pa_count += 1
        p.pitches_thrown += 4  # average

        if outcome == "strikeout":
            b.strikeouts += 1
            p.strikeouts += 1
            p.outs_recorded += 1
        elif outcome == "walk":
            b.walks += 1
            b.total_bases += 1
            p.walks_allowed += 1
        elif outcome == "single":
            b.hits += 1
            b.singles += 1
            b.total_bases += 1
            p.hits_allowed += 1
        elif outcome == "double":
            b.hits += 1
            b.doubles += 1
            b.total_bases += 2
            p.hits_allowed += 1
        elif outcome == "triple":
            b.hits += 1
            b.triples += 1
            b.total_bases += 3
            p.hits_allowed += 1
        elif outcome == "home_run":
            b.hits += 1
            b.hr += 1
            b.total_bases += 4
            p.hits_allowed += 1
        elif outcome == "out_in_play":
            p.outs_recorded += 1

        # Run and RBI attribution
        for rid in scored_ids:
            batter_registry[rid].runs += 1
            if rid != batter_id or outcome == "home_run":
                b.rbi += 1
            p.runs_allowed += 1

    def run_trials(
        self, context: MatchupContext, trials: int = 10000
    ) -> List[Dict[int, BatterSimState | PitcherSimState]]:
        """Executes N Monte Carlo trials and returns a list of resulting player states per trial."""
        if context is None:
            raise ValueError("SimulationEngine.run_trials received context=None. This indicates a failure in MatchupLoader.")
            
        all_trial_results = []
        logger.info(
            f"Starting {trials} Monte Carlo trials for game {context.game_pk}..."
        )
        for i in range(trials):
            if i > 0 and i % 1000 == 0:
                logger.debug(f"Completed {i} trials...")
            all_trial_results.append(self.simulate_game(context))
        logger.success(f"Simulation of {trials} trials complete.")
        return all_trial_results
