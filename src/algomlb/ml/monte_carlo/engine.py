import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from algomlb.core.logger import logger
from algomlb.ml.monte_carlo.state import (
    GameState,
    BatterSimState,
    PitcherSimState,
)
from algomlb.ml.monte_carlo.loader import MatchupContext
from algomlb.ml.monte_carlo.bullpen import BullpenManager

# Define exactly what features belong to which entity based on training schema
BATTER_FEATURES = {
    "roll_pas",
    "roll_hits_per_pa",
    "roll_k_pct_batter",
    "roll_bb_pct_batter",
    "roll_barrel_pct",
    "roll_avg_launch_speed",
    "roll_avg_launch_angle",
    "roll_avg_batter_xwoba",
    "roll_batter_xwoba_shrunk",
    "ema_batter_xwoba_3g",
    "ema_batter_xwoba_7g",
    "ema_bat_speed_3g",
    "ema_attack_angle_3g",
    "ema_chase_pct_3g",
    "ema_iz_whiff_pct_3g",
    "std_batter_xwoba_15g",
    "std_launch_angle_15g",
    "seasonal_xwoba_vs_rh",
    "seasonal_xwoba_vs_lh",
    "roll_re24",
}

PITCHER_FEATURES = {
    "n_games_used",
    "days_since_last_game",
    "roll_pitches",
    "roll_strikes_pct",
    "roll_whiff_pct",
    "roll_k_pct",
    "roll_bb_pct",
    "roll_avg_release_speed",
    "roll_avg_pfx_x",
    "roll_avg_pfx_z",
    "roll_avg_pitcher_xwoba",
    "roll_pitcher_xwoba_shrunk",
    "ema_pitcher_xwoba_3g",
    "ema_pitcher_xwoba_7g",
    "ema_edge_pct_3g",
    "ema_velo_degradation_3g",
    "std_pitcher_xwoba_15g",
    "std_edge_pct_15g",
    "std_release_pos_z_15g",
    "fatigue_index_7d",
    "fatigue_index_14d",
    "delta_spin_rate_3g",
    "delta_extension_3g",
    "delta_fb_velo_3g",
    "roll_re24",
}


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
        bp_manager = self._setup_bullpen_manager(context)
        bat_registry, pit_registry = self._init_registries(context)

        active_pitchers = {
            "home": context.home_starter.pitcher_id,
            "away": context.away_starter.pitcher_id,
        }
        queues = {
            "home": [p.pitcher_id for p in context.home_relievers],
            "away": [p.pitcher_id for p in context.away_relievers],
        }

        ptrs = {"home": 0, "away": 0}

        while state.inning <= 9 or state.home_score == state.away_score:
            for half in [True, False]:  # True = Top (Away), False = Bottom (Home)
                state.top_half, state.outs = half, 0
                state.clear_bases()

                while state.outs < 3:
                    if self._execute_pa(
                        state,
                        context,
                        bp_manager,
                        bat_registry,
                        pit_registry,
                        active_pitchers,
                        queues,
                        ptrs,
                    ):
                        return {**bat_registry, **pit_registry}

            state.inning += 1
            if state.inning > 20:
                break

        return {**bat_registry, **pit_registry}

    def _execute_pa(
        self, state, context, bp_manager, b_reg, p_reg, active_p, queues, ptrs
    ) -> bool:
        """Executes a single plate appearance and returns True if a walk-off occurred."""
        p_side = "home" if state.top_half else "away"
        b_side = "away" if state.top_half else "home"
        p_id = active_p[p_side]

        # 1. Hook Check
        mgr_id = (
            context.home_manager_id if state.top_half else context.away_manager_id
        ) or 0
        if bp_manager.should_hook(p_reg[p_id], state, mgr_id):
            if queues[p_side]:
                p_id = queues[p_side].pop(0)
                active_p[p_side] = p_id

        # 2. Batter Resolution
        batter = (context.away_lineup if state.top_half else context.home_lineup)[
            ptrs[b_side]
        ]
        ptrs[b_side] = (ptrs[b_side] + 1) % 9

        # 3. Outcome & Attribution
        outcome = self._sample_pa(batter.player_id, p_id, context, b_side)
        scored_ids = state.process_event(outcome, batter.player_id)
        self._attribute_stats(outcome, batter.player_id, p_id, scored_ids, b_reg, p_reg)

        # 4. Walk-off Check
        return (
            not state.top_half
            and state.inning >= 9
            and state.home_score > state.away_score
        )

    def _setup_bullpen_manager(self, context: MatchupContext) -> BullpenManager:
        """Create a BullpenManager with initialized availability DataFrames."""
        h_pen = pd.DataFrame(
            [
                {
                    "team_id": 0,
                    "pitcher_id": p.pitcher_id,
                    "availability_score": 1.0,
                    "role": "mid_rel",
                }
                for p in context.home_relievers
            ]
        )
        a_pen = pd.DataFrame(
            [
                {
                    "team_id": 1,
                    "pitcher_id": p.pitcher_id,
                    "availability_score": 1.0,
                    "role": "mid_rel",
                }
                for p in context.away_relievers
            ]
        )
        return BullpenManager(pd.concat([h_pen, a_pen]), context.manager_profiles)

    def _init_registries(
        self, context: MatchupContext
    ) -> Tuple[Dict[int, BatterSimState], Dict[int, PitcherSimState]]:
        """Deep copy lineups and pitchers into fresh registries for a trial."""
        bat_reg = {
            b.player_id: b.model_copy(deep=True)
            for b in (context.home_lineup + context.away_lineup)
        }
        pit_reg = {
            context.home_starter.pitcher_id: context.home_starter.model_copy(deep=True),
            context.away_starter.pitcher_id: context.away_starter.model_copy(deep=True),
        }
        for p in context.home_relievers + context.away_relievers:
            pit_reg[p.pitcher_id] = p.model_copy(deep=True)
        return bat_reg, pit_reg
    def _sample_pa(
        self,
        batter_id: int,
        pitcher_id: int,
        context: MatchupContext,
        batting_side: str,
    ) -> str:
        """
        Samples one outcome from the ML model.
        Because of batch precomputation, this is now a lightning-fast O(1) dictionary lookup.
        """
        # 1. Retrieval
        cache_key = (batter_id, pitcher_id)

        # O(1) retrieval — NO pandas overhead in the inner loop
        probs = self.matchup_cache.get(cache_key)

        # Failsafe if a matchup wasn't precomputed (e.g. pinch hitter edge cases)
        if probs is None:
            probs = np.array([0.05, 0.03, 0.40, 0.05, 0.15, 0.22, 0.01, 0.09])

        # 2. Sample outcome
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
        
        # Replaced static +4 with probabilistic sampling based on outcome
        p.pitches_thrown += self._sample_pitch_count(outcome)

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
            raise ValueError(
                "SimulationEngine.run_trials received context=None. This indicates a failure in MatchupLoader."
            )

        all_trial_results = []
        logger.info(
            f"Starting {trials} Monte Carlo trials for game {context.game_pk}..."
        )
        
        # 🚀 OPTIMIZATION: Batch infer all probabilities BEFORE the loop begins
        self._precompute_matchups(context)
        
        all_trial_results = []
        for i in range(trials):
            if i > 0 and i % 1000 == 0:
                logger.debug(f"Completed {i} trials...")
            all_trial_results.append(self.simulate_game(context))
        logger.success(f"Simulation of {trials} trials complete.")
        return all_trial_results

    def _precompute_matchups(self, context):
        """
        Vectorized batch inference with strict feature alignment and role-specific mapping.
        """
        matchup_rows = []
        keys = []

        # Gather all IDs that could face each other
        away_batters = [b.player_id for b in context.away_lineup]
        home_batters = [b.player_id for b in context.home_lineup]

        home_pitchers = [context.home_starter.pitcher_id] + [
            p.pitcher_id for p in context.home_relievers
        ]
        away_pitchers = [context.away_starter.pitcher_id] + [
            p.pitcher_id for p in context.away_relievers
        ]

        # Helper to construct the feature matrix
        def build_matrix(batters, pitchers, batting_side):
            p_prefix = "h_sp_" if batting_side == "away" else "a_sp_"
            b_prefix = "a_bat_" if batting_side == "away" else "h_bat_"

            for b_id in batters:
                b_feats = context.batter_features.get(b_id, {})
                for p_id in pitchers:
                    p_feats = context.pitcher_features.get(p_id, {})

                    combined = {}
                    # 1. Apply Batter-specific features only
                    for k, v in b_feats.items():
                        if k in BATTER_FEATURES:
                            combined[f"{b_prefix}{k}"] = v

                    # 2. Apply Pitcher-specific features only
                    for k, v in p_feats.items():
                        if k in PITCHER_FEATURES:
                            combined[f"{p_prefix}{k}"] = v

                    # 3. Add Global Matchup Features (Rescue from context)
                    for k, v in context.matchup_features.items():
                        combined[k] = v

                    matchup_rows.append(combined)
                    keys.append((b_id, p_id))

        # Build Both Halves
        build_matrix(away_batters, home_pitchers, "away")
        build_matrix(home_batters, away_pitchers, "home")

        if not matchup_rows:
            logger.warning("No matchups generated for batch precomputation.")
            return

        X_batch = pd.DataFrame(matchup_rows)

        # Safely unwrap the base XGBoost model if wrapped
        actual_model = (
            self.pa_model.model if hasattr(self.pa_model, "model") else self.pa_model
        )

        if actual_model and hasattr(actual_model, "predict_proba"):
            try:
                # 🚀 Bulletproof Feature Alignment
                expected_features = None
                if hasattr(actual_model, "feature_names_in_"):
                    expected_features = actual_model.feature_names_in_
                elif (
                    hasattr(actual_model, "feature_names")
                    and actual_model.feature_names is not None
                ):
                    expected_features = actual_model.feature_names
                elif hasattr(actual_model, "get_booster"):
                    expected_features = actual_model.get_booster().feature_names

                if expected_features is not None:
                    expected_list = list(expected_features)
                    missing_cols = set(expected_list) - set(X_batch.columns)

                    if missing_cols:
                        # Final failsafe injecting NaN for totally missing columns
                        X_batch = X_batch.assign(**{col: np.nan for col in missing_cols})

                    # Filter exact columns in exact order
                    X_batch = X_batch[expected_list]

                # Run Batch Prediction
                prob_matrix = actual_model.predict_proba(X_batch)

                for i, key in enumerate(keys):
                    self.matchup_cache[key] = prob_matrix[i]

                logger.debug(f"Successfully batch-inferred {len(keys)} unique matchups.")

            except Exception as e:
                logger.error(f"CRITICAL: Batch inference failed! {str(e)}")
                raise RuntimeError(
                    f"PA Model inference failed due to feature mismatch or XGBoost error: {e}"
                )
        else:
            raise ValueError("The provided pa_model does not have a predict_proba method.")

    def _sample_pitch_count(self, outcome: str) -> int:
        """
        Samples a realistic pitch count conditional on the PA outcome using normal distributions.
        """
        # Baseline historical parameters: (mean, standard_deviation, minimum_possible)
        dist_params = {
            "strikeout": (4.8, 1.3, 3),
            "walk": (5.2, 1.2, 4),
            "single": (3.8, 1.8, 1),
            "double": (3.9, 1.8, 1),
            "triple": (4.0, 1.8, 1),
            "home_run": (4.1, 1.9, 1),
            "out_in_play": (3.3, 1.7, 1),
            "hbp": (3.5, 1.5, 1)
        }
        
        mean, std, min_p = dist_params.get(outcome, (3.9, 1.7, 1))

        # Sample from the normal distribution managed by our reproducible RNG
        sampled = int(np.round(self.rng.normal(mean, std)))
        
        # Cap tails (avoid 20-pitch at-bats while ensuring physical limits)
        return max(min_p, min(sampled, 14))
