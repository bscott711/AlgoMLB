import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from algomlb.core.logger import logger
from algomlb.ml.monte_carlo.state import (
    GameState,
    BatterSimState,
    PitcherSimState,
    SimulationResult,
)
from algomlb.ml.monte_carlo.loader import MatchupContext
from algomlb.ml.monte_carlo.bullpen import BullpenManager
from algomlb.ml.monte_carlo.bullpen_orchestrator import BullpenOrchestrator
from algomlb.ml.hook_model import HookModel

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
    "delta_spin_rate_3g",
    "delta_extension_3g",
    "delta_fb_velo_3g",
    "roll_re24",
}

# All 12 valid ball-strike count states in baseball
COUNT_STATES = [
    "0-0",
    "0-1",
    "0-2",
    "1-0",
    "1-1",
    "1-2",
    "2-0",
    "2-1",
    "2-2",
    "3-0",
    "3-1",
    "3-2",
]

# Empirical MLB terminal count frequencies (2023-2024 seasons).
# These represent the fraction of plate appearances that END at each count.
# Source: aggregate Statcast data — ~190k PAs sampled.
TERMINAL_COUNT_WEIGHTS = {
    "0-0": 0.077,  # First-pitch outcomes
    "0-1": 0.065,
    "0-2": 0.060,
    "1-0": 0.072,
    "1-1": 0.085,
    "1-2": 0.120,
    "2-0": 0.038,
    "2-1": 0.070,
    "2-2": 0.138,
    "3-0": 0.022,
    "3-1": 0.053,
    "3-2": 0.200,
}


class SimulationEngine:
    """Core engine executing thousands of Markov-chain game trials using ML outcomes."""

    def __init__(
        self, pa_model: Any, hook_model: Optional[HookModel] = None, seed: int = 42
    ):
        self.pa_model = pa_model
        self.hook_model = hook_model
        # Explicit Random Generator for strict reproducibility
        self.rng = np.random.default_rng(seed)
        self.matchup_cache: Dict[Tuple, np.ndarray] = {}
        self._count_aware = False  # Set True after 3D precomputation

        # Dynamically load outcomes from the model's LabelEncoder classes
        # This ensures we support hbp, triple, etc. without hardcoding
        if hasattr(pa_model, "le") and pa_model.le is not None:
            self.outcome_map = list(pa_model.le.classes_)
        else:
            # Fallback to canonical order if no encoder exists
            self.outcome_map = [
                "double",
                "hbp",
                "home_run",
                "out_in_play",
                "single",
                "strikeout",
                "triple",
                "walk",
            ]

    def simulate_game(
        self, context: MatchupContext, bullpen_params: Optional[Dict[str, float]] = None
    ) -> SimulationResult:
        """Simulates a single 9-inning game trial and returns the final player states."""
        state = GameState()
        bp_manager = self._setup_bullpen_manager(context)
        bat_registry, pit_registry = self._init_registries(context)

        active_pitchers = {
            "home": context.home_starter.pitcher_id,
            "away": context.away_starter.pitcher_id,
        }

        # Initialize the Orchestrator for both sides
        all_profiles = context.home_reliever_profiles + context.away_reliever_profiles

        # Bullpen Params (platoon_advantage, fatigue_decay, min_fatigue_floor)
        bp_config = bullpen_params or {}
        orchestrator = BullpenOrchestrator(
            all_profiles,
            self.rng,
            platoon_advantage=bp_config.get("platoon_advantage", 1.12),
            fatigue_decay=bp_config.get("fatigue_decay", 0.003),
            min_fatigue_floor=bp_config.get("min_fatigue_floor", 0.82),
        )

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
                        orchestrator,
                        bat_registry,
                        pit_registry,
                        active_pitchers,
                        queues,
                        ptrs,
                    ):
                        return SimulationResult(
                            home_score=state.home_score,
                            away_score=state.away_score,
                            late_inning_runs=state.late_inning_runs,
                            player_states={**bat_registry, **pit_registry},
                            inning_count=state.inning,
                        )

            state.inning += 1
            if state.inning > 20:
                break

        return SimulationResult(
            home_score=state.home_score,
            away_score=state.away_score,
            late_inning_runs=state.late_inning_runs,
            player_states={**bat_registry, **pit_registry},
            inning_count=state.inning,
        )

    def _execute_pa(
        self,
        state,
        context,
        bp_manager,
        orchestrator,
        b_reg,
        p_reg,
        active_p,
        queues,
        ptrs,
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
                # 1.1 Use Orchestrator for intelligent selection
                # We need upcoming batters for platoon matching (next 3)
                upcoming_hands = []
                bat_side_lineup = (
                    context.away_lineup if state.top_half else context.home_lineup
                )
                for i in range(3):
                    idx = (ptrs[b_side] + i) % 9
                    upcoming_hands.append(bat_side_lineup[idx].hand)

                li = state.leverage_index if hasattr(state, "leverage_index") else 1.0
                next_p = orchestrator.select_next(
                    queues[p_side], upcoming_hands, li, state.inning
                )

                if next_p:
                    p_id = next_p
                    active_p[p_side] = p_id
                    # Remove from queue if it was in there
                    if p_id in queues[p_side]:
                        queues[p_side].remove(p_id)
                else:
                    # Fallback to FIFO
                    p_id = queues[p_side].pop(0)
                    active_p[p_side] = p_id

        # 2. Batter Resolution
        batter = (context.away_lineup if state.top_half else context.home_lineup)[
            ptrs[b_side]
        ]
        ptrs[b_side] = (ptrs[b_side] + 1) % 9

        # 3. Simulate count and sample count-conditional outcome
        self._simulate_count(state)
        outcome = self._sample_pa(
            batter.player_id,
            p_id,
            context,
            b_side,
            state.count_state,
            orchestrator,
            batter.hand,
            p_reg[p_id].pitches_thrown,
        )
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
        # Track which pitcher IDs are game starters so BullpenManager can
        # populate the is_starter feature correctly during inference.
        starter_ids = {
            context.home_starter.pitcher_id,
            context.away_starter.pitcher_id,
        }
        # Seed the bullpen RNG as a reproducible child of the engine RNG
        bp_seed = int(self.rng.integers(1 << 31))
        return BullpenManager(
            pd.concat([h_pen, a_pen]),
            context.manager_profiles,
            hook_model=self.hook_model,
            rng=np.random.default_rng(bp_seed),
            starter_ids=starter_ids,
        )

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
        count_state: str = "0-0",
        orchestrator: Optional[BullpenOrchestrator] = None,
        batter_hand: str = "R",
        pitches_thrown: int = 0,
    ) -> str:
        """
        Samples one outcome from the ML model using flat (count-marginalized)
        probabilities from the 2D cache.
        """
        probs = self.matchup_cache.get((batter_id, pitcher_id))

        if probs is None:
            raise RuntimeError(
                f"Failsafe triggered: No precomputed probabilities found for "
                f"matchup {batter_id} vs {pitcher_id}. "
                f"Cache miss indicates inference architecture failure."
            )

        # Apply Multipliers (Fatigue/Platoon)
        if orchestrator:
            k_adj, bb_adj, hr_adj = orchestrator.compute_platoon_fatigue_adjustment(
                pitcher_id, batter_hand, pitches_thrown
            )
            adj_probs = probs.copy()
            for i, outcome in enumerate(self.outcome_map):
                if outcome == "strikeout":
                    adj_probs[i] *= k_adj
                elif outcome == "walk":
                    adj_probs[i] *= bb_adj
                elif outcome == "home_run":
                    adj_probs[i] *= hr_adj

            p_sum = adj_probs.sum()
            if p_sum > 0:
                probs = adj_probs / p_sum
            else:
                logger.warning(
                    f"Probability collapse for {pitcher_id} vs {batter_id}. Using base probs."
                )

        return self.rng.choice(self.outcome_map, p=probs)

    def _simulate_count(self, state: GameState) -> None:
        """
        No-op: count information is now baked into the marginalized probabilities
        computed during _precompute_matchups(). The Markov count simulation was
        removed because the PA model's count features encode terminal counts
        (walks only at 3-ball, Ks only at 2-strike), making count-conditional
        lookups fundamentally incompatible with a mid-PA count simulator.
        """
        state.reset_count()

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
        elif outcome == "hbp":
            b.hbp += 1
            b.total_bases += 1
            p.walks_allowed += 1  # Standardizing on Walks Allowed for HBP in basic sims
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
        self,
        context: MatchupContext,
        trials: int = 10000,
        bullpen_params: Optional[Dict[str, float]] = None,
    ) -> List[SimulationResult]:
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
            all_trial_results.append(
                self.simulate_game(context, bullpen_params=bullpen_params)
            )
        logger.success(f"Simulation of {trials} trials complete.")
        return all_trial_results

    def _precompute_matchups(self, context):
        """
        Vectorized batch inference with strict feature alignment.

        When the model has count features (cnt_*), we compute probabilities
        at all 12 count states and **marginalize** them by weighting with
        real MLB terminal-count frequencies.  This produces a single, flat
        2D cache: (batter_id, pitcher_id) -> prob_array.

        Why marginalize instead of 3D lookup?
        The model was trained on *terminal* counts — walks only appear at
        3-ball counts, strikeouts only at 2-strike counts.  A Markov count
        simulator feeding those terminal counts back creates circular logic
        that collapses K/BB rates.  Marginalizing recovers realistic
        unconditional outcome distributions.
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

        # Detect if the model was trained with count features
        model_has_count = self._model_has_count_features()
        count_loop = COUNT_STATES if model_has_count else ["0-0"]

        # Helper to construct the feature matrix
        def build_matrix(batters, pitchers, batting_side):
            p_prefix = "pitcher_"
            b_prefix = "batter_"

            # --- SHIM SETUP: Calculate Team Averages ---
            def get_avg(team_batters):
                res = {}
                count = 0
                for b in team_batters:
                    feats = context.batter_features.get(b, {})
                    for k, v in feats.items():
                        res[k] = res.get(k, 0) + v
                    count += 1
                if count > 0:
                    for k in res:
                        res[k] /= count
                return res

            away_avg = get_avg(away_batters)
            home_avg = get_avg(home_batters)

            for b_id in batters:
                b_feats = context.batter_features.get(b_id, {})
                for p_id in pitchers:
                    p_feats = context.pitcher_features.get(p_id, {})

                    # Build base feature row (shared across all counts)
                    base_combined = {}
                    for k, v in b_feats.items():
                        base_combined[f"{b_prefix}{k}"] = v
                    for k, v in p_feats.items():
                        base_combined[f"{p_prefix}{k}"] = v

                    # Global matchup features
                    for k in ["elo_diff", "home_team_elo_pre", "away_team_elo_pre"]:
                        if k in context.matchup_features:
                            base_combined[k] = context.matchup_features[k]

                    # --- COMPATIBILITY SHIM (legacy features) ---
                    for k, v in home_avg.items():
                        base_combined[f"h_bat_{k}"] = v
                    for k, v in away_avg.items():
                        base_combined[f"a_bat_{k}"] = v

                    if "roll_re24" in p_feats:
                        base_combined["pitcher_roll_re24_x"] = p_feats["roll_re24"]
                        base_combined["pitcher_roll_re24_y"] = p_feats["roll_re24"]
                    if "roll_re24" in b_feats:
                        base_combined["batter_roll_re24_x"] = b_feats["roll_re24"]
                        base_combined["batter_roll_re24_y"] = b_feats["roll_re24"]

                    for f in [
                        "n_games_used",
                        "window_games",
                        "days_since_last_game",
                        "fatigue_index_7d",
                        "fatigue_index_14d",
                    ]:
                        if f in p_feats:
                            base_combined[f"pitcher_{f}"] = p_feats[f]
                        if f in b_feats:
                            base_combined[f"batter_{f}"] = b_feats[f]

                    # Build rows for each count state
                    for count_str in count_loop:
                        combined = base_combined.copy()
                        if model_has_count:
                            for cs in COUNT_STATES:
                                combined[f"cnt_{cs}"] = 1.0 if cs == count_str else 0.0
                        keys.append((b_id, p_id, count_str))
                        matchup_rows.append(combined)

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
                expected_features = self._get_expected_features()
                if expected_features is not None:
                    expected_list = list(expected_features)
                    X_batch = X_batch.reindex(columns=expected_list, fill_value=0.0)

                # Run Batch Prediction
                prob_matrix = actual_model.predict_proba(X_batch)

                if model_has_count:
                    # ── Count-marginalized aggregation ────────────────────
                    # The model was trained on terminal counts, so each
                    # count-conditional vector encodes which outcomes are
                    # "legally possible" at that terminal state.  We weight
                    # by real MLB terminal frequencies to recover realistic
                    # unconditional probabilities.
                    weight_vec = np.array(
                        [TERMINAL_COUNT_WEIGHTS[cs] for cs in COUNT_STATES]
                    )
                    weight_vec = weight_vec / weight_vec.sum()  # ensure sums to 1

                    n_counts = len(COUNT_STATES)
                    n_outcomes = prob_matrix.shape[1]
                    # keys has (b_id, p_id, count_str) triples; stride by n_counts
                    for i in range(0, len(keys), n_counts):
                        b_id = keys[i][0]
                        p_id = keys[i][1]
                        # Slice the 12 rows for this matchup
                        count_probs = prob_matrix[
                            i : i + n_counts
                        ]  # shape (12, n_outcomes)
                        # Weighted average across counts
                        marginalized = np.dot(
                            weight_vec, count_probs
                        )  # shape (n_outcomes,)
                        self.matchup_cache[(b_id, p_id)] = marginalized

                    logger.debug(
                        f"Successfully batch-inferred {len(keys) // n_counts} "
                        f"matchups (count-marginalized from {n_counts} states)."
                    )
                else:
                    # Flat model — store directly as 2D cache
                    for i, key in enumerate(keys):
                        b_id, p_id = key[0], key[1]
                        self.matchup_cache[(b_id, p_id)] = prob_matrix[i]
                    logger.debug(
                        f"Successfully batch-inferred {len(keys)} matchups (2D flat)."
                    )

            except Exception as e:
                logger.error(f"CRITICAL: Batch inference failed! {str(e)}")
                raise RuntimeError(
                    f"PA Model inference failed due to feature mismatch or XGBoost error: {e}"
                )
        else:
            raise ValueError(
                "The provided pa_model does not have a predict_proba method."
            )

    def _get_expected_features(self) -> Optional[list]:
        """Extract the ordered feature name list from the underlying XGBoost model."""
        actual_model = (
            self.pa_model.model if hasattr(self.pa_model, "model") else self.pa_model
        )
        base_estimator = None
        if hasattr(self.pa_model, "get_base_xgb_estimator"):
            base_estimator = self.pa_model.get_base_xgb_estimator()
        elif hasattr(actual_model, "model"):
            base_estimator = actual_model.model
        else:
            base_estimator = actual_model

        if (
            hasattr(base_estimator, "feature_names_in_")
            and base_estimator.feature_names_in_ is not None
        ):
            return list(base_estimator.feature_names_in_)
        elif (
            hasattr(base_estimator, "feature_names")
            and base_estimator.feature_names is not None
        ):
            return list(base_estimator.feature_names)
        elif hasattr(base_estimator, "get_booster"):
            return base_estimator.get_booster().feature_names
        return None

    def _model_has_count_features(self) -> bool:
        """Check if the loaded PA model was trained with cnt_* count features."""
        feature_names = self._get_expected_features()
        if feature_names:
            return any(f.startswith("cnt_") for f in feature_names)
        return False

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
            "hbp": (3.5, 1.5, 1),
        }

        mean, std, min_p = dist_params.get(outcome, (3.9, 1.7, 1))

        # Sample from the normal distribution managed by our reproducible RNG
        sampled = int(np.round(self.rng.normal(mean, std)))

        # Cap tails (avoid 20-pitch at-bats while ensuring physical limits)
        return max(min_p, min(sampled, 14))
