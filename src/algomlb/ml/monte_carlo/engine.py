import numpy as np
from typing import Dict, List, Any
from algomlb.core.logger import logger
from algomlb.ml.monte_carlo.state import GameState, BatterSimState, PitcherSimState
from algomlb.ml.monte_carlo.loader import MatchupContext


class SimulationEngine:
    """Core engine executing thousands of Markov-chain game trials using ML outcomes."""

    def __init__(self, pa_model: Any, seed: int = 42):
        self.pa_model = pa_model
        # Explicit Random Generator for strict reproducibility
        self.rng = np.random.default_rng(seed)

        # Mapping for pa_outcome_v1.0 (Top 8 classes in encoded order)
        # Based on alphabetical order of common outcomes:
        # 1: double, 2: field_out, 3: grounded_into_double_play,
        # 4: home_run, 5: single, 6: strikeout, 7: triple, 8: walk
        # NOTE: This must be verified against the model's actual LabelEncoder.
        self.outcomes = [
            "double",
            "out_in_play",
            "out_in_play",  # mapping gidp to a generic out for stat tracking
            "home_run",
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

        # Local copies of states to accumulate stats in this trial
        batter_registry: Dict[int, BatterSimState] = {
            b.player_id: b.model_copy(deep=True)
            for b in (context.home_lineup + context.away_lineup)
        }
        pitcher_registry: Dict[int, PitcherSimState] = {
            context.home_starter.pitcher_id: context.home_starter.model_copy(deep=True),
            context.away_starter.pitcher_id: context.away_starter.model_copy(deep=True),
        }

        # Lineup pointers
        home_ptr = 0
        away_ptr = 0

        while state.inning <= 9 or state.home_score == state.away_score:
            for half in [True, False]:  # True = Top (Away), False = Bottom (Home)
                state.top_half = half
                state.outs = 0
                state.clear_bases()

                # Determine active pitcher and batter
                pitcher_id = (
                    context.home_starter.pitcher_id
                    if half
                    else context.away_starter.pitcher_id
                )

                while state.outs < 3:
                    # Current batter
                    if half:
                        batter = context.away_lineup[away_ptr]
                        away_ptr = (away_ptr + 1) % 9
                    else:
                        batter = context.home_lineup[home_ptr]
                        home_ptr = (home_ptr + 1) % 9

                    # 1. Infer PA Probability Distribution
                    outcome = self._sample_pa(batter.player_id, pitcher_id, context)

                    # 2. Update States
                    scorer_ids = state.process_event(outcome, batter.player_id)

                    # 3. Attribute Stats
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
        self, batter_id: int, pitcher_id: int, context: MatchupContext
    ) -> str:
        """Merges features and samples one outcome from the ML model."""
        # Simple stub for now while we refine the feature vector assembly
        # In production:
        # feats = combine(context.batter_features[b], context.pitcher_features[p], context.game_context)
        # probs = self.pa_model.predict_proba(feats)
        # return self.rng.choice(self.outcomes, p=probs)

        # For development verification:
        probs = [0.05, 0.40, 0.05, 0.03, 0.15, 0.22, 0.01, 0.09]
        return self.rng.choice(self.outcomes, p=probs)

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
