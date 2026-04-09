import numpy as np
from typing import Dict, Any, List
from .state import GameState
from .bullpen import BullpenManager


class SimulationEngine:
    """Core engine executing thousands of Markov-chain game trials."""

    def __init__(self, pa_model: Any, bullpen_manager: BullpenManager, seed: int = 42):
        self.pa_model = pa_model
        self.bullpen_manager = bullpen_manager
        # Explicit Random Generator for strict reproducibility
        self.rng = np.random.default_rng(seed)

        # The exact order must match the PAOutcomeModel's class_mapping_
        self.outcomes = [
            "strikeout",
            "walk",
            "hbp",
            "single",
            "double",
            "triple",
            "home_run",
            "out_in_play",
        ]

    def _simulate_pa(self) -> str:
        """Samples a single Plate Appearance outcome using the component model."""
        # For full implementation, this uses self.pa_model.predict_matchup(...)
        # Stubbing probabilities for the standalone loop verification
        probs = [0.22, 0.08, 0.01, 0.15, 0.05, 0.01, 0.03, 0.45]
        return self.rng.choice(self.outcomes, p=probs)

    def _simulate_game(self) -> Dict[str, int]:
        """Simulates a single 9-inning game."""
        state = GameState()

        # Play 9+ innings
        while state.inning <= 9 or state.home_score == state.away_score:
            for half in [True, False]:  # True = Top (Away), False = Bottom (Home)
                state.top_half = half
                state.outs = 0
                state.clear_bases()

                # Half-inning loop
                while state.outs < 3:
                    # In a full simulation, we fetch batter/pitcher state here
                    outcome = self._simulate_pa()
                    state.process_event(outcome)

                    # End game immediately if home team takes lead in bottom of 9th+
                    if (
                        not state.top_half
                        and state.inning >= 9
                        and state.home_score > state.away_score
                    ):
                        return {
                            "home_runs": state.home_score,
                            "away_runs": state.away_score,
                        }

            state.inning += 1

        return {"home_runs": state.home_score, "away_runs": state.away_score}

    def run_trials(self, trials: int = 10000) -> List[Dict[str, int]]:
        """Executes N Monte Carlo trials."""
        results = []
        for _ in range(trials):
            results.append(self._simulate_game())
        return results
