import numpy as np
import pandas as pd
from typing import Dict, List, TYPE_CHECKING
from algomlb.ml.monte_carlo.state import BatterSimState, PitcherSimState, SimulationResult

if TYPE_CHECKING:
    from algomlb.ml.monte_carlo.loader import MatchupContext


class SimulationAggregator:
    """Reduces raw Monte Carlo trial dicts into a structured results DataFrame."""

    def aggregate_results(
        self,
        game_pk: int,
        season: int,
        trial_results: List[SimulationResult],
        context: "MatchupContext",
    ) -> pd.DataFrame:
        """
        Collapses N trials into a single summary DataFrame ready for DB insertion.
        Includes Batter props, Pitcher props, and Team Win/Loss probabilities.
        """
        player_stats: Dict[int, Dict[str, List[float]]] = {}
        home_ids = {b.player_id for b in context.home_lineup}
        away_ids = {b.player_id for b in context.away_lineup}
        team_wins = {"home": 0, "away": 0}

        # 1. Flip & Accumulate Data
        for trial in trial_results:
            self._process_trial(trial, player_stats, team_wins, home_ids, away_ids)

        # 2. Compute Probabilities & Records
        records = []
        for player_id, stats in player_stats.items():
            for s_name, values in stats.items():
                if values:
                    records.append(
                        self._create_record(player_id, s_name, values, game_pk, season)
                    )

        # 3. Add Team Win Probability
        n_trials = len(trial_results)
        if n_trials > 0:
            for side, p_idx in [("home", 1), ("away", 0)]:
                prob = team_wins[side] / n_trials
                records.append(
                    self._create_team_win_record(
                        side, p_idx, prob, n_trials, game_pk, season
                    )
                )

        return pd.DataFrame(records)

    def _process_trial(self, trial: SimulationResult, player_stats, team_wins, home_ids, away_ids):
        """Processes a single trial to accumulate player stats and determine game winner."""
        # 1. Process All Players from the flattened player_states dict
        for p_id, state in trial.player_states.items():
            if p_id not in player_stats:
                player_stats[p_id] = self._init_player_stats()

            ps = player_stats[p_id]
            if isinstance(state, BatterSimState):
                ps["H"].append(float(state.hits))
                ps["HR"].append(float(state.hr))
                ps["RBI"].append(float(state.rbi))
                ps["R"].append(float(state.runs))
                ps["K_batter"].append(float(state.strikeouts))
                ps["W_batter"].append(float(state.walks))
                ps["HRR"].append(float(state.hrr))
                ps["TB"].append(float(state.total_bases))
            elif isinstance(state, PitcherSimState):
                ps["K_pitcher"].append(float(state.strikeouts))
                ps["W_pitcher"].append(float(state.walks_allowed))
                ps["Hits_allowed"].append(float(state.hits_allowed))
                ps["Outs"].append(float(state.outs_recorded))

        # 2. Winning Team
        h_score = trial.home_score
        a_score = trial.away_score
        if h_score > a_score:
            team_wins["home"] += 1
        elif a_score > h_score:
            team_wins["away"] += 1

    def _init_player_stats(self):
        """Initializes the metrics dictionary for a player."""
        return {
            k: []
            for k in [
                "H", "HR", "RBI", "R", "K_batter", "W_batter",
                "HRR", "TB", "K_pitcher", "W_pitcher", "Hits_allowed", "Outs"
            ]
        }

    def _create_record(self, player_id, stat_name, values, game_pk, season):
        """Helper to create a standardized prop result record for a player/stat pair."""
        public_stat = (
            stat_name.replace("_batter", "")
            .replace("_pitcher", "")
            .replace("Hits_allowed", "H")
        )
        if stat_name == "Outs":
            public_stat = "PO"

        arr = np.array(values)
        return {
            "game_pk": game_pk,
            "season": season,
            "player_id": player_id,
            "stat_type": public_stat,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "prob_over_0_5": float(np.mean(arr > 0.5)),
            "prob_over_1_5": float(np.mean(arr > 1.5)),
            "prob_over_2_5": float(np.mean(arr > 2.5)),
            "prob_over_3_5": float(np.mean(arr > 3.5)),
            "prob_over_4_5": float(np.mean(arr > 4.5)),
            "trials": len(values),
        }

    def _create_team_win_record(self, side, p_idx, prob, n_trials, game_pk, season):
        """Helper to create a WIN probability record for a team."""
        return {
            "game_pk": game_pk,
            "season": season,
            "player_id": p_idx,
            "stat_type": "WIN",
            "mean": prob,
            "median": 1.0 if prob > 0.5 else 0.0,
            "p10": 0.0,
            "p90": 1.0,
            "prob_over_0_5": prob,
            "prob_over_1_5": 0.0,
            "prob_over_2_5": 0.0,
            "prob_over_3_5": 0.0,
            "prob_over_4_5": 0.0,
            "trials": n_trials,
        }
