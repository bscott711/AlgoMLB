import numpy as np
import pandas as pd
from typing import Dict, List, TYPE_CHECKING
from algomlb.ml.monte_carlo.state import BatterSimState, PitcherSimState

if TYPE_CHECKING:
    from algomlb.ml.monte_carlo.loader import MatchupContext


class SimulationAggregator:
    """Reduces raw Monte Carlo trial dicts into a structured results DataFrame."""

    def aggregate_results(
        self,
        game_pk: int,
        season: int,
        trial_results: List[Dict[int, BatterSimState | PitcherSimState]],
        context: "MatchupContext",
    ) -> pd.DataFrame:
        """
        Collapses N trials into a single summary DataFrame ready for DB insertion.
        Includes Batter props, Pitcher props, and Team Win/Loss probabilities.
        """
        player_stats: Dict[int, Dict[str, List[float]]] = {}
        home_ids = {b.player_id for b in context.home_lineup}
        away_ids = {b.player_id for b in context.away_lineup}

        # Tracks team-level outcomes across trials
        team_wins = {"home": 0, "away": 0}

        # 1. Flip the data: from [Trials -> Players -> Stat] to [Player -> Stat -> TrialValues]
        for trial in trial_results:
            h_score = 0
            a_score = 0

            for player_id, state in trial.items():
                if player_id not in player_stats:
                    player_stats[player_id] = {
                        "H": [],
                        "HR": [],
                        "RBI": [],
                        "R": [],
                        "K_batter": [],
                        "W_batter": [],
                        "HRR": [],
                        "TB": [],
                        "K_pitcher": [],
                        "W_pitcher": [],
                        "Hits_allowed": [],
                        "Outs": [],
                    }

                # Batter stats
                if isinstance(state, BatterSimState):
                    ps = player_stats[player_id]
                    ps["H"].append(float(state.hits))
                    ps["HR"].append(float(state.hr))
                    ps["RBI"].append(float(state.rbi))
                    ps["R"].append(float(state.runs))
                    ps["K_batter"].append(float(state.strikeouts))
                    ps["W_batter"].append(float(state.walks))
                    ps["HRR"].append(float(state.hrr))
                    ps["TB"].append(float(state.total_bases))

                    # Accumulate score for win tracking
                    if player_id in home_ids:
                        h_score += state.runs
                    elif player_id in away_ids:
                        a_score += state.runs

                # Pitcher stats
                elif isinstance(state, PitcherSimState):
                    ps = player_stats[player_id]
                    ps["K_pitcher"].append(float(state.strikeouts))
                    ps["W_pitcher"].append(float(state.walks_allowed))
                    ps["Hits_allowed"].append(float(state.hits_allowed))
                    ps["Outs"].append(float(state.outs_recorded))

            # 2. Track Wins
            if h_score > a_score:
                team_wins["home"] += 1
            elif a_score > h_score:
                team_wins["away"] += 1

        # 3. Compute Probabilities
        records = []
        for player_id, stats in player_stats.items():
            for stat_name, values in stats.items():
                if not values:
                    continue

                # Map internal stat names to public Prop names
                public_stat = (
                    stat_name.replace("_batter", "")
                    .replace("_pitcher", "")
                    .replace("Hits_allowed", "H")
                )
                if stat_name == "Outs":
                    public_stat = "PO"

                arr = np.array(values)
                record = {
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
                records.append(record)

        # 4. Add Team Win Probability (PlayerID 0=Away, 1=Home for simplicity in this schema)
        n_trials = len(trial_results)
        if n_trials > 0:
            for side, player_idx in [("home", 1), ("away", 0)]:
                prob = team_wins[side] / n_trials
                records.append(
                    {
                        "game_pk": game_pk,
                        "season": season,
                        "player_id": player_idx,
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
                )

        return pd.DataFrame(records)
