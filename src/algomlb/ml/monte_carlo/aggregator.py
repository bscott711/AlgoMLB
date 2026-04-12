import numpy as np
import pandas as pd
from typing import Dict, List
from algomlb.ml.monte_carlo.state import BatterSimState, PitcherSimState


class SimulationAggregator:
    """Reduces raw Monte Carlo trial dicts into a structured results DataFrame."""

    def aggregate_results(
        self,
        game_pk: int,
        season: int,
        trial_results: List[Dict[int, BatterSimState | PitcherSimState]],
    ) -> pd.DataFrame:
        """
        Collapses N trials into a single summary DataFrame ready for DB insertion.
        """
        player_stats: Dict[int, Dict[str, List[float]]] = {}

        # 1. Flip the data: from [Trials -> Players -> Stat] to [Player -> Stat -> TrialValues]
        for trial in trial_results:
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
                # Pitcher stats
                elif isinstance(state, PitcherSimState):
                    ps = player_stats[player_id]
                    ps["K_pitcher"].append(float(state.strikeouts))
                    ps["W_pitcher"].append(float(state.walks_allowed))
                    ps["Hits_allowed"].append(float(state.hits_allowed))
                    ps["Outs"].append(float(state.outs_recorded))

        # 2. Compute Probabilities
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

        return pd.DataFrame(records)
